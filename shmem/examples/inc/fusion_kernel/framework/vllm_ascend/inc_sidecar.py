"""ProcessGroup-independent launcher control for the model-free INC sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time
from typing import Any

from inc_moe_runtime import (
    FusionPeMapping,
    PlanKey,
    SymmetricHeapPolicy,
    WorkerExecutorConfig,
)


@dataclass(frozen=True, slots=True)
class IncSidecarSpec:
    key: PlanKey
    config: WorkerExecutorConfig
    mapping: FusionPeMapping
    expert_owner: tuple[int, ...]
    expert_local_index: tuple[int, ...]
    bridge_library: str
    worker_ready_dir: str | None = None
    uid_path: str | None = None
    heap_policy: SymmetricHeapPolicy = SymmetricHeapPolicy()

    def validate(self) -> None:
        self.config.validate(self.key)
        family = self.key.family
        self.mapping.validate(family.worker_count)
        if self.mapping.inc_pe != self.config.inc_pe:
            raise ValueError("sidecar mapping/config disagree on inc_pe")
        if len(self.expert_owner) != family.expert_count or len(
            self.expert_local_index
        ) != family.expert_count:
            raise ValueError("sidecar expert placement length mismatch")
        if not self.bridge_library:
            raise ValueError("Torch bridge library path is required")
        if self.worker_ready_dir is not None and not self.worker_ready_dir:
            raise ValueError("worker_ready_dir must be a non-empty path")
        if self.uid_path is not None and not self.uid_path:
            raise ValueError("uid_path must be a non-empty path")


def _send(connection: Connection, state: str, **payload: Any) -> None:
    connection.send({"state": state, **payload})


def run_inc_sidecar(
    spec: IncSidecarSpec,
    uid: Any,
    control: Connection,
) -> None:
    """Child entrypoint; called with multiprocessing ``spawn``.

    The sidecar sends PREPARED immediately before entering the setup barrier,
    then READY after the persistent service is launched.  On STOP it stops the
    service first, sends STOPPING, enters the teardown barrier, and only then
    frees SHMEM and sends STOPPED.
    """
    session = None
    service = None
    try:
        spec.validate()
        import torch
        import torch_npu  # noqa: F401 - installs the PrivateUse1 backend

        torch.npu.set_device(spec.mapping.inc_device)
        bridge = str(Path(spec.bridge_library).expanduser().resolve())
        torch.ops.load_library(bridge)
        from torch_fusion_runtime import (
            FusionShmemSession,
            PreparedIncService,
            query_all_worker_plan_info,
        )

        infos = query_all_worker_plan_info(
            spec.key,
            spec.config,
            spec.expert_owner,
            spec.expert_local_index,
        )
        common = infos[0]
        heap_bytes = spec.heap_policy.heap_bytes(common)
        # Announce the exact allocation before joining the SHMEM bootstrap.
        # ``shmem.core.init`` is a W+1 rendezvous on the uid backend; sending
        # PREPARED after it would deadlock a launcher which starts worker PEs
        # only after learning the required heap size.
        _send(
            control,
            "PREPARED",
            symmetric_bytes=common.symmetric_bytes,
            heap_bytes=heap_bytes,
            abi=common.fusion_abi_version,
        )
        # vLLM constructs process groups and imports its worker stack before
        # calling Worker.load_model().  Joining ACLSHMEM immediately after
        # PREPARED can therefore exhaust the bootstrap timeout before any
        # worker PE exists.  Wait for one setup-time marker from every worker
        # and only then enter the W+1 rendezvous.  Markers contain no data-plane
        # state and are scoped to a unique launcher directory.
        if spec.worker_ready_dir is not None:
            ready_dir = Path(spec.worker_ready_dir)
            deadline = time.monotonic() + 300.0
            expected = [
                ready_dir / f"worker_{rank}.ready"
                for rank in range(spec.key.family.worker_count)
            ]
            while not all(path.is_file() for path in expected):
                if time.monotonic() >= deadline:
                    missing = [str(path) for path in expected if not path.is_file()]
                    raise TimeoutError(
                        "vLLM workers did not reach ACLSHMEM setup: "
                        + ", ".join(missing)
                    )
                time.sleep(0.01)
        if uid is None:
            if spec.uid_path is None:
                raise RuntimeError("sidecar requires uid or uid_path")
            uid_file = Path(spec.uid_path)
            deadline = time.monotonic() + 300.0
            while not uid_file.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"worker rank 0 did not publish UID: {uid_file}"
                    )
                time.sleep(0.01)
            uid = bytes.fromhex(uid_file.read_text().strip())
        session = FusionShmemSession(
            uid=uid,
            pe=spec.mapping.inc_pe,
            world_size=spec.key.family.worker_count + 1,
            heap_bytes=heap_bytes,
            symmetric_bytes=common.symmetric_bytes,
            device=f"npu:{spec.mapping.inc_device}",
        )
        service = PreparedIncService(
            spec.key,
            spec.config,
            spec.expert_owner,
            spec.expert_local_index,
            spec.mapping.worker_pes,
            session,
        )
        session.barrier()
        service.start()
        _send(control, "READY")

        command = control.recv()
        if command != "STOP":
            raise RuntimeError(f"unexpected sidecar command: {command!r}")
        service.close()
        service = None
        _send(control, "STOPPING")
        session.barrier()
        session.close()
        session = None
        _send(control, "STOPPED")
    except BaseException as exc:
        try:
            _send(
                control,
                "ERROR",
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except BaseException:
            pass
        # Do not enter an implicit collective from an error path.  The parent
        # decides whether the entire W+1 job can still shut down orderly.
        if service is not None:
            service.close()
        if session is not None:
            session.close()
        raise
    finally:
        control.close()


class IncSidecarController:
    """Parent-side non-destructive controller for one sidecar process."""

    def __init__(
        self,
        spec: IncSidecarSpec,
        uid: Any,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        spec.validate()
        ctx = context or mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        self._connection = parent
        self._process = ctx.Process(
            target=run_inc_sidecar,
            args=(spec, uid, child),
            name="single-inc-sidecar",
            daemon=False,
        )
        self._process.start()
        child.close()
        self._last_state = "STARTED"

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def wait_state(self, expected: str, timeout_s: float) -> dict[str, Any]:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self._connection.poll(timeout_s):
            raise TimeoutError(
                f"INC sidecar did not reach {expected}; last state "
                f"{self._last_state}, pid={self.pid}"
            )
        message = self._connection.recv()
        state = message.get("state")
        self._last_state = str(state)
        if state == "ERROR":
            raise RuntimeError(
                f"INC sidecar {message.get('error_type')}: "
                f"{message.get('message')}"
            )
        if state != expected:
            raise RuntimeError(
                f"expected INC sidecar {expected}, received {state}"
            )
        return message

    def request_stop(self) -> None:
        self._connection.send("STOP")

    def join(self, timeout_s: float) -> None:
        self._process.join(timeout_s)
        if self._process.is_alive():
            raise TimeoutError(
                "INC sidecar is still alive; do not terminate it while a "
                "push-only request or teardown barrier may be in flight"
            )
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"INC sidecar exited with code {self._process.exitcode}"
            )
        self._connection.close()

    def abort(self, timeout_s: float = 10.0) -> None:
        """Stop only this launcher's sidecar after setup has failed.

        Normal teardown must use STOP and the SHMEM barrier.  This method is
        intentionally reserved for a failed W+1 bootstrap where no collective
        teardown is possible.
        """
        if not self._process.is_alive():
            self._process.join(timeout_s)
            self._connection.close()
            return
        self._process.terminate()
        self._process.join(timeout_s)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout_s)
        self._connection.close()


__all__ = [
    "IncSidecarController",
    "IncSidecarSpec",
    "run_inc_sidecar",
]
