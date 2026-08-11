import unittest

from inc_moe_runtime import (
    BackendMode,
    CapacityPolicy,
    FusionPlanInfo,
    FusionPeMapping,
    PlanFamily,
    PlanKey,
    PreparedPlanCache,
    WorkerExecutorConfig,
    SymmetricHeapPolicy,
    WeightLayout,
)
from inc_sidecar import IncSidecarSpec


def family() -> PlanFamily:
    return PlanFamily(
        worker_count=4,
        worker_rank=1,
        hidden=2048,
        intermediate=8192,
        expert_count=64,
        topk=8,
        dtype="bf16",
        expert_placement_digest=11,
        hardware_profile_digest=12,
    )


class RuntimeConfigTests(unittest.TestCase):
    def test_backend_axes(self):
        self.assertFalse(BackendMode.NATIVE_VLLM.is_factorial)
        self.assertTrue(BackendMode.FUSED_INC.uses_inc)
        self.assertTrue(BackendMode.FUSED_INC.is_fused)
        self.assertFalse(BackendMode.SERIAL_SHMEM.uses_inc)
        self.assertEqual(WeightLayout.FUSION_ND.value, "fusion_nd")

    def test_arbitrary_tokens_use_capacity_buckets(self):
        policy = CapacityPolicy(minimum=64)
        self.assertEqual(policy.capacity_for(0), 64)
        self.assertEqual(policy.capacity_for(1), 64)
        self.assertEqual(policy.capacity_for(64), 64)
        self.assertEqual(policy.capacity_for(65), 128)
        self.assertEqual(policy.capacity_for(4097), 8192)

    def test_hbm_budget_is_explicit(self):
        with self.assertRaises(MemoryError):
            CapacityPolicy(maximum=1024).capacity_for(1025)

    def test_hot_lookup_never_creates(self):
        created = []
        cache = PreparedPlanCache(lambda key: created.append(key) or object())
        with self.assertRaises(RuntimeError):
            cache.lookup(family(), 257)
        first = cache.prepare(family(), 257)
        second = cache.prepare(family(), 300)
        self.assertIs(first, second)
        self.assertIs(first, cache.lookup(family(), 300))
        self.assertEqual(len(created), 1)

    def test_persistent_executor_uses_one_fixed_capacity(self):
        key = PlanKey(family(), 4096)
        config = WorkerExecutorConfig(
            live_aiv=48,
            live_aic=24,
            inc_pe=4,
            tokens_per_wave=32,
        )
        self.assertTrue(config.accepts(key, 0))
        self.assertTrue(config.accepts(key, 4096))
        self.assertFalse(config.accepts(key, 4097))

    def test_executor_config_rejects_unsafe_ring(self):
        key = PlanKey(family(), 64)
        with self.assertRaises(ValueError):
            WorkerExecutorConfig(
                live_aiv=48,
                live_aic=24,
                inc_pe=4,
                tokens_per_wave=32,
                executor_ring_size=1,
            ).validate(key)

    def test_service_layout_check_ignores_rank_local_workspace(self):
        first = FusionPlanInfo(1000, 200, 300, 8, 2, 6, 400)
        second = FusionPlanInfo(1000, 250, 300, 8, 3, 6, 400)
        first.require_same_service_layout(second)
        with self.assertRaises(RuntimeError):
            first.require_same_service_layout(
                FusionPlanInfo(1001, 200, 300, 8, 2, 6, 400)
            )

    def test_heap_and_pe_mapping_are_explicit_deployment_policy(self):
        info = FusionPlanInfo(600 << 20, 1, 1, 8, 2, 6, 1)
        policy = SymmetricHeapPolicy(
            minimum_bytes=512 << 20,
            reserve_bytes=64 << 20,
            alignment_bytes=2 << 20,
        )
        self.assertEqual(policy.heap_bytes(info), 664 << 20)
        FusionPeMapping((0, 1, 2, 3), 4, (1, 2, 3, 4), 0).validate(4)
        with self.assertRaises(ValueError):
            FusionPeMapping((0, 1, 2, 3), 4, (1, 2, 3, 3), 0).validate(4)

    def test_sidecar_spec_validates_without_importing_torch(self):
        key = PlanKey(family(), 4096)
        config = WorkerExecutorConfig(48, 24, 4, 256)
        mapping = FusionPeMapping((0, 1, 2, 3), 4, (1, 2, 3, 4), 0)
        owner = tuple(expert % 4 for expert in range(64))
        local = tuple(expert // 4 for expert in range(64))
        IncSidecarSpec(
            key, config, mapping, owner, local, "/opt/inc/bridge.so"
        ).validate()
        with self.assertRaises(ValueError):
            IncSidecarSpec(
                key,
                config,
                FusionPeMapping((0, 1, 2, 3), 4, (1, 2, 3, 4), 0),
                owner[:-1],
                local,
                "/opt/inc/bridge.so",
            ).validate()


if __name__ == "__main__":
    unittest.main()
