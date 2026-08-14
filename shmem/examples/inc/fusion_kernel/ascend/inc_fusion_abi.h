#ifndef INC_FUSION_ABI_H
#define INC_FUSION_ABI_H

#include <cstddef>
#include <cstdint>

namespace inc::fusion {

constexpr uint32_t kFusionMagic = 0x4655534eu; // FUSN
constexpr uint32_t kFusionAbiVersion = 13u;
constexpr uint32_t kFusionMinSlots = 3u;
constexpr uint32_t kFusionCacheLineBytes = 64u;
constexpr uint32_t kFusionMaxWorkers = 64u;
// Device stack arrays and the 4KiB INC trace reserve one line per live AIV.
// Refuse a wider device explicitly instead of silently overflowing; raising
// this cap only requires an ABI/workspace revision, not a data-path rewrite.
constexpr uint32_t kFusionMaxAiv = 64u;
constexpr uint32_t kFusionServiceMagic = 0x46535256u; // FSRV
constexpr uint32_t kFusionServiceAbiVersion = 1u;
constexpr uint32_t kFusionMinServiceRing = 2u;
constexpr uint32_t kFusionMaxServiceRing = 64u;

enum FusionRole : uint32_t {
    kFusionWorker = 1u,
    kFusionInc = 2u,
};

enum FusionDType : uint32_t {
    kFusionBf16 = 1u,
};

enum FusionFlags : uint32_t {
    kFusionPushOnly = 1u << 0,
    kFusionDynamicRoute = 1u << 1,
    kFusionTileReadyCompute = 1u << 2,
    kFusionConcurrentDispatchCombine = 1u << 3,
    kFusionWeightedIncReduce = 1u << 4,
    kFusionSerializeIncDc = 1u << 5,
    // Worker rank zero publishes waves/active lengths and a ticket directly
    // into the INC's symmetric command ring before entering Dispatch.
    kFusionRemoteIncService = 1u << 6,
    // Conservative measured baseline: do not begin the next token wave until
    // the previous result is returned, and do not expose this worker's
    // Combine traffic until all local GMM2 slices for the wave are complete.
    kFusionStrictSerialPipeline = 1u << 7,
    // Worker-direct ACLSHMEM baseline: workers exchange dispatch packets and
    // reduce combine packets without an INC PE.  Compute/layout stay shared
    // with the INC modes so the 2x2 comparison changes only communication.
    kFusionWorkerDirectShmem = 1u << 8,
    // vLLM stores each expert weight as contiguous [K,N].  Catlass can
    // consume that storage directly with a RowMajor B layout, avoiding a
    // setup-time [N,K] transpose/copy of every MoE layer.
    kFusionWeightBRowMajor = 1u << 9,
    // Wave-granular transport for the single-INC path.  Inputs, route
    // records, expert outputs and final rows are moved as contiguous bulk
    // regions; cache-line generation records are only control-plane
    // doorbells.  The bounded token-wave remains the outer pipeline unit.
    kFusionBulkWaveTransport = 1u << 10,
    // INC Combine returns the complete source-rank-ordered token tensor to
    // every worker.  vLLM can then skip its per-MoE-layer HCCL all-gather.
    // This mode is currently qualified for the dense top-k>=2 path.
    kFusionGlobalOutputFanout = 1u << 11,
};

enum FusionSlotError : uint32_t {
    kFusionSlotOk = 0u,
    kFusionSlotBadGeneration = 1u,
    kFusionSlotBadRoute = 2u,
    kFusionSlotTimeout = 3u,
    kFusionSlotCancelled = 4u,
};

enum FusionServiceRequestStatus : uint32_t {
    kFusionServiceRequestPending = 0u,
    kFusionServiceRequestSuccess = 1u,
    kFusionServiceRequestBadArgs = 2u,
    kFusionServiceRequestOperatorError = 3u,
};

enum FusionPacketKind : uint32_t {
    kFusionDispatchPayload = 1u,
    kFusionDispatchAssignment = 2u,
    kFusionDispatchLaneEnd = 3u,
    kFusionCombinePayload = 4u,
    kFusionCombineLaneEnd = 5u,
    kFusionResultPayload = 6u,
};

// Hardware-derived static resource partition. It deliberately contains no
// W/top-k/shape fields, so one policy is used by every workload on a device.
struct alignas(64) FusionResourcePlan {
    uint32_t live_aiv = 0u;
    uint32_t inc_dispatch_aiv = 0u;
    uint32_t inc_combine_aiv = 0u;
    uint32_t worker_dispatch_aiv = 0u;
    uint32_t worker_combine_aiv = 0u;
    uint32_t worker_compute_aiv = 0u;
    uint32_t policy_version = 1u;
    uint32_t live_aic = 0u;
    uint64_t fingerprint = 0u;
    uint64_t reserved64[3] = {};
};
static_assert(sizeof(FusionResourcePlan) == 64u, "resource ABI");

// A payload row is unique by (source rank, source token, destination rank).
// Multiple experts on that destination expand from one physical hidden row.
struct alignas(32) FusionDispatchRow {
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    uint32_t destination_rank = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
    uint32_t wave = 0u;
    uint64_t payload_byte_offset = 0u;
};
static_assert(sizeof(FusionDispatchRow) == 32u, "dispatch row ABI");

// Combine consumes one independent contribution per logical expert instance.
// The worker must not pre-reduce different local experts into one token row.
struct alignas(32) FusionExpertAssignment {
    uint32_t dispatch_row = 0u;
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t destination_token = 0u;
    uint32_t weight_bits = 0u;
    uint32_t wave = 0u;
    // Expert-grouped row on the destination worker. This is distinct from
    // destination_token, which remains the source token reduced by Combine.
    uint32_t destination_row = 0u;
};
static_assert(sizeof(FusionExpertAssignment) == 32u, "assignment ABI");

// Each row has one writer, but adjacent destination rows can be owned by
// different Dispatch RX AIVs. Keep them on distinct cache lines so a DCCI
// writeback cannot lose the neighbour's concurrent scalar metadata stores.
struct alignas(64) FusionReceivedAssignment {
    uint32_t source_rank = 0u;
    uint32_t source_token = 0u;
    uint32_t expert_id = 0u;
    uint32_t local_expert = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t destination_row = 0u;
    uint32_t wave = 0u;
};
static_assert(sizeof(FusionReceivedAssignment) == 64u,
              "received assignment ABI");

// Outer scheduler unit. Inner expert/tile readiness is described by the
// assignment range and does not add a whole-wave compute barrier.
struct alignas(64) FusionWaveDesc {
    uint64_t generation = 0u;
    uint32_t token_begin = 0u;
    uint32_t token_count = 0u;
    uint32_t dispatch_row_begin = 0u;
    uint32_t dispatch_row_count = 0u;
    uint32_t assignment_begin = 0u;
    uint32_t assignment_count = 0u;
    uint32_t slot = 0u;
    uint32_t activation_waves = 1u;
    uint32_t reserved32[4] = {};
    uint64_t reserved64 = 0u;
};
static_assert(sizeof(FusionWaveDesc) == 64u, "wave ABI");

// Every state owns a cache line. Reuse is legal only after release_generation
// reaches the previous generation. This is the protection against arbitrary
// Dispatch/Combine timing, cancellation and cross-layer buffer reuse.
struct alignas(64) FusionSlotState {
    uint64_t claim_generation = 0u;
    uint64_t dispatch_generation = 0u;
    uint64_t compute_generation = 0u;
    uint64_t combine_generation = 0u;
    uint64_t output_generation = 0u;
    uint64_t release_generation = 0u;
    uint32_t error = kFusionSlotOk;
    uint32_t owner = 0u;
    uint64_t reserved = 0u;
};
static_assert(sizeof(FusionSlotState) == 64u, "slot ABI");

struct alignas(64) FusionLaneTrace {
    uint64_t start_cycle = 0u;
    uint64_t end_cycle = 0u;
    uint32_t role = 0u;
    uint32_t lane = 0u;
    uint64_t reserved64[5] = {};
};
static_assert(sizeof(FusionLaneTrace) == 64u, "lane trace ABI");

// Local-to-INC descriptor ring.  The host publishes descriptor metadata and
// `ready` last on a submission stream.  Every persistent AIV consumes requests
// in ticket order; `complete` is published only after all INC Dispatch and
// Combine lanes have left the request.  A ring slot is therefore safe to
// reuse exactly when complete == its previous ready ticket.
struct alignas(64) FusionServiceDescriptor {
    uint64_t ready = 0u;
    uint64_t complete = 0u;
    uint64_t device_args = 0u;
    uint64_t request_id = 0u;
    uint32_t status = kFusionServiceRequestPending;
    uint32_t operator_error = kFusionSlotOk;
    uint64_t reserved64[3] = {};
};
static_assert(sizeof(FusionServiceDescriptor) == 64u,
              "service descriptor ABI");

// Dynamic request data is committed as one contiguous SDMA record before the
// descriptor ready word.  Keeping all independently changing fields in one
// transfer avoids back-to-back tiny NBI submissions while preserving a
// ready-last acquire/release edge for the persistent INC.
struct alignas(64) FusionRemoteRequestHeader {
    uint64_t operation_generation = 0u;
    uint64_t service_ticket = 0u;
    uint64_t request_id = 0u;
    uint32_t flags = 0u;
    uint32_t wave_count = 0u;
    uint32_t worker_count = 0u;
    uint32_t bytes = 0u;
    uint64_t reserved64[3] = {};
};
static_assert(sizeof(FusionRemoteRequestHeader) == 64u,
              "remote request header ABI");

// The first cache line contains all live control words so host polling and
// device stop checks require one DCCI. `lane_progress` points to live_aiv
// cache lines, one exclusive writer per AIV, and acts as the request barrier.
struct alignas(64) FusionServiceControl {
    uint32_t magic = kFusionServiceMagic;
    uint32_t abi_version = kFusionServiceAbiVersion;
    uint32_t ring_size = 0u;
    uint32_t live_aiv = 0u;
    uint64_t descriptors = 0u;
    uint64_t lane_progress = 0u;
    uint64_t completed_sequence = 0u;
    uint64_t stop = 0u;
    uint64_t service_error = 0u;
    uint64_t reserved0 = 0u;
    uint64_t reserved64[8] = {};
};
static_assert(sizeof(FusionServiceControl) == 128u,
              "service control ABI");

// Fixed-depth transport packet. Payload is split into transport_tile_bytes
// chunks, so queue memory is independent of the total token count. `credit`
// is written by the consumer only after the payload has been consumed.
struct alignas(64) FusionPacketHeader {
    // `ready` is a two-part commit: high 32 bits are the wave generation and
    // low 32 bits are packet_ordinal+1. Metadata is put and quieted before
    // this word is published, so a consumer can never accept a torn header.
    // `credit` returns the exact observed commit after payload consumption.
    uint64_t ready = 0u;
    uint32_t wave = 0u;
    uint32_t source_rank = 0u;
    uint32_t destination_rank = 0u;
    uint32_t source_token = 0u;
    uint32_t destination_token = 0u;
    uint32_t expert_id = 0u;
    uint32_t route_ordinal = 0u;
    uint32_t weight_bits = 0u;
    uint32_t payload_bytes = 0u;
    uint32_t payload_offset = 0u;
    uint32_t kind = 0u;
    uint32_t reserved = 0u;
    uint64_t producer_reserved = 0u;
    // Consumer-only line. Keeping credit away from producer-owned ready and
    // metadata prevents either side's DCCI writeback from losing the other
    // side's concurrent 8-byte update.
    uint64_t credit = 0u;
    uint64_t credit_reserved[7] = {};
};
static_assert(offsetof(FusionPacketHeader, credit) == 64u,
              "credit must own a cache line");
static_assert(sizeof(FusionPacketHeader) == 128u, "packet ABI");

// One exclusive writer owns a control record at a time.  Counts describe the
// dynamic prefix in the corresponding bounded wave arena and generation is
// published last through a full-cacheline RMA.  This keeps arbitrary routing
// and uneven rank lengths without returning to one doorbell per hidden row.
struct alignas(64) FusionBulkControl {
    uint64_t generation = 0u;
    uint32_t count0 = 0u;
    uint32_t count1 = 0u;
    uint64_t bytes0 = 0u;
    uint64_t bytes1 = 0u;
    uint64_t reserved64[4] = {};
};
static_assert(sizeof(FusionBulkControl) == 64u, "bulk control ABI");

struct alignas(64) FusionSymmetricLayout {
    uint64_t dispatch_header_off = 0u;
    uint64_t dispatch_payload_off = 0u;
    uint64_t combine_header_off = 0u;
    uint64_t combine_payload_off = 0u;
    uint64_t dispatch_result_header_off = 0u;
    uint64_t dispatch_result_payload_off = 0u;
    uint64_t control_off = 0u;
    uint64_t total_bytes = 0u;
    uint32_t queue_lanes = 0u;
    uint32_t queue_depth = 0u;
    uint32_t packet_bytes = 0u;
    uint32_t reserved32 = 0u;
    uint64_t combine_result_header_off = 0u;
    uint64_t combine_result_payload_off = 0u;
    uint64_t reserved64[4] = {};
};
static_assert(sizeof(FusionSymmetricLayout) == 128u, "symmetric ABI");

// Cross-process INC command transport lives inside the SHMEM symmetric heap.
// Every offset is relative to symmetric_base. Args/waves/active lengths are
// ring-indexed; worker_pes is one setup-time vector shared by all requests.
struct alignas(64) FusionRemoteServiceLayout {
    uint64_t control_off = 0u;
    uint64_t descriptors_off = 0u;
    uint64_t args_off = 0u;
    uint64_t waves_off = 0u;
    uint64_t active_token_counts_off = 0u;
    uint64_t worker_pes_off = 0u;
    uint64_t lane_progress_off = 0u;
    // [ring_size][worker_count] cache lines, one exclusive worker writer per
    // line. The INC starts a ticket only after every rank published it.
    uint64_t worker_ready_off = 0u;
    // [ring_size][request_stride]. Each record contains one header followed
    // by waves and active-token counts; waves_off/active_token_counts_off are
    // the corresponding addresses inside record zero.
    uint64_t request_off = 0u;
    uint64_t total_bytes = 0u;
    uint32_t ring_size = 0u;
    uint32_t request_stride = 0u;
    uint32_t request_bytes = 0u;
    uint32_t reserved32[9] = {};
};
static_assert(sizeof(FusionRemoteServiceLayout) == 128u,
              "remote service layout ABI");

struct alignas(64) FusionWorkspaceLayout {
    uint64_t slot_state_off = 0u;
    uint64_t dispatch_ring_off = 0u;
    uint64_t grouped_input_off = 0u;
    uint64_t gate_up_off = 0u;
    uint64_t activation_off = 0u;
    uint64_t expert_output_off = 0u;
    uint64_t combine_ring_off = 0u;
    uint64_t final_output_off = 0u;
    uint64_t expert_ready_off = 0u;
    uint64_t tile_ready_off = 0u;
    uint64_t trace_off = 0u;
    uint64_t total_bytes = 0u;
    uint64_t dispatch_slot_bytes = 0u;
    uint64_t assignment_slot_bytes = 0u;
    uint64_t output_slot_bytes = 0u;
    uint64_t assignment_meta_off = 0u;
};
static_assert(sizeof(FusionWorkspaceLayout) == 128u, "workspace ABI");

// Pointer fields are integer addresses so this host/device ABI does not depend
// on GM_ADDR typedefs or the framework language binding.
struct alignas(64) FusionKernelArgs {
    uint32_t magic = kFusionMagic;
    uint32_t abi_version = kFusionAbiVersion;
    uint32_t role = kFusionWorker;
    uint32_t flags = 0u;
    uint32_t rank = 0u;
    uint32_t worker_count = 0u;
    uint32_t inc_pe = 0u;
    uint32_t dtype = kFusionBf16;
    uint32_t hidden = 0u;
    uint32_t intermediate = 0u;
    uint32_t expert_count = 0u;
    uint32_t local_expert_count = 0u;
    uint32_t topk = 0u;
    uint32_t token_count = 0u;
    uint32_t tokens_per_wave = 0u;
    uint32_t wave_count = 0u;
    uint32_t slot_count = 0u;
    uint32_t activation_waves = 1u;
    uint32_t transport_tile_bytes = 0u;
    uint32_t compute_tile_rows = 0u;
    uint32_t ready_producers = 0u;
    uint32_t spin_cap = 0u;
    uint32_t reserved32[2] = {};
    uint64_t operation_generation = 0u;
    // Zero disables cross-process publication. Prepared worker executors set
    // a contiguous ticket only when kFusionRemoteIncService is enabled.
    uint64_t service_ticket = 0u;
    uint64_t request_id = 0u;
    // INC-only pointer to the immutable ready-committed request header for
    // this service-ring slot. Worker kernels leave it null and use the inline
    // fields above.
    uint64_t remote_request = 0u;
    uint64_t symmetric_base = 0u;
    uint64_t input = 0u;
    uint64_t output = 0u;
    uint64_t w13 = 0u;
    uint64_t w2 = 0u;
    uint64_t dispatch_rows = 0u;
    uint64_t assignments = 0u;
    uint64_t waves = 0u;
    uint64_t expert_owner = 0u;
    uint64_t expert_local_index = 0u;
    uint64_t worker_pes = 0u;
    // Optional [worker_count] uint32_t active-token vector. A null pointer
    // preserves the uniform prepared-capacity behavior for legacy callers.
    uint64_t active_token_counts = 0u;
    uint64_t group_lists = 0u;
    uint64_t workspace = 0u;
    uint64_t system_workspace = 0u;
    uint64_t ffts_addr = 0u;
    FusionResourcePlan resources{};
    FusionSymmetricLayout symmetric_layout{};
    FusionRemoteServiceLayout remote_service{};
    FusionWorkspaceLayout layout{};
};
static_assert(sizeof(FusionKernelArgs) % 64u == 0u, "kernel args ABI");

} // namespace inc::fusion

#endif
