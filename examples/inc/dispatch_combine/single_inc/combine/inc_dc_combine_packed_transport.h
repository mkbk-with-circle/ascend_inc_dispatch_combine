#ifndef INC_DC_COMBINE_PACKED_TRANSPORT_H
#define INC_DC_COMBINE_PACKED_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "inc_dc_combine_upload_transport.h"
#include "inc_dc_combine_symmetric_layout.h"

namespace inc {
namespace dc {

// BW05: per-(src,lane) remote SPSC + per-(dst,lane) producer state (formal packed path).
constexpr uint32_t kCombineBw05LaneCount = kCombineFixedUploadLaneCount;
constexpr uint32_t kCombineBw05MaxGroupSize = 8u;
constexpr uint32_t kCombineBw05MaxWorkerChannels = kCombineBw05MaxGroupSize * kCombineBw05LaneCount; // 64
constexpr uint32_t kCombineBw05ChannelCapacity = 64u;
// Production RX consumers: 1 = true bid0-only fast path (no RX progress / reduce_done alias).
// Multi-RX (>1) is intentionally unsupported until a dedicated rx_progress[] lands.
constexpr uint32_t kCombineBw05RxAivCount = 1u;
constexpr uint32_t kCombineBw05MinPackedPutBytes = 64u * 1024u;
constexpr uint32_t kCombineBw05MaxPackedPutBytes = 256u * 1024u;
// PERF1-A: double-buffer lane staging so multiple putmem_nbi can share one quiet
// before putmem_signal (payload/desc must be visible before tail). Depth=1 forces
// per-put quiet to protect staging source lifetime.
constexpr uint32_t kCombineBw05StagingDepth = 1u; // PERF1-A device coalesce deferred; keep depth=1
constexpr uint32_t kCombineBw05LayoutSchema = 3u;
constexpr uint32_t kCombineBw05MaxSwitchBlocks = 32u;
constexpr uint32_t kCombineBw05CacheLine = 64u;

inline uint32_t CombineBw05ProducerStateIndex(uint32_t dst_local_idx, uint32_t lane_id, uint32_t lane_count)
{
    return dst_local_idx * lane_count + lane_id;
}

inline uint32_t CombineBw05RemoteChannelIndex(uint32_t src_worker_rank, uint32_t lane_id, uint32_t lane_count)
{
    return src_worker_rank * lane_count + lane_id;
}

inline uint32_t CombineBw05DstLocalIndex(uint32_t target_switch_pe, uint32_t group_size)
{
    if (group_size == 0) {
        return 0;
    }
    if (target_switch_pe >= group_size) {
        return target_switch_pe - group_size;
    }
    return target_switch_pe % group_size;
}

// Local producer state (dst,lane): one Worker AIV owns each slot — exclusive 64B line.
struct alignas(64) IncDcCombineBw05ProducerHeader {
    uint32_t tail = 0;
    uint32_t credited_head = 0;
    uint32_t capacity = kCombineBw05ChannelCapacity;
    uint32_t published = 0;
    uint32_t published_payload_bytes = 0;
    uint32_t last_ack_epoch = 0; // ack_epoch == epoch+1 (32-bit signal)
    uint32_t overflow = 0;
    uint32_t pad0 = 0;
    uint8_t pad[32]{};
};
static_assert(sizeof(IncDcCombineBw05ProducerHeader) == 64, "BW05 producer header must be 64B");
static_assert(alignof(IncDcCombineBw05ProducerHeader) >= 64, "BW05 producer header align");

// Producer-only remote cursor (Worker putmem_signal writes tail). Exclusive 64B line.
struct alignas(64) IncDcCombineBw05ProducerCursorLine {
    uint32_t tail = 0;
    uint32_t capacity = kCombineBw05ChannelCapacity; // read-only on Switch
    uint32_t pad[14]{};
};
static_assert(sizeof(IncDcCombineBw05ProducerCursorLine) == 64, "producer cursor line 64B");
static_assert(alignof(IncDcCombineBw05ProducerCursorLine) >= 64, "producer cursor align");

// Consumer-only remote cursor (Switch writes head/consumed). Exclusive 64B line.
struct alignas(64) IncDcCombineBw05ConsumerCursorLine {
    uint32_t head = 0;
    uint32_t consumed = 0;
    uint32_t overflow = 0;
    uint32_t capacity = kCombineBw05ChannelCapacity;
    uint32_t pad[12]{};
};
static_assert(sizeof(IncDcCombineBw05ConsumerCursorLine) == 64, "consumer cursor line 64B");
static_assert(alignof(IncDcCombineBw05ConsumerCursorLine) >= 64, "consumer cursor align");

// Host-protocol convenience view (not a device-memory layout unit).
struct IncDcCombineBw05RemoteChannelHeader {
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t capacity = kCombineBw05ChannelCapacity;
    uint32_t consumed = 0;
    uint32_t overflow = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};
using IncDcCombineBw05ChannelHeader = IncDcCombineBw05RemoteChannelHeader;

struct alignas(64) IncDcCombineBw05AckSlot {
    uint32_t epoch = 0;
    uint8_t pad[60]{};
};
static_assert(sizeof(IncDcCombineBw05AckSlot) == 64, "ACK slot 64B");

struct alignas(64) IncDcCombineBw05ReduceDoneSlot {
    uint64_t generation = 0;
    // Reserved pad only — do NOT use for RX progress (PERF2-M: reduce_done == reduce completion).
    uint8_t pad[56]{};
};
static_assert(sizeof(IncDcCombineBw05ReduceDoneSlot) == 64, "reduce_done slot 64B");

struct alignas(64) IncDcCombineBw05IngressReadyLine {
    uint64_t generation = 0;
    uint8_t pad[56]{};
};
static_assert(sizeof(IncDcCombineBw05IngressReadyLine) == 64, "ingress_ready line 64B");

struct IncDcCombineBw05PackedDescriptor {
    uint64_t generation = 0;
    uint32_t payload_bytes = 0;
    uint32_t entry_count = 0;
    uint32_t worklist_offset = 0;
    uint32_t payload_slot = 0;
    uint32_t worker_rank = 0;
    uint32_t lane_id = 0;
};

// PERF2-O O2: location entry lives in switch_ingress header (first 64B of pkt_stride).
// Reduce reads packed payload ring via (channel, payload_slot, entry_index) — no row copy.
struct alignas(64) IncDcCombineBw05LocationEntry {
    uint64_t generation = 0;
    uint32_t channel = 0;
    uint32_t payload_slot = 0;
    uint32_t entry_index = 0;
    uint32_t payload_bytes = 0;
    uint32_t magic = 0x4C4F4332u; // 'LOC2'
    uint32_t pad0 = 0;
    uint8_t pad[32]{};
};
static_assert(sizeof(IncDcCombineBw05LocationEntry) == 64, "location entry 64B");
static_assert(alignof(IncDcCombineBw05LocationEntry) >= 64, "location entry align");

// Production hot path: zero-copy packed-ring reduce (location only; copy is isolation F).
constexpr uint32_t kCombineBw05ZeroCopyPackedReduce = 1u;

struct IncDcCombineBw05WorklistEntry {
    int32_t contributor_rank = -1;
    uint32_t assignment_id = 0;
    uint32_t ingress_slot = 0;
    uint32_t expert_output_offset = 0;
    uint32_t target_switch_pe = 0;
    uint32_t result_token_idx = 0;
};

struct IncDcCombineBw05ConservationCounters {
    uint64_t published_descriptors = 0;
    // Historical name: assignment arrivals (entry_count sum), not packed descriptor slots.
    uint64_t consumed_descriptors = 0;
    uint64_t published_payload_bytes = 0;
    uint64_t consumed_payload_bytes = 0;
    uint32_t stale_count = 0;
    uint32_t future_count = 0;
    uint32_t dup_count = 0;
    uint32_t missing_count = 0;
    uint32_t timeout_count = 0;
    uint32_t overflow_count = 0;
    uint64_t unique_arrived_count = 0;
    uint64_t expected_arrived_count = 0;
    uint64_t ingress_ready_generation = 0;
    uint64_t last_future_key_lo = 0;
    uint64_t last_future_key_hi = 0;
    uint32_t worker_ack_timeout_count = 0;
    uint32_t first_missing_ack_destination = 0xFFFFFFFFu;
    // PERF2-M descriptor-level stats (per packed descriptor, not cumulative put misuse).
    uint64_t consumed_descriptor_slots = 0;
    uint64_t entry_count_sum = 0;
    uint32_t entry_count_min = 0xFFFFFFFFu;
    uint32_t entry_count_max = 0;
    uint32_t payload_bytes_min = 0xFFFFFFFFu;
    uint32_t payload_bytes_max = 0;
    // PERF3-V device attestation (PE-level; AIVs bump location_* ; bid0 stamps path flags).
    uint32_t vector_reduce_active = 0;
    uint32_t zero_copy_packed_reduce = 0;
    uint32_t location_invalid_count = 0;
    uint32_t location_fallback_count = 0;
};

struct alignas(64) IncDcCombineBw05WorkerAckCtrl {
    uint32_t ready_generation = 0;
    uint32_t error_generation = 0;
    uint32_t first_missing_ack_destination = 0xFFFFFFFFu;
    uint32_t reserved = 0;
    uint8_t pad[48]{};
};
static_assert(sizeof(IncDcCombineBw05WorkerAckCtrl) == 64, "worker_ack_ctrl 64B");

struct __attribute__((packed)) IncDcCombineBw05LayoutHeader {
    uint32_t schema = kCombineBw05LayoutSchema;
    uint32_t group_size = kCombineBw05MaxGroupSize;
    uint32_t lane_count = kCombineBw05LaneCount;
    uint32_t remote_channel_count = kCombineBw05MaxWorkerChannels;
    uint32_t producer_state_count = kCombineBw05MaxWorkerChannels;
    uint32_t channel_capacity = kCombineBw05ChannelCapacity;
    uint32_t chunk_bytes = 0;
    uint32_t switch_blocks = 20;
    uint64_t lane_ranges_off = 0;
    uint64_t producer_headers_off = 0;
    uint64_t remote_headers_off = 0;          // producer cursor array
    uint64_t remote_consumer_headers_off = 0; // consumer cursor array (schema3)
    uint64_t descriptors_off = 0;
    uint64_t payload_off = 0;
    uint64_t worklist_off = 0;
    uint64_t staging_off = 0;
    uint64_t ack_off = 0;
    uint64_t worker_ack_ctrl_off = 0;
    uint64_t ingress_ready_off = 0;
    uint64_t reduce_done_off = 0;
    uint64_t conservation_off = 0;
    uint64_t total_bytes = 0;
    uint64_t layout_digest = 0;
};
// 8×u32 + 15×u64
static_assert(sizeof(IncDcCombineBw05LayoutHeader) == 8 * 4 + 15 * 8, "BW05 layout header size drift");

// Shared field offsets for host + device (no magic numbers in kernel).
constexpr uint32_t kBw05HdrOffSchema = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, schema));
constexpr uint32_t kBw05HdrOffGroupSize = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, group_size));
constexpr uint32_t kBw05HdrOffLaneCount = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, lane_count));
constexpr uint32_t kBw05HdrOffRemoteChCount =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, remote_channel_count));
constexpr uint32_t kBw05HdrOffProducerCount =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, producer_state_count));
constexpr uint32_t kBw05HdrOffCapacity =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, channel_capacity));
constexpr uint32_t kBw05HdrOffChunk = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, chunk_bytes));
constexpr uint32_t kBw05HdrOffSwitchBlocks =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, switch_blocks));
constexpr uint32_t kBw05HdrOffLaneRanges =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, lane_ranges_off));
constexpr uint32_t kBw05HdrOffProducerHdrs =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, producer_headers_off));
constexpr uint32_t kBw05HdrOffRemoteHdrs =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, remote_headers_off));
constexpr uint32_t kBw05HdrOffRemoteConsumerHdrs =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, remote_consumer_headers_off));
constexpr uint32_t kBw05HdrOffDescriptors =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, descriptors_off));
constexpr uint32_t kBw05HdrOffPayload = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, payload_off));
constexpr uint32_t kBw05HdrOffWorklist = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, worklist_off));
constexpr uint32_t kBw05HdrOffStaging = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, staging_off));
constexpr uint32_t kBw05HdrOffAck = static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, ack_off));
constexpr uint32_t kBw05HdrOffWorkerAckCtrl =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, worker_ack_ctrl_off));
constexpr uint32_t kBw05HdrOffIngressReady =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, ingress_ready_off));
constexpr uint32_t kBw05HdrOffReduceDone =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, reduce_done_off));
constexpr uint32_t kBw05HdrOffConservation =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, conservation_off));
constexpr uint32_t kBw05HdrOffTotalBytes =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, total_bytes));
constexpr uint32_t kBw05HdrOffLayoutDigest =
    static_cast<uint32_t>(offsetof(IncDcCombineBw05LayoutHeader, layout_digest));

static_assert(kBw05HdrOffSchema == 0, "schema off");
static_assert(kBw05HdrOffGroupSize == 4, "group_size off");
static_assert(kBw05HdrOffLaneCount == 8, "lane_count off");
static_assert(kBw05HdrOffRemoteChCount == 12, "remote_ch off");
static_assert(kBw05HdrOffProducerCount == 16, "producer_count off");
static_assert(kBw05HdrOffCapacity == 20, "capacity off");
static_assert(kBw05HdrOffChunk == 24, "chunk off");
static_assert(kBw05HdrOffSwitchBlocks == 28, "switch_blocks off");
static_assert(kBw05HdrOffLaneRanges == 32, "lane_ranges off");
static_assert(kBw05HdrOffProducerHdrs == 40, "producer_hdrs off");
static_assert(kBw05HdrOffRemoteHdrs == 48, "remote_hdrs off");
static_assert(kBw05HdrOffRemoteConsumerHdrs == 56, "remote_consumer off");
static_assert(kBw05HdrOffDescriptors == 64, "descriptors off");
static_assert(kBw05HdrOffPayload == 72, "payload off");
static_assert(kBw05HdrOffWorklist == 80, "worklist off");
static_assert(kBw05HdrOffStaging == 88, "staging off");
static_assert(kBw05HdrOffAck == 96, "ack off");
static_assert(kBw05HdrOffWorkerAckCtrl == 104, "worker_ack_ctrl off");
static_assert(kBw05HdrOffIngressReady == 112, "ingress_ready off");
static_assert(kBw05HdrOffReduceDone == 120, "reduce_done off");
static_assert(kBw05HdrOffConservation == 128, "conservation off");
static_assert(kBw05HdrOffTotalBytes == 136, "total_bytes off");
static_assert(kBw05HdrOffLayoutDigest == 144, "layout_digest off");

struct IncDcCombineBw05DeviceLayout {
    uint64_t layout_header = 0;
    uint64_t lane_ranges = 0;
    uint64_t producer_headers = 0;
    uint64_t remote_headers = 0;          // producer cursors
    uint64_t remote_consumer_headers = 0; // consumer cursors
    uint64_t channel_headers = 0;         // alias of remote_headers (producer)
    uint64_t descriptors = 0;
    uint64_t payload_ring = 0;
    uint64_t worklist = 0;
    uint64_t lane_staging = 0;
    uint64_t ack_slots = 0;
    uint64_t worker_ack_ctrl = 0;
    uint64_t ingress_ready = 0;
    uint64_t reduce_done = 0;
    uint64_t conservation = 0;
    uint64_t total_bytes = 0;
    IncDcCombineBw05LayoutHeader header{};
};

struct IncDcCombineBw05WorklistPartition {
    IncDcCombineUploadLaneRange lanes[kCombineBw05LaneCount]{};
    uint32_t lane_count = kCombineBw05LaneCount;
};

inline uint32_t CombineBw05PayloadChunkBytes(uint32_t hidden_bytes)
{
    // Slot capacity (not put size): clamp to [64KiB, 256KiB] so small hidden can batch.
    // Oversized slot with payload_bytes < chunk is legal; remaining bytes need not be zeroed.
    if (hidden_bytes == 0) {
        return kCombineBw05MinPackedPutBytes;
    }
    if (hidden_bytes < kCombineBw05MinPackedPutBytes) {
        return kCombineBw05MinPackedPutBytes;
    }
    if (hidden_bytes > kCombineBw05MaxPackedPutBytes) {
        return kCombineBw05MaxPackedPutBytes;
    }
    return hidden_bytes;
}

inline bool CombineBw05PackedChannelEnabled()
{
    const char *raw = std::getenv("INC_DC_COMBINE_PACKED_CHANNEL");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

// Returns 0 when unset/invalid → host must run EstimateBw05ChannelCapacity (auto).
// Explicit env in [4,512] wins (e.g. ring-wrap cap=4). Never silently force 64.
inline uint32_t CombineBw05ChannelCapacityFromEnv()
{
    const char *raw = std::getenv("INC_DC_BW05_CHANNEL_CAPACITY");
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }
    const int v = std::atoi(raw);
    if (v < 4 || v > 512) {
        return 0;
    }
    return static_cast<uint32_t>(v);
}

uint64_t CombineBw05LayoutDigest(const IncDcCombineBw05LayoutHeader &h);

// ABI digest: compile-unit constants only (schema/sizeof/alignof/kBw05HdrOff*). Host and device
// must compute independently — never from a host-provided expected digest field.
inline uint64_t CombineBw05AbiDigest()
{
    uint64_t hash = 14695981039346656037ull;
    auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ull;
    };
    // Keep numeric aligns frozen (device AscendC cannot reliably use alignof in ABI digest).
    mix(kCombineBw05LayoutSchema);
    mix(sizeof(IncDcCombineBw05LayoutHeader));
    mix(1ull); // packed layout header
    mix(sizeof(IncDcCombineBw05ProducerHeader));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05ProducerCursorLine));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05ConsumerCursorLine));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05AckSlot));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05ReduceDoneSlot));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05IngressReadyLine));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05WorkerAckCtrl));
    mix(64ull);
    mix(sizeof(IncDcCombineBw05PackedDescriptor));
    mix(kBw05HdrOffSchema);
    mix(kBw05HdrOffGroupSize);
    mix(kBw05HdrOffLaneCount);
    mix(kBw05HdrOffRemoteChCount);
    mix(kBw05HdrOffProducerCount);
    mix(kBw05HdrOffCapacity);
    mix(kBw05HdrOffChunk);
    mix(kBw05HdrOffSwitchBlocks);
    mix(kBw05HdrOffLaneRanges);
    mix(kBw05HdrOffProducerHdrs);
    mix(kBw05HdrOffRemoteHdrs);
    mix(kBw05HdrOffRemoteConsumerHdrs);
    mix(kBw05HdrOffDescriptors);
    mix(kBw05HdrOffPayload);
    mix(kBw05HdrOffWorklist);
    mix(kBw05HdrOffStaging);
    mix(kBw05HdrOffAck);
    mix(kBw05HdrOffWorkerAckCtrl);
    mix(kBw05HdrOffIngressReady);
    mix(kBw05HdrOffReduceDone);
    mix(kBw05HdrOffConservation);
    mix(kBw05HdrOffTotalBytes);
    mix(kBw05HdrOffLayoutDigest);
    return hash;
}

// Schema2 total_bytes for same params (historical layout) — for H0 delta accounting only.
uint64_t ComputeCombineBw05Schema2TotalBytes(uint64_t base_off, uint32_t worklist_count, uint32_t hidden_bytes,
                                             uint32_t channel_capacity, uint32_t group_size = kCombineBw05MaxGroupSize,
                                             uint32_t switch_blocks = 20u);

void BuildBw05CompactWorklist(const std::vector<IncDcCombineUploadEntry> &plan,
                              std::vector<IncDcCombineBw05WorklistEntry> *out);

IncDcCombineBw05WorklistPartition PartitionBw05WorklistIntoLanes(
    const std::vector<IncDcCombineBw05WorklistEntry> &worklist, int32_t contributor_rank);

// Host mirror of LlBw05FormBatchEnd + per-(src,lane) remote channel descriptor counts.
// Credit is only reclaimed at epoch ACK → required_capacity = max descriptors on any channel / epoch.
struct IncDcBw05CapacityReport {
    uint32_t required_capacity = 0;
    uint32_t selected_capacity = 0;
    uint32_t max_descriptors_per_channel = 0;
    uint32_t env_capacity = 0; // 0 = auto
    uint32_t chunk_bytes = 0;
    uint32_t channel_count = 0;
    uint64_t payload_ring_bytes = 0;
    uint64_t payload_ring_bytes_cap64 = 0;
    uint64_t bytes_saved_vs_cap64 = 0;
    bool auto_selected = false;
};

IncDcBw05CapacityReport EstimateBw05ChannelCapacity(const std::vector<IncDcCombineUploadEntry> &upload_plan,
                                                    uint32_t group_size, uint32_t hidden_bytes,
                                                    uint32_t env_capacity /*0=auto*/);

IncDcCombineBw05DeviceLayout ComputeCombineBw05DeviceLayout(uint64_t base_off, uint32_t worklist_count,
                                                            uint32_t hidden_bytes, uint32_t channel_capacity,
                                                            uint32_t group_size = kCombineBw05MaxGroupSize,
                                                            uint32_t switch_blocks = 20u);

bool ValidateBw05DeviceLayout(const IncDcCombineBw05DeviceLayout &lo, uint64_t avail_bytes);

// Writer-identity cacheline audit. Returns mixed_writer_cacheline_count (0 = PASS).
struct Bw05CachelineAuditResult {
    uint32_t mixed_writer_cacheline_count = 0;
    uint32_t mutable_field_count = 0;
    uint32_t cacheline_span = 0;
    std::string first_conflict;
};

Bw05CachelineAuditResult AuditBw05CachelineOwnership(const IncDcCombineBw05DeviceLayout &lo);

} // namespace dc
} // namespace inc

#endif
