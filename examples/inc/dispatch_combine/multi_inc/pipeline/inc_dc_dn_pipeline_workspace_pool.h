/**
 * M5：Worker local workspace grow-only pool（benchmark fallback）。
 * 热路径禁止 aclrtMalloc/Free；同 bucket 复用同一块 session 视图。
 * Runtime Payload V2：bucket 含 payload_bytes。
 */
#pragma once

#include "inc_dc_dn_pipeline_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"

namespace inc::dc::dn::pl {

inline uint32_t PlWorkspaceBucketRoundTokens(uint32_t tokens)
{
    if (tokens <= 128u) {
        return 128u;
    }
    if (tokens <= 512u) {
        return 512u;
    }
    if (tokens <= 4096u) {
        return 4096u;
    }
    return tokens;
}

struct PlWorkspacePoolBucketKey {
    uint32_t tokens = 0;
    uint32_t route_topk = 0;
    uint32_t final_slot_capacity = 0;
    uint32_t input_epoch_count = 0;
    uint32_t expert_per_pe = 0;
    uint32_t worker_count = 0;
    uint32_t payload_bytes = kQv2TokenBytes;

    bool operator==(const PlWorkspacePoolBucketKey &o) const
    {
        return tokens == o.tokens && route_topk == o.route_topk && final_slot_capacity == o.final_slot_capacity &&
               input_epoch_count == o.input_epoch_count && expert_per_pe == o.expert_per_pe &&
               worker_count == o.worker_count && payload_bytes == o.payload_bytes;
    }
};

struct PlWorkspacePoolEntry {
    PlWorkspacePoolBucketKey key{};
    uint8_t *base = nullptr;
    uint64_t total_bytes = 0;
    uint8_t *input = nullptr;
    int32_t *expert_ids = nullptr;
    uint8_t *gather_payload = nullptr;
    uint8_t *gather_desc = nullptr;
    uint8_t *worker_desc = nullptr;
    uint8_t *expand_x = nullptr;
    uint8_t *assist = nullptr;
    int32_t *ep_recv_count = nullptr;
    int32_t *expert_token_nums = nullptr;
};

// 进程级 grow-only；不在热路径 Free
class PlWorkerWorkspacePool {
public:
    bool Acquire(const PlWorkspacePoolBucketKey &want, PlWorkspacePoolEntry *out, bool *grew)
    {
        if (out == nullptr) {
            return false;
        }
        PlWorkspacePoolBucketKey key = want;
        key.tokens = PlWorkspaceBucketRoundTokens(want.tokens);
        if (key.payload_bytes == 0u) {
            key.payload_bytes = kQv2TokenBytes;
        }
        for (auto &e : entries_) {
            if (e.key == key) {
                *out = e;
                if (grew != nullptr) {
                    *grew = false;
                }
                std::cout << "PL_WORKSPACE_POOL_REUSE bucket_tokens=" << key.tokens << " topk=" << key.route_topk
                          << " final_slots=" << key.final_slot_capacity << " payload_bytes=" << key.payload_bytes
                          << " bytes=" << e.total_bytes << std::endl;
                return true;
            }
        }
        const uint32_t epochs = key.input_epoch_count == 0u ? 1u : key.input_epoch_count;
        const uint32_t pb = key.payload_bytes;
        const uint64_t need = PlWorkerLocalWorkspaceBytes(key.tokens, key.route_topk, key.final_slot_capacity, epochs,
                                                          key.expert_per_pe, key.worker_count, pb);
        if (need == 0u) {
            return false;
        }
        uint8_t *base = nullptr;
        if (aclrtMalloc(reinterpret_cast<void **>(&base), need, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            return false;
        }
        PlWorkspacePoolEntry e{};
        e.key = key;
        e.base = base;
        e.total_bytes = need;
        uint64_t off = 0u;
        const uint64_t input_bytes =
            static_cast<uint64_t>(key.tokens) * static_cast<uint64_t>(epochs) * static_cast<uint64_t>(pb);
        e.input = e.base + off;
        off += input_bytes;
        const uint64_t expert_bytes = static_cast<uint64_t>(key.tokens) * static_cast<uint64_t>(key.route_topk) *
                                     static_cast<uint64_t>(epochs) * sizeof(int32_t);
        e.expert_ids = reinterpret_cast<int32_t *>(e.base + off);
        off += expert_bytes;
        e.gather_payload = e.base + off;
        off += PlWorkerGatherPayloadBytesFor(pb);
        e.gather_desc = e.base + off;
        off += kPlWorkerGatherDescBytes;
        e.worker_desc = e.base + off;
        off += kPlWorkerDescBytes;
        const uint64_t expand_bytes = static_cast<uint64_t>(key.final_slot_capacity) * static_cast<uint64_t>(pb);
        e.expand_x = e.base + off;
        off += expand_bytes;
        const uint64_t assist_bytes =
            static_cast<uint64_t>(key.final_slot_capacity) * kPlDispatchAssistStrideBytes;
        e.assist = e.base + off;
        off += assist_bytes;
        const uint64_t ep_recv_bytes =
            static_cast<uint64_t>(key.expert_per_pe) * static_cast<uint64_t>(key.worker_count) * sizeof(int32_t);
        e.ep_recv_count = reinterpret_cast<int32_t *>(e.base + off);
        off += ep_recv_bytes;
        const uint64_t expert_nums_bytes = static_cast<uint64_t>(key.expert_per_pe) * sizeof(int32_t);
        e.expert_token_nums = reinterpret_cast<int32_t *>(e.base + off);
        off += expert_nums_bytes;
        if (off > need) {
            aclrtFree(base);
            return false;
        }
        entries_.push_back(e);
        *out = e;
        if (grew != nullptr) {
            *grew = true;
        }
        std::cout << "PL_WORKSPACE_POOL_GROW bucket_tokens=" << key.tokens << " topk=" << key.route_topk
                  << " final_slots=" << key.final_slot_capacity << " payload_bytes=" << key.payload_bytes
                  << " bytes=" << e.total_bytes << std::endl;
        return true;
    }

    size_t size() const { return entries_.size(); }

private:
    std::vector<PlWorkspacePoolEntry> entries_;
};

inline bool PlBuildInvocationWorkspace(PlInvocationWorkspaceDesc *out, uint64_t expand_x_ptr, uint64_t assist_info_ptr,
                                       uint64_t ep_recv_count_ptr, uint64_t expert_token_nums_ptr,
                                       uint32_t source_token_capacity, uint32_t output_route_capacity, uint32_t topk,
                                       uint32_t generation, uint32_t alloc_flags,
                                       uint32_t payload_bytes = kQv2TokenBytes, uint32_t hidden_size = 4096u,
                                       uint32_t dtype_bytes = kPlDtypeBytesFp16,
                                       uint32_t layout_version = kPlLayoutVersionRuntimePayloadV2)
{
    if (out == nullptr) {
        return false;
    }
    const uint32_t pb = (payload_bytes == 0u) ? kQv2TokenBytes : payload_bytes;
    *out = PlInvocationWorkspaceDesc{};
    out->magic = kPlInvocationWorkspaceMagic;
    out->flags = kPlInvocationWorkspaceFlagWorkerLocalFinal | alloc_flags;
    out->source_token_capacity = source_token_capacity;
    out->output_route_capacity = output_route_capacity;
    out->topk = topk;
    out->generation = generation;
    out->expand_x_ptr = expand_x_ptr;
    out->assist_info_ptr = assist_info_ptr;
    out->ep_recv_count_ptr = ep_recv_count_ptr;
    out->expert_token_nums_ptr = expert_token_nums_ptr;
    out->input_bytes = static_cast<uint64_t>(source_token_capacity) * static_cast<uint64_t>(pb);
    out->output_bytes = static_cast<uint64_t>(output_route_capacity) * static_cast<uint64_t>(pb);
    out->assist_bytes =
        static_cast<uint64_t>(output_route_capacity) * static_cast<uint64_t>(kPlDispatchAssistStrideBytes);
    out->layout_version = layout_version;
    out->hidden_size = hidden_size;
    out->dtype_bytes = dtype_bytes;
    out->payload_bytes = pb;
    const char *reason = nullptr;
    return PlValidateInvocationWorkspace(out, &reason);
}

} // namespace inc::dc::dn::pl
