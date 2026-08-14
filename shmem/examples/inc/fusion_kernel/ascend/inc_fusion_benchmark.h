#ifndef INC_FUSION_BENCHMARK_H
#define INC_FUSION_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INC_FUSION_BENCHMARK_ABI_VERSION 1u

/*
 * The four factorial baselines differ along exactly two axes.  Native vLLM is
 * an external end-to-end reference and is deliberately not a member of the
 * factorial comparison.
 */
typedef enum inc_fusion_benchmark_mode {
    INC_FUSION_BENCH_SERIAL_SHMEM = 1,
    INC_FUSION_BENCH_SERIAL_INC = 2,
    INC_FUSION_BENCH_FUSED_SHMEM = 3,
    INC_FUSION_BENCH_FUSED_INC = 4,
    INC_FUSION_BENCH_NATIVE_VLLM = 5,
} inc_fusion_benchmark_mode_t;

typedef enum inc_fusion_benchmark_transport {
    INC_FUSION_BENCH_TRANSPORT_SHMEM = 1,
    INC_FUSION_BENCH_TRANSPORT_INC = 2,
    INC_FUSION_BENCH_TRANSPORT_NATIVE = 3,
} inc_fusion_benchmark_transport_t;

typedef enum inc_fusion_benchmark_schedule {
    INC_FUSION_BENCH_SCHEDULE_SERIAL = 1,
    INC_FUSION_BENCH_SCHEDULE_TOKEN_WAVE = 2,
    INC_FUSION_BENCH_SCHEDULE_NATIVE = 3,
} inc_fusion_benchmark_schedule_t;

enum {
    /* Route packing is part of every measured production invocation. */
    INC_FUSION_BENCH_TIME_ROUTE_PACK = 1ull << 0,
    /* Plan creation, allocation, JIT and weight transforms are excluded. */
    INC_FUSION_BENCH_TIME_EXCLUDE_SETUP = 1ull << 1,
    /* One sample is the maximum completion time over all worker ranks. */
    INC_FUSION_BENCH_TIME_MAX_WORKER = 1ull << 2,
    /* Timing is taken from device events/cycles, not unsynchronised host time. */
    INC_FUSION_BENCH_TIME_DEVICE = 1ull << 3,
};

#define INC_FUSION_BENCH_REQUIRED_TIMING_FLAGS                              \
    (INC_FUSION_BENCH_TIME_ROUTE_PACK |                                    \
     INC_FUSION_BENCH_TIME_EXCLUDE_SETUP |                                 \
     INC_FUSION_BENCH_TIME_MAX_WORKER |                                    \
     INC_FUSION_BENCH_TIME_DEVICE)

typedef struct inc_fusion_benchmark_mode_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode;
    uint32_t transport;
    uint32_t schedule;
    /* Boolean: this mode uses the one dedicated INC PE. */
    uint32_t uses_dedicated_inc;
    uint32_t is_factorial_baseline;
    uint32_t reserved32;
    uint64_t reserved[4];
} inc_fusion_benchmark_mode_info_t;

/*
 * Digests are produced by the harness after canonicalising the corresponding
 * payload.  A zero digest is invalid: accepting it would silently make an
 * unverified field look comparable.
 */
typedef struct inc_fusion_benchmark_signature {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t mode;
    uint32_t worker_count;
    /* Boolean: the invocation uses the one dedicated INC PE. */
    uint32_t uses_dedicated_inc;
    uint32_t dtype;
    uint64_t token_count;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t expert_count;
    uint32_t topk;
    uint32_t token_wave_capacity;
    uint32_t reserved32;
    uint64_t route_digest;
    uint64_t expert_placement_digest;
    uint64_t compute_implementation_digest;
    uint64_t weight_layout_digest;
    uint64_t timing_flags;
    uint64_t reserved[4];
} inc_fusion_benchmark_signature_t;

/* Per-wave stage costs for a fixed shape and route distribution. */
typedef struct inc_fusion_benchmark_stage_model {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t wave_count;
    uint32_t reserved32;
    double dispatch_us;
    double ffn_us;
    double combine_us;
    double reserved[4];
} inc_fusion_benchmark_stage_model_t;

typedef struct inc_fusion_benchmark_theory {
    uint32_t struct_size;
    uint32_t abi_version;
    double communication_window_speedup_limit;
    double serial_pipeline_us;
    double ideal_fused_pipeline_us;
    double end_to_end_speedup_limit;
    double reserved[4];
} inc_fusion_benchmark_theory_t;

void inc_fusion_benchmark_signature_init(
    inc_fusion_benchmark_signature_t *signature);
void inc_fusion_benchmark_stage_model_init(
    inc_fusion_benchmark_stage_model_t *model);

int inc_fusion_benchmark_mode_info(
    inc_fusion_benchmark_mode_t mode,
    inc_fusion_benchmark_mode_info_t *info);

/*
 * Validate one member first, then validate a pair.  Pair validation accepts
 * different mode and dedicated-INC usage, but every workload/compute/timing
 * field must match.  Native vLLM is rejected here because it is an external
 * baseline rather than one cell of the 2x2 attribution experiment.
 */
int inc_fusion_benchmark_validate_signature(
    const inc_fusion_benchmark_signature_t *signature,
    char *error, size_t error_bytes);
int inc_fusion_benchmark_validate_factorial_pair(
    const inc_fusion_benchmark_signature_t *reference,
    const inc_fusion_benchmark_signature_t *candidate,
    char *error, size_t error_bytes);

/*
 * Finite-wave ideal: D+F+C fills/drains the first wave, then every additional
 * wave costs max(D,F,C).  It is an optimistic upper bound, not a measured
 * speedup.  In particular the D/C-only limit is <= 2 and is independent of W.
 */
int inc_fusion_benchmark_compute_theory(
    const inc_fusion_benchmark_stage_model_t *model,
    inc_fusion_benchmark_theory_t *theory);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
