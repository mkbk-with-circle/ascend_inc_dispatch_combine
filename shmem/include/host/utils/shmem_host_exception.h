/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef ACLSHMEM_HOST_EXCEPTION_H
#define ACLSHMEM_HOST_EXCEPTION_H

#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Runtime exception report diagnostic level.
 *
 * INFO records Runtime exception fields in the Runtime callback path and
 * reports the cached fields only. INFO does not copy ACLSHMEM transport state.
 * DEBUG reports Runtime fields and collects detailed ACLSHMEM transport state
 * when aclshmemx_report_exception() is called. DEBUG may launch diagnostic
 * read kernels on the active ACLSHMEM default stream, synchronize that stream,
 * and copy device memory to host before returning.
 */
typedef enum aclshmemx_exception_report_level {
    ACLSHMEMX_EXCEPTION_REPORT_INFO = 0,
    ACLSHMEMX_EXCEPTION_REPORT_DEBUG = 1,
} aclshmemx_exception_report_level_t;

/**
 * @brief User exception information callback.
 *
 * The exception_info argument is the original Runtime exception information
 * pointer. ACLSHMEM passes it as an opaque value to avoid exposing Runtime
 * callback types in the ACLSHMEM public interface. The pointer is owned by
 * Runtime and is valid only for the duration of the callback invocation.
 */
typedef void (*aclshmemx_exception_info_callback_t)(void* exception_info);

/**
 * @brief Enable runtime exception reporting.
 *
 * ACLSHMEM installs an internal Runtime exception callback trampoline. This
 * function takes ownership of aclrtSetExceptionInfoCallback for exception
 * reporting, records runtime exception snapshots, and then forwards the runtime
 * exception info as an opaque pointer when callback is not null.
 * Passing nullptr enables ACLSHMEM's internal exception record and later
 * aclshmemx_report_exception() diagnostics without forwarding to a user callback.
 * Runtime exception callbacks are process-wide. In multi-instance processes,
 * ACLSHMEM reports diagnostics against the currently active ACLSHMEM instance;
 * asynchronous attribution to a non-active instance is not supported.
 * Call this function after selecting the device used by the active ACLSHMEM
 * instance. When called before ACLSHMEM init and Runtime exception callback
 * symbols are unavailable, the configuration is cached and retried during
 * ACLSHMEM initialization. Reconfiguring this API replaces the previously
 * configured ACLSHMEM callback and level. This API is serialized internally
 * against ACLSHMEM exception-report finalize/restore operations.
 *
 * ACLSHMEM finalize clears ACLSHMEM's internal exception-report state. If
 * ACLSHMEM successfully registered its Runtime exception callback trampoline,
 * finalize also clears the process-wide Runtime callback by setting it to
 * nullptr. If exception reporting was never enabled or the Runtime callback was
 * never registered by ACLSHMEM, finalize does not modify the Runtime callback.
 * A callback that existed before ACLSHMEM exception reporting was enabled is not
 * restored because Runtime does not expose the previous callback value.
 *
 * @param[in] callback User exception information callback, or nullptr for
 * ACLSHMEM-only reporting. ACLSHMEM stores the function pointer; the callback
 * target must remain valid until exception reporting is disabled, reconfigured,
 * or ACLSHMEM is finalized.
 * @param[in] level Diagnostic level used by aclshmemx_report_exception().
 * Must be ACLSHMEMX_EXCEPTION_REPORT_INFO or ACLSHMEMX_EXCEPTION_REPORT_DEBUG.
 * @return Returns ACLSHMEM_SUCCESS on success, ACLSHMEM_INVALID_PARAM for an
 * invalid level, or the Runtime callback registration error code on failure.
 */
ACLSHMEM_HOST_API int aclshmemx_enable_exception_report(
    aclshmemx_exception_info_callback_t callback, aclshmemx_exception_report_level_t level);

/**
 * @brief Report pending runtime exception diagnostics at a safe point.
 *
 * When no unreported runtime exception snapshot exists, this function returns
 * immediately without logging or copying device memory.
 * In INFO mode this function only logs the cached Runtime exception fields. In
 * DEBUG mode it additionally inspects the active ACLSHMEM instance and selected
 * transport engines. DEBUG reporting uses the active instance's device context
 * and default stream; callers must invoke it from a context where that device,
 * stream, and ACLSHMEM instance are still valid. The function may block while
 * synchronizing the default stream and copying diagnostic data.
 *
 * Do not call this function concurrently with ACLSHMEM finalize or with a
 * thread switching the active ACLSHMEM instance/device. Calls are safe to repeat;
 * a successfully reported snapshot is marked reported and later calls return
 * immediately until a new Runtime exception snapshot is recorded.
 *
 * @return Returns ACLSHMEM_SUCCESS when reporting succeeds, no report is
 * pending, or Runtime exception reporting is unsupported. Returns an ACLSHMEM
 * error code if DEBUG transport-state collection fails.
 */
ACLSHMEM_HOST_API int aclshmemx_report_exception(void);

#ifdef __cplusplus
}
#endif

#endif // ACLSHMEM_HOST_EXCEPTION_H
