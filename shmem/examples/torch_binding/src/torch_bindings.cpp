/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <string>
#include <algorithm>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "torch_npu/csrc/aten/common/from_blob.h"

#include "shmem.h"
#include "shmem_torch_kernel.h"
#include "torch_register.h"
#include "utils.h"

#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

namespace ShmemOps {

void print_tensor_info(const at::Tensor& tensor)
{
    // 打印张量的形状
    std::cout << "Shape: ";
    for (int64_t dim : tensor.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 打印张量的数据类型
    std::cout << "Dtype: " << tensor.dtype() << std::endl;

    // 打印张量的首地址
    if (tensor.defined()) {
        std::cout << "First element address: " << reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) << std::endl;
    } else {
        std::cout << "Tensor is not defined." << std::endl;
    }
}

aclshmemx_uniqueid_t default_flag_uid;

class Manager : public torch::jit::CustomClassHolder {
public:
    // 默认构造函数
    Manager() : name_("Manager") {}

    std::string get_name() const
    {
        return name_;
    }

    int64_t attr_init(int64_t my_pe, int64_t n_ranks, int64_t local_mem_size, const std::string& ip_port)
    {
        int64_t status = 0;
        status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
        aclshmemx_init_attr_t attributes;
        test_set_attr(my_pe, n_ranks, local_mem_size, ip_port.c_str(), default_flag_uid, &attributes);

        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        return status;
    }
    int64_t finalize()
    {
        return aclshmem_finalize();
    }
    
    at::Tensor malloc_tensor(int64_t size)
    {
        void *symmPtr = aclshmem_malloc(size);
        at::Tensor aclshmem_tensor = at_npu::native::from_blob(symmPtr, size, torch::dtype(torch::kUInt8));

        return aclshmem_tensor;
    }

    at::Tensor malloc_like(const at::Tensor& npu_tensor)
    {
        void *npu_tensor_ptr = static_cast<void *>(const_cast<void *>(npu_tensor.storage().data()));
        int64_t size = npu_tensor.storage().nbytes();
        void *symmPtr = aclshmem_malloc(size);
        at::Tensor aclshmem_tensor = at_npu::native::from_blob(symmPtr, npu_tensor.sizes(), npu_tensor.dtype());
        aclrtMemcpy(symmPtr, size, npu_tensor_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE);
        return aclshmem_tensor;
    }

    void free_tensor(const at::Tensor& aclshmem_tensor)
    {
        void *aclshmem_ptr = static_cast<void *>(const_cast<void *>(aclshmem_tensor.storage().data()));
        aclshmem_free(aclshmem_ptr);
        return;
    }

private:
    std::string name_;
};

static constexpr uint32_t DEFAULT_BLOCK_DIM = 16;
class KVShuffle : public torch::jit::CustomClassHolder {
public:
    // 默认构造函数
    KVShuffle() : name_("ShmemKVShuffle"), count_(0), block_dim_(DEFAULT_BLOCK_DIM), fftsAddr_(util_get_ffts_config()), sync_ptr_(nullptr)
    {
        int64_t SYNC_FLAG_INTERVAL = 16;
        sync_ptr_ = aclshmem_malloc(sizeof(int32_t) * aclshmem_n_pes() * block_dim_ * SYNC_FLAG_INTERVAL);
        aclrtMemset(sync_ptr_, sizeof(int32_t) * aclshmem_n_pes() * block_dim_ * SYNC_FLAG_INTERVAL, 0,
                    sizeof(int32_t) * aclshmem_n_pes() * block_dim_ * SYNC_FLAG_INTERVAL);
    }

    ~KVShuffle()
    {
        aclshmem_free(sync_ptr_);
    }

    std::string get_name() const
    {
        return name_;
    }
    
    void compute(const at::Tensor &ShuffleTable, const at::Tensor &KeyCache, const at::Tensor &ValueCache,
                 const at::Tensor &SrcBlockTable, const at::Tensor &DstBlockTable)
    {
        void *global_shuffle_table = const_cast<void *>(ShuffleTable.storage().data());

        void *k_cache = const_cast<void *>(KeyCache.storage().data());

        void *v_cache = const_cast<void *>(ValueCache.storage().data());

        void *src_block_table = const_cast<void *>(SrcBlockTable.storage().data());
        void *dst_block_table = const_cast<void *>(DstBlockTable.storage().data());

        int64_t block_nums = DstBlockTable.size(0);
        int64_t kv_head_num = KeyCache.size(1);
        int64_t head_dim = KeyCache.size(3);
        int64_t page_size = KeyCache.size(2);

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        count_++;
        ShmemKernel::aclshmem_kv_shuffle(block_dim_, stream, fftsAddr_, k_cache,
            v_cache, global_shuffle_table, src_block_table, dst_block_table, sync_ptr_,
            block_nums, kv_head_num, page_size, head_dim, count_);
    }
private:
    std::string name_;
    int32_t count_;
    uint32_t block_dim_;
    uint64_t fftsAddr_;
    void* sync_ptr_;
};

class AllGather : public torch::jit::CustomClassHolder {
public:
    // 默认构造函数
    AllGather() : name_("ShmemAllGather"), count_(0), fftsAddr_(util_get_ffts_config()), sync_ptr_(nullptr)
    {
        int64_t sync_size = BLOCK_NUM_LARGE_DATA * SYNC_FLAG_INTERVAL * sizeof(int32_t) + GVA_BUFF_MAX_SIZE / sizeof(fp16_t);
        sync_ptr_ = aclshmem_malloc(sync_size);
        aclrtMemset(sync_ptr_, sync_size, 0, sync_size);
    }

    ~AllGather()
    {
        aclshmem_free(sync_ptr_);
    }

    std::string get_name() const
    {
        return name_;
    }
    
    void compute(const at::Tensor &output_tensor, const at::Tensor &input_tensor)
    {
        TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
                "Compute Error: Input/Output dtype mismatch! \n",
                "Input dtype:  ", input_tensor.dtype().name(), "\n",
                "Output dtype: ", output_tensor.dtype().name());

        TORCH_CHECK((input_tensor.numel() * aclshmem_n_pes()) == output_tensor.numel(),
                    "Compute Error: Input/Output shape mismatch! \n",
                    "Input shape:  ", input_tensor.numel(), "\n",
                    "Output shape: ", output_tensor.numel());

        TORCH_CHECK(input_tensor.device() == output_tensor.device(),
                    "Compute Error: Input/Output device mismatch! \n",
                    "Input device:  ", input_tensor.device(), "\n",
                    "Output device: ", output_tensor.device());
        at::Tensor input_contig = input_tensor.contiguous();
        at::Tensor output_contig = output_tensor.contiguous();

        int64_t elements = input_contig.numel();
        int elem_byte = input_contig.element_size();
        size_t total_bytes = (size_t)elements * elem_byte;
        int n_blocks = (total_bytes < DATA_SIZE_THRESHOLD) ? BLOCK_NUM_SMALL_DATA : BLOCK_NUM_LARGE_DATA;

        void *input = const_cast<void *>(input_tensor.storage().data());

        void *output = const_cast<void *>(output_tensor.storage().data());


        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        count_++;
        at::ScalarType dtype = input_tensor.scalar_type();
        if (dtype == at::kInt) {
            ShmemKernel::aclshmem_allgather<int>(n_blocks, stream, fftsAddr_, input, output, sync_ptr_, elements, count_ * MAGIC_MULTIPLIER);
        } else if (dtype == at::kHalf) {
            ShmemKernel::aclshmem_allgather<fp16_t>(n_blocks, stream, fftsAddr_, input, output, sync_ptr_, elements, count_ * MAGIC_MULTIPLIER);
        } else if (dtype == at::kBFloat16) {
            ShmemKernel::aclshmem_allgather<bfloat16>(n_blocks, stream, fftsAddr_, input, output, sync_ptr_, elements, count_ * MAGIC_MULTIPLIER);
        } else {
            TORCH_CHECK(false, "Unsupported tensor dtype for test_aclshmem_all_gather! ",
                        "Current dtype: ", input_tensor.dtype().name(),
                        " | Supported dtypes: int, int32, float16, bfloat16");
        }
    }
private:
    std::string name_;
    int32_t count_;
    uint64_t fftsAddr_;
    void* sync_ptr_;
    
    uint32_t SYNC_FLAG_INTERVAL = 16;
    uint32_t MAGIC_MULTIPLIER = 1024;
    uint32_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
    uint32_t DATA_SIZE_THRESHOLD = 2097152;
    uint32_t BLOCK_NUM_SMALL_DATA = 8;
    uint32_t BLOCK_NUM_LARGE_DATA = 16;
};

}  // namespace ShmemOps



REGISTER_SHMEM_OPS_CLASS(Manager, attr_init, finalize, malloc_tensor, free_tensor, malloc_like, get_name);
REGISTER_SHMEM_OPS_CLASS(KVShuffle, compute, get_name);
REGISTER_SHMEM_OPS_CLASS(AllGather, compute, get_name);