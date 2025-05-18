#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
// #include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifndef __builtin_assume
#  if defined(__has_builtin)
#    if __has_builtin(__builtin_unreachable)
#      define __builtin_assume(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#    else
#      define __builtin_assume(cond) ((void)0)
#    endif
#  else
#    define __builtin_assume(cond) ((void)0)
#  endif
#endif

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
