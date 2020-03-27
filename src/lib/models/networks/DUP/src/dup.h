#pragma once

//#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor
dup_forward(const at::Tensor &input,
               const at::Tensor &offset,
               const int kernel_h,
               const int kernel_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dup_cuda_forward(input, offset,
                                   kernel_h, kernel_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
dup_backward(const at::Tensor &input,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                int kernel_h, int kernel_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return dup_cuda_backward(input,
                                    offset,
                                    grad_output,
                                    kernel_h, kernel_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

