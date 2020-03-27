#pragma once
#include <torch/extension.h>

at::Tensor
dup_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w);

std::vector<at::Tensor>
dup_cuda_backward(const at::Tensor &input,
                     const at::Tensor &offset,
                     const at::Tensor &grad_output,
                     int kernel_h, int kernel_w);

