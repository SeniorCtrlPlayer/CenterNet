#include <vector>
#include "cuda/dup_im2col_cuda.h"
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

// [batch gemm]
// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/generic/THCTensorMathBlas.cu

at::Tensor
dup_cuda_forward(const at::Tensor &input,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w)
{
    using scalar_t = float;
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    //AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int height_out = height * kernel_h;
    const int width_out = width * kernel_w;

    auto columns = at::empty({batch, channels, height_out, width_out}, input.options());
    // launch batch threads
    
    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto columns_n = columns.select(0, b);
        modulated_deformable_im2col_cuda(THCState_getCurrentStream(state),
                                         input_n.data<scalar_t>(),
                                         offset_n.data<scalar_t>(),
                                         channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         columns_n.data<scalar_t>());
    }

    return columns;
}
std::vector<at::Tensor> dup_cuda_backward(const at::Tensor &input,
                                             const at::Tensor &offset,
                                             const at::Tensor &grad_output,
                                             int kernel_h, int kernel_w)
{

    THArgCheck(input.is_contiguous(), 1, "input tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int height_out = height * kernel_h;
    const int width_out = width * kernel_w;

    //auto ones = at::ones({height_out, width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_offset = at::zeros_like(offset);
    //std::cout<<grad_offset<<std::endl;

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto columns_n = grad_output.select(0, b);
        //columns_n = columns_n.sum(0);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        
        modulated_deformable_col2im_coord_cuda(THCState_getCurrentStream(state),
                                               columns_n.data<scalar_t>(),
                                               input_n.data<scalar_t>(),
                                               offset_n.data<scalar_t>(),
                                               channels, height, width,
                                               kernel_h, kernel_w,
                                               grad_offset_n.data<scalar_t>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(THCState_getCurrentStream(state),
                                         columns_n.data<scalar_t>(),
                                         offset_n.data<scalar_t>(),
                                         channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         grad_input_n.data<scalar_t>());
    }
    //std::cout<<"after"<<std::endl;
    //std::cout<<grad_offset<<std::endl;
    
    return {
        grad_input, grad_offset
    };
}
