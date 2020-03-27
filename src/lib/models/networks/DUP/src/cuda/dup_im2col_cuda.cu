#include "dup_im2col_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
                                            const int data_width, const int h_or_w/*, const int debug*/)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  /*
  if (debug)
  {
    printf("=======biliner=======\n");
    printf("h: %f", argmax_h);
    printf(" w: %f", argmax_w);
    printf(" h_low: %d", argmax_h_low);
    printf(" w_low: %d", argmax_w_low);
    printf(" h_high: %d", argmax_h_high);
    printf(" w_high: %d\n", argmax_w_high);
  }*/

  float weight = 0;
// 0 is to get grad_y, 1 is to get grad_x
  if (h_or_w == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
    {
      weight += -1 * (argmax_w_high - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
      weight += (argmax_w_high - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    }
  }
  else if (h_or_w == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
    {
      weight += -1 * (argmax_h_high - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
      weight += (argmax_h_high - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    }
  }

  return weight;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int num_channels,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int w_in = w_col / kernel_w;
    const int h_in = h_col / kernel_h;
    const int channel = (index / width_col / height_col) % num_channels;

    float *data_col_ptr = data_col + (channel * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im + channel * height * width;
    const int kernel_index_h = h_col % kernel_h * 2 + w_col % 2;
    const int kernel_index_w = kernel_index_h + kernel_h * kernel_w;
    
    const int data_offset_h_ptr = (kernel_index_h * height + h_in) * width + w_in;
    const int data_offset_w_ptr = (kernel_index_w * height + h_in) * width + w_in;
    const float offset_h = data_offset[data_offset_h_ptr];
    const float offset_w = data_offset[data_offset_w_ptr];
    const float h_im = h_in + offset_h;
    const float w_im = w_in + offset_w;
    float val = static_cast<float>(0);
    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
    {
      val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
    }
    *data_col_ptr = val;
  }
}
/*
__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    //int b = (index / width_col / height_col);
    
    const float *data_offset_ptr = data_offset;// + b * 2 * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    //const float mask = data_mask_ptr[data_mask_hw_ptr];
    //const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    //const float cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const float cur_inv_h_data = h_out + offset_h;
    const float cur_inv_w_data = w_out + offset_w;

    const float cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
          float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}
*/
__device__ void grad_to_data_im(float *bottom_data, const int data_width,
                                const int height, const int width, float h, float w,
                                const float grad_output_p)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  
  if (h_low >= 0 && w_low >= 0 && h_high <= height - 1 && w_high <= width - 1)
  {/*
    bottom_data[h_low * data_width + w_low] += w1 * grad_output_p;
    bottom_data[h_low * data_width + w_high] += w2 * grad_output_p;
    bottom_data[h_high * data_width + w_low] += w3 * grad_output_p;
    bottom_data[h_high * data_width + w_high] += w4 * grad_output_p;
    */
    atomicAdd(bottom_data + h_low * data_width + w_low, w1 * grad_output_p);
    atomicAdd(bottom_data + h_low * data_width + w_high, w2 * grad_output_p);
    atomicAdd(bottom_data + h_high * data_width + w_low, w3 * grad_output_p);
    atomicAdd(bottom_data + h_high * data_width + w_high, w4 * grad_output_p);
  }
}

__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *grad_output, const float *data_offset,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
{
// add by lwk
  CUDA_KERNEL_LOOP(index, n)
  {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int w_in = w_col / kernel_w;
    const int h_in = h_col / kernel_h;
    const int channel = (index / width_col / height_col) % channels;

    //float *data_col_ptr = data_col + (channel * height_col + h_col) * width_col + w_col;
    const int data_col_index = (channel * height_col + h_col) * width_col + w_col;
    float *data_im_ptr = grad_im + channel * height * width;
    const int kernel_index_h = h_col % kernel_h * 2 + w_col % 2;
    const int kernel_index_w = kernel_index_h + kernel_h * kernel_w;
    
    const int data_offset_h_ptr = (kernel_index_h * height + h_in) * width + w_in;
    const int data_offset_w_ptr = (kernel_index_w * height + h_in) * width + w_in;
    const float offset_h = data_offset[data_offset_h_ptr];
    const float offset_w = data_offset[data_offset_w_ptr];
    const float h_im = h_in + offset_h;
    const float w_im = w_in + offset_w;
    const float grad_output_p = grad_output[data_col_index];
    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
    {
      grad_to_data_im(data_im_ptr, width, height, width, h_im, w_im, grad_output_p);
    }
  }
}

__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *grad_out, const float *data_im,
                                                             const float *data_offset,
                                                             const int out_channels, const int height, const int width,
                                                             const int hw_out, const int kernel_h, const int kernel_w,
                                                             const int xy_offset_channels,
                                                             float *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0;
    int offset_h_ptr = index % hw_out;
    int offset_w_ptr = offset_h_ptr + hw_out;
    int h_or_w = index / hw_out;
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % xy_offset_channels;
    
    float inv_h = h + data_offset[offset_h_ptr];
    float inv_w = w + data_offset[offset_w_ptr];
    // get grad_out_ptr
    int w_out = kernel_w * w + c % 2;
    int h_out = kernel_h * h + c / 2;
    int grad_out_ptr = h_out * (width * kernel_w) + w_out;
    
    float weight = 0;
    
    if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
    {
      inv_h = inv_w = -2;
    }
    for (int data_c=0; data_c < out_channels; data_c++)
    {
        weight = dmcn_get_coordinate_weight(inv_h, inv_w,
                                            height, width,
                                            data_im + data_c * hw_out,
                                            width, h_or_w);

        val += weight * grad_out[grad_out_ptr + data_c * hw_out];
    }
    grad_offset[index] = val;
  }
}
void modulated_deformable_im2col_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset,
  const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  float* data_col) {
  // num_axes should be smaller than block size
  //const int num_kernels = channels * batch_size * height_col * width_col;
  const int num_kernels = channels * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, height_im, width_im, kernel_h, kernel_w,
      channels, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}
/*
 如果通过遍历grad_im来计算，因为不知道每个坐标点被使用了多少次，所以没办法计算，
 于是通过遍历grad_out的方式进行计算，每个grad_out的像素点都是由4个input组成，即可计算分发在每个p0上的梯度
*/
void modulated_deformable_col2im_cuda(cudaStream_t stream,
  const float* data_col, const float* data_offset,
  const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  float* grad_im){

  //const int num_kernels = channels * kernel_h * kernel_w * height_col * width_col;
  const int num_kernels = channels * kernel_h * kernel_w * height_im * width_im;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, channels, height_im, width_im,
        kernel_h, kernel_w, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

void modulated_deformable_col2im_coord_cuda(cudaStream_t stream,
  const float* data_col, const float* data_im, const float* data_offset,
  const int channels, const int height_im, const int width_im,
  const int height_out, const int width_out, const int kernel_h, const int kernel_w,
  float* grad_offset) {
  //const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w;
  const int num_kernels = 2 * height_out * width_out;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, channels, height_im, width_im,
        num_kernels / 2, kernel_h, kernel_w, 2 * kernel_h * kernel_w, grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}
