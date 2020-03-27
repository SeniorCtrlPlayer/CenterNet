#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dup import dup, Dup

N, inC, inH, inW = 2, 2, 4, 4
kH, kW = 2, 2

def check_dup_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()
    dcn_v2 = DCNv2(inC, outC, (kH, kW),
                   stride=1, padding=1, dilation=1,
                   deformable_groups=deformable_groups).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn_v2.weight, dcn_v2.bias)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn_v2(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('Zero offset passed')
    else:
        print('Zero offset failed')
        print(input)
        print(output)
        
def check_zero_offset():
    offset = torch.ones(N, 2 * kH * kW, inH, inW).cuda() * 0.5

    input = torch.linspace(1, 16, 16).reshape(N,1,inH,inW).expand((N,2,inH,inW)).cuda()
    input[0][1]+=1
    print(input)
    dup = Dup(inC, inH, inW, kH, kW).cuda()
    output = dup(input, offset)
    print(output)

def check_gradient_dconv():

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.zeros(N, 2 * kW * kH, inH, inW).cuda()
    offset.requires_grad = True

    print('check_gradient_dconv: ',
          gradcheck(dup, (input, offset, (kW,kH)),
                    eps=1e-1, atol=1e-4, rtol=1e-2))
def example_dconv():
    input = torch.ones(N, inC, inH, inW).cuda()
    input[0][0][0][0] = 2
    input[0][1][0][0] = 2
    input.requires_grad = True
    offset = torch.zeros(N, 2 * kH * kW, inH, inW).cuda()
    offset[0][0][0][0] = 0.4
    offset[0][kW*kH][0][0] = 0.4
    offset.requires_grad = True
    # wrap all things (offset and mask) in DCN
    dcn = Dup(kH).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input, offset)
    target = torch.ones(N, inC, kW*inH, kH*inW).cuda()
    # targert = output.new(*output.size())
    # targert.data.uniform_(-0.01, 0.01)
    # error = (targert - output).mean()
    # error.backward()
    loss = (target-output).sum()
    loss.backward()
    print(output.shape)

def example_dup():
    # wrap all things (offset and mask) in DCN
    input = torch.randn((N, inC, inH, inW)).cuda()
    dcn = Dup(input.size(1), input.size(2), input.size(3)).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)
    
if __name__ == '__main__':

    example_dup()
    #example_dconv()
    # example_dpooling()
    # example_mdpooling()

    # check_pooling_zero_offset()
    # zero offset check
    # if inC == outC:
    #     check_zero_offset()
    #check_zero_offset()
    # check_gradient_dpooling()
    #check_gradient_dconv()
    #check_gradient_dconv1()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
