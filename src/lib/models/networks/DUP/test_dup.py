#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dup import dup, Dup

N, inC, inH, inW = 1, 3, 2, 2
kH, kW = 2, 2
        
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
                    eps=1e-3, atol=1e-4, rtol=1e-2))
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
    dcn = Dup(input.size(1), kH, kW).cuda()
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
    check_gradient_dconv()
    #check_gradient_dconv1()
    # """
    # ****** Note: backward is not reentrant error may not be a serious problem,
    # ****** since the max error is less than 1e-7,
    # ****** Still looking for what trigger this problem
    # """
