#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import _dup as _backend
#a=True

class _DUP(Function):
    @staticmethod
    def forward(ctx, input, offset, kernel_size):
        ctx.kernel_size = kernel_size
        output = _backend.dup_forward(input, offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1])
        ctx.save_for_backward(input, offset)
        """
        global a
        if a:
            #print("output:",output.size())
            #print(output)
            pass
        """
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset = ctx.saved_tensors
        grad_input, grad_offset = \
            _backend.dup_backward(input, offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1])
        """
        global a
        if a:
            print("offset", offset)
            print("input", input)
            print("grad_offset", grad_offset)
            print("grad_input", grad_input)
            pass
        a = False
        """
        return grad_input, grad_offset, None


dup = _DUP.apply


class Dup(nn.Module):

    #def __init__(self, c, h, w, kh=2, kw=2):
    def __init__(self, c, kh=2, kw=2):
        super(Dup, self).__init__()
        self.channel = c
        #self.h = h
        #self.w = w
        self.kernel_size = (kh, kw)
        offset_channels_ = 2 * kh * kw
        self.offset_conv = nn.Conv2d(self.channel,
                                     offset_channels_,
                                     kernel_size=3,
                                     padding=1)
        self.init_offset()
        
    def init_offset(self):
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
    
    def forward(self, input):
        # assert input.shape[1] == self.channel and \
        #         input.shape[2] == self.h and \
        #         input.shape[3] == self.w
        assert input.shape[1] == self.channel
        offset = self.offset_conv(input)
        return dup(input, offset, self.kernel_size)
