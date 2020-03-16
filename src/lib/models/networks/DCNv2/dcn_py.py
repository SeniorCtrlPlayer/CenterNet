from torch.autograd import Variable, Function
import torch
from torch import nn


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc=None, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.outc = outc
        self.kernel_size = kernel_size
        self.offsets = nn.Conv2d(inc, 3*kernel_size*kernel_size, kernel_size=3, padding=1)

        if self.outc is not None:
            self.conv_kernel2 = nn.Conv2d(inc, outc, kernel_size=3, stride=kernel_size, padding=1)
        self._init_weight()

    def _init_weight(self):
        self.offsets.weight.data = torch.zeros_like(self.offsets.weight.data)
        if self.offsets.bias is not None:
            self.offsets.bias.data = torch.FloatTensor(self.offsets.bias.shape[0]).zero_()

    def forward(self, x):
        offset_weight = self.offsets(x)
        offset_x, offset_y, weight = torch.chunk(offset_weight, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        weight = torch.sigmoid(weight)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # p = torch.cat([torch.clamp(p[..., :N], 1, x.size(2)-1-self.padding),
        #                 torch.clamp(p[..., N:], 1, x.size(3)-1-self.padding)], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1),
                       torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        # g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        # g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        # g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        # x_q_lt = self._get_x_q(x, q_lt, N)
        # x_q_rb = self._get_x_q(x, q_rb, N)
        # x_q_lb = self._get_x_q(x, q_lb, N)
        # x_q_rt = self._get_x_q(x, q_rt, N)

        # # (b, c, h, w, N)
        # # featureMap rearrangements
        # x_up_reset = g_lt.unsqueeze(dim=1) * x_q_lt + \
        #              g_rb.unsqueeze(dim=1) * x_q_rb + \
        #              g_lb.unsqueeze(dim=1) * x_q_lb + \
        #              g_rt.unsqueeze(dim=1) * x_q_rt
                     
        # featureMap rearrangements
        x_up_reset = ((1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))).unsqueeze(dim=1) * self._get_x_q(x, q_lt, N) + \
                     ((1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))).unsqueeze(dim=1) * self._get_x_q(x, q_rb, N) + \
                     ((1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))).unsqueeze(dim=1) * self._get_x_q(x, q_lb, N) + \
                     ((1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))).unsqueeze(dim=1) * self._get_x_q(x, q_rt, N)

        weight = weight.permute(0, 2, 3, 1).unsqueeze(dim=1) \
                        .expand(-1, x_up_reset.size(1), -1, -1, -1)
        # x_reset = torch.sum(x_up_reset * weight, -1).squeeze(1)
        x_up_reset = self._reshape_x_offset(x_up_reset, ks)
        out = self.conv_kernel2(x_up_reset)

        # return out, x_reset, p[..., :N].permute(0, 3, 1, 2), p[..., N:].permute(0, 3, 1, 2)
        return out
    
    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.linspace(1, h, h).type(dtype)-1,
                                      torch.linspace(1, w, w).type(dtype)-1)
        p_0_x = p_0_x.expand(1, N, h, w)
        p_0_y = p_0_y.expand(1, N, h, w)
        p_0 = torch.cat((p_0_x, p_0_y), dim=1)
        # p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # p = p_0 + p_n + offset
        p = p_0 + offset
        return p

    def _get_x_q(self, x, q, N):
        # 1d liner
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        # x = x.contiguous().view(b, c, -1)
        x = x.reshape(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        # index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        index = index.unsqueeze(dim=1).expand(-1, c, -1, -1, -1).reshape(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).reshape(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].reshape(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.reshape(b, c, h*ks, w*ks)

        return x_offset
    
class Deformable_Pooling_1D(nn.Module):
    def __init__(self, inc, kernel_size=1, stride=1):
        self.newindex = nn.Conv2d(inc, 2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        offset = self.newindex(x)
        N = offset.size(1)//2
        dtype = offset.data.type()
        
        p = self.get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-2),
                       torch.clamp(p[..., N:], 0, x.size(3)-2)], dim=-1)
        
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([q_lt[..., :N], q_lt[..., N:]], dim=-1).long()
        q_rb = torch.cat([q_rb[..., :N], q_rb[..., N:]], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # p = torch.cat([torch.clamp(p[..., :N], 1, x.size(2)-1-self.padding),
        #                 torch.clamp(p[..., N:], 1, x.size(3)-1-self.padding)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # featureMap rearrangements
        x_up_reset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                     g_rb.unsqueeze(dim=1) * x_q_rb + \
                     g_lb.unsqueeze(dim=1) * x_q_lb + \
                     g_rt.unsqueeze(dim=1) * x_q_rt
        
        out = x_up_reset.squeeze(1)
        return out
        
    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.linspace(1, h, h).type(dtype)-1,
                                      torch.linspace(1, w, w).type(dtype)-1)
        p_0_x = p_0_x.expand(1, N, h, w)
        p_0_y = p_0_y.expand(1, N, h, w)
        p_0 = torch.cat((p_0_x, p_0_y), dim=1)
        # p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # p = p_0 + p_n + offset
        p = p_0 + offset
        return p
        