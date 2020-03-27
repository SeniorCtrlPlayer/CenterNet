
import torch
import numpy as np
 
# 定义一个继承了Function类的子类，实现y=f(x)的正向运算以及反向求导
class sqrt_and_inverse(torch.autograd.Function):
    '''
    forward和backward可以定义成静态方法，向定义中那样，也可以定义成实例方法
    '''
    # 前向运算
    @staticmethod
    def forward(ctx, input_x,input_y):
        ctx.save_for_backward(input_x,input_y)                  
        output=torch.sqrt(input_x)+torch.reciprocal(input_x)+2*torch.pow(input_y,2)
        return output                              

    @staticmethod                        
    def backward(ctx, grad_output):                             
        input_x,input_y=ctx.saved_tensors  # 获取前面保存的参数,也可以使用self.saved_variables
        grad_x = grad_output *(torch.reciprocal(2*torch.sqrt(input_x))-torch.reciprocal(torch.pow(input_x,2)))
        grad_y= grad_output *(4*input_y)
 
        return grad_x, grad_y
myfunc = sqrt_and_inverse.apply
def sqrt_and_inverse_func(input_x,input_y):
    return myfunc(input_x,input_y)  # 这里是对象调用的含义，因为function中实现了__call__
 
x=torch.tensor(3.0,requires_grad=True) #标量
y=torch.tensor(2.0,requires_grad=True)
 
print('开始前向传播')
z=sqrt_and_inverse_func(x,y)                      
 
print('开始反向传播')
z.backward()   # 这里是标量对标量求导                         
 
print(x.grad)
print(y.grad)
a = torch.linspace(1, 8, 8).reshape(2,2,2)
print(a[0], a[1])
print(a.sum(dim=0))