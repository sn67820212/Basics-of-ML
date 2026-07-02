# This is second basic tutorial for learning Neural Networks using torch.
# We will discuss gradient in this particular tutorial.
# If u are completely new to neural networks, I would suggest you to go through the first tutorial
# before going through this one.
import torch

# level 9: requires_grad = True
# When we set requires_grad = True, pytorch records every operation in a computational graph. 
# This is essential for performing automatic differentiation, and is the basis for backpropagation.

"""x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
# though x,y are tensor objects,but let's see their detail by printing them
print(x)
print(y)

# Now, let's create x,y without requires_grad and check if y has grad_fn attribute or not.
x1 = torch.tensor(2.0)
y1 = x1 ** 3
print(x1)
print(y1)

a = torch.linspace(0,1,5, requires_grad=True) # does linspace support requires_grad? yes it does.
print(a)
# let's try .requires_grad_(True) on same linspace tensor
b = torch.linspace(0,1,5).requires_grad_(True)
print(b)
#alternative:
b = torch.linspace(0,1,5)
b.requires_grad_(True)
print(b)"""

"""# Level 10: .backward() and .grad
# The .backward() walks computational graph backwards, computing derivative via chain rule. 
# After calling .backward(), .grad on any requires_grad tensor will hold its derivative w.r.t. the scalar u called backward() on.
# let's see the concepts from example:
x = torch.tensor(2.0, requires_grad=True)
y = x ** 4 # dy/dx = 4*x^3 = 4*2^3 = 32
y.backward() # this will compute the derivative of y w.r.t. x and store it
print(x.grad) # prints 32.0 .. why we use x.grad but clearly it's grad of y w.r.t. x. Certain things to note here:
# 1. computaion is not always identical to standard calculus.
# 2. In neural network, we are not interested in derivative of single function
# 3. In neural network, we are interested in derivative of loss function.
# 4. Loss funtion can be due to different sources, so we track loss on each sources and 
# 5. x.grad is the sum of all the gradients from different sources. This is called gradient accumulation.
# 6. If u got confused, don't worry, u may realize when solving neural network problems and
# 7. U can come back later and read this tutorial again. It will make more sense then.
# 8. For now, we move onto another example

x = torch.tensor(1.0, requires_grad=True)
y = 3*x**2 + 2*x # dy/dx = 6*x + 2 = 6*1 + 2 = 8
y.backward()
print(x.grad)

# A common bug: calling .backward() multiple times
x = torch.tensor(1.0, requires_grad=True)
y = 3*x**2 + 2*x # dy/dx = 6*x + 2 = 6*1 + 2 = 8
y.backward()
y.backward() # returns error massege 
print(x.grad)"""
# after backward() is called, the computational graph is freed by default to save memory.
# If you want to call backward() on the same graph multiple times, you need to pass retain_graph=True. i.e.:

"""x = torch.tensor(1.0, requires_grad=True)
y = 3*x**2 + 2*x # dy/dx = 6*x + 2 = 6*1 + 2 = 8
y.backward(retain_graph=True)
y.backward() # returns  no error massege 
print(x.grad) # the gradient is accumulated, so it will be 16.0 now (8.0 + 8.0)
# this should have made clear gradient is not calculated over single function. It gets accumulated as,
# We want to minimize total error not the individual error. To understand this concept U could watch:
# 3 blue 1 brown's vedio series on youtube. It's really great."""

# Level 11: torch.autograd.grad()
# It calculates gradient directly, no .backward() required as shown below:
# Code 11.1
"""x = torch.tensor(2.0, requires_grad=True)
y = x**3
#compute dy/dx explicitly without using .backward()
grad = torch.autograd.grad(y,x) # dy/dx = 3*x**2 = 12
print(grad) # (tensor(6.),) # This is tuple
print(grad[0]) # tensor(6.), # This is value i.e. slicing is required when our interest is on values"""

# Code 11.2
"""x = torch.linspace(0,1,4).requires_grad_(True)
y = x**2
z = torch.sum(y)
dzdx= torch.autograd.grad(z,x) # 0,0.667,1.333,2
print(dzdx) # The result is equal to classical dydx
# z = sum(y_i)
# del z / del x_i = dy/dx_i.. calculus trick we used because:
# gradient finds derivative w.r.t coordinates and removes unnececessary terms by calculus"""

# Level 12: create_graph = True
# Usually pytorch erases computation graph after doing derivative operations to save memory
# To use the derivative to find higher order derivatives, we need to save the computation graph.
# We do it by create_graph=True command

#problem 12.1
"""x = torch.tensor(1.0,requires_grad=True)
y = x**4
# First derivative (dy/dx)
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print (dy_dx) # expected: 4
# Second derivative
dy_dx2 = torch.autograd.grad(dy_dx,x)[0]
print(dy_dx2) # expected: 12
"""
#problem 12.2
"""x = torch.tensor(0.,requires_grad=True)
y = torch.sin(x)
# First derivative (dy/dx)
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print (dy_dx) # expected: 1
# Second derivative
dy_dx2 = torch.autograd.grad(dy_dx,x)[0]
print(dy_dx2) # expected: 0"""

#Level 13: .detach() and torch.no_grad()
#e.g. 13.1: following code uses detach to remove computation graph from x in order to save memory
"""x = torch.linspace(0,1,10).requires_grad_(True)
y = x**2
z = torch.sum(y) 
grad = torch.autograd.grad(z,x)
print(grad)
# if the part that requires grad is finished, we need only tensor for remaining part we do:
x_val = x.detach()
print(x_val) # no grad_fn etc.
print(x_val.requires_grad) # output: False"""

#e.g. 13.2: following code uses torch.no_grad() to create block without computation graph draining memory and speed
x = torch.linspace(0,1,10).requires_grad_(True)
y = x**2
z = torch.sum(y) 
grad = torch.autograd.grad(z,x)
print(grad)
with torch.no_grad():
    z = x**3
    print(z.requires_grad)