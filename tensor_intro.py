# THis file is intro to basic tensor operations in pytorch
#level 1: tensor
import torch
"""a = torch.tensor(7.5) # scalar
b = torch.tensor([10,20,30,40]) 
c = torch.tensor([[1,2,3],[4,5,6]])
print(a, ":datatype is:", a.dtype)
print("a has dimension:", a.ndim)
print(b, ":datatype is:", b.dtype)
print("b has dimension:", b.ndim)
print(c, ":datatype is:", c.dtype)
print("c has dimension:", c.ndim)"""
#Output:
#tensor(7.5000) :datatype is: torch.float32
#a has dimension: 0
#tensor([10, 20, 30, 40]) :datatype is: torch.int64
#b has dimension: 1
#tensor([[1, 2, 3],
#        [4, 5, 6]]) :datatype is: torch.int64
#c has dimension: 2

#level 2: built-in-tensors (torch.zeros, torch.ones)
"""a = torch.zeros(5) # 1D tensor of zeros
b = torch.ones(3,3) # 2D tensor of ones
c = torch.zeros(2,2,2) # 3D tensor of ones
print(a)
print(b)
print(c)"""

"""# level 3: torch.linspace()
a = torch.linspace (0,10,10) # 10 evenly spaced points from 0 to 10
b = torch.linspace (-1,1,50) # 50 evenly spaced points from -1 to 1
pi = 3.14159
c = torch.linspace (0, 2*pi,100) # 100 evenly spaced points from 0 to 2*pi
print(a)
print(b[24],b[25]) #middle values of b"""
 
"""# level 4: torch.shape() e.g. (6,), (3,4)
# shape is very important in ML as most error comes from shape mismatch
a = torch.ones(6) # 1D tensor of ones
b = torch.ones(3,4) # 2D tensor of ones
c = torch.linspace (0,1,100) # 1D tensor of 100 evenly spaced points from 0 to 1
print(a.shape) # torch.Size([6])
print(b.shape) # torch.Size([3, 4])
print(c.shape) # torch.Size([100])"""

"""#level 5: torch.reshape() 
# e.g.NN expects specific shapes if u have 1d shape, u may convert into 2d shape using reshape
a = torch.ones(12) # 1D tensor 12 elements
b = a.reshape(3,4) # 2D tensor 3 rows and 4 columns
c = torch.linspace(0,1,100) # 1D tensor of 100 evenly spaced points from 0 to 1
d = c.reshape(-1,1) # 2D tensor of 100 rows and 1 column
print(f"shape of\n a: {a.shape},\n b: {b.shape},\n c: {c.shape},\n d: {d.shape}")
# bonus: what does -1 mean in reshape? 
# using -1 in a dimension adjusts that dimension automatically based on the other dimensions and the total number of elements in the tensor.
e =c.reshape(10,-1) # 2D tensor of 10 rows and 10 columns
f =c.reshape(-1,10) # 2D tensor of 10 rows and 10 columns
print(f"shape of\n e: {e.shape},\n f: {f.shape}")"""

# level 6: torch.zeros_like() and torch.ones_like()
# manage shape from existing tensor
# prevents shape mismatch error
# more of dynamic style rather than passive entry if shape each time
"""a = torch.ones(3,3) # (3,3) tensor of ones
b = torch.zeros_like(a) # (3,3) tensor of zeros
c = torch.linspace(0,1,10).reshape(-1,1) #linspace but of shape (10,1)
d = torch.ones_like(c) # (10,1) tensor of ones
print(f"shape of\n a: {a.shape},\n b: {b.shape},\n c: {c.shape},\n d: {d.shape}")"""

# level 7: .squeeze() and .unsqueeze()
# squeeze removes dimensions of size 1, unsqueeze adds a dimension of size 1 at the specified position
# to regulate compressed or extra dimension
"""x = torch.zeros(5,1) # shape is (5,1)
x1 = x.squeeze() # shape is (5,)
y = torch.tensor([1,1,2,3]) # dimension=1, shape is (4,)
y1 = y.unsqueeze(0) #torch.size([1, 4])
y2 = y.unsqueeze(1) #torch.size([4, 1])
z = torch.zeros(1,3,1) # shape is (1,3,1)
z1 = z.squeeze() # shape is (3,)
print(f"shape of\n a: {x.shape},\n b: {x1.shape},\n c: {y.shape},\n d: {y1.shape},\n e: {y2.shape},\n f: {z1.shape}")"""

# level 8: Basic arithmetic operations on tensors
"""all math operations are element-wise.
pow(n) is equivalent to **n but cleaner to chain.
.mean(),.sum(),.max(),.min() map tensor to scalar.
mean,sum are required for backward propagation in ML 
because they are differentiable, max and min are not differentiable."""
a = torch.tensor([1.,2.,3.])
b = torch.tensor([2.,2.,2.])
Loss = (a-b).pow(2).mean() # MSE loss
print(f"Loss: {Loss}") 
x = torch.linspace(0,1,5)
y = x.pow(2).sum() # sum of squares
print(f"sum of squares: {y}") 

# Thank u for visiting my code. I hope u have learned something new.
# The next part will cover gradient