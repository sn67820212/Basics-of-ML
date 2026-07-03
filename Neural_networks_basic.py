# part 3 of series: Neural network
import torch.nn as nn
import torch

# Level 15: nn.Linear(in,out)
# output = x @ W^T + b
# W is weight and b is basis.
# W and b are adjustablw parameters.
# The core goal of NN is to adjust parameter and transformation till error gets minimized

#e.g. 15.1
"""layer = nn.Linear(2,5) # 2 inputs, 5 outputs
x = torch.tensor([1.,2.]) # (1,2) tensor
y = layer(x)
print(y.shape)
print(y)"""

"""#e.g. 15.2
layer = nn.Linear(3,1) # 3 inputs, 1 output
x = torch.ones(10,3)
y = layer(x)
print(y.shape)"""

"""# e.g. layer.weight and layer.bias i.e. w,b
layer = nn.Linear(1,4)
print(layer.weight)
print(layer.bias)
# Note: the parameters are initialized randomly"""

# Level 16: activation functions: e.g. nn.tanh, nn.ReLU, nn.Sigmoid
# no matter how much linear transformation codes you use, it's reducible to one linear transformation.
# Activations introduce nonlinearity so network can approximate curved functions in better way.
# tanh is preferred in physics because it is smooth and has smooth derivatives

# e.g. nn.Tanh()
"""x = torch.linspace(-3,3,7) 
tanh = nn.Tanh()
print(tanh(x))"""

#e.g.nn.ReLU()
"""x = torch.tensor([-3,-1,0,1,3])
relu = nn.ReLU()
print(relu(x)) #clips negative to zero"""

#e.g.nn.sigmoid()
"""x = torch.tensor([-3,-1,0,1,3])
sigmoid = nn.Sigmoid()
print(sigmoid(x))""" # 0 to 1
# For physics informed neural network Tanh is preferred
# I mention physics in different parts because it's my code and my choice.
# Another thing u would have thought is why I introduced non linear transformations like tanh etc. 
# You will realize later and i will leave a gap here.

# Level 17: nn.sequential
# nn.sequential allows the transformation of input into output via a set of layers.
# Why do we introduce these layers ?
# Because we want to approximate our data and find the optimum layer structre.
# Having multiple layers allows us to approximate more independent parameters and:
# more flexibility means it can model more accurately.
# If u are confused, you should watch 3B1b

# e.g 17.1: 3 layer model
"""model = nn.Sequential(
    # layer 1
    nn.Linear(1,10), # 1 input 20 output
    nn.Tanh(), # non linearity to approximate non linear relationship between input and output
    # layer 2
    nn.Linear(10,10),
    nn.Tanh(),
    # layer 3
    nn.Linear(10,1)
)
x = torch.linspace(0,1,4).reshape(-1,1)
print (model(x))
y = torch.tensor([[1.],[2.],[3.],[4.]])
print(model(y))"""
# Note: using integer data instead of floating value causes an error as pytorch expects it, u can test it yourself

# e.g 17.1: Wider network of 1 to 50 to 1
"""model = nn.Sequential(
    # layer 1
    nn.Linear(1,50), # 1 input 50 output
    nn.Tanh(), # non linearity to approximate non linear relationship between input and output
    # layer 2
    nn.Linear(50,50),
    nn.Tanh(),
    # layer 3
    nn.Linear(50,1)
)
x = torch.linspace(0,1,4).reshape(-1,1)
print (model(x))
y = torch.tensor([[1.],[2.],[3.],[4.]])
print(model(y))"""

# Level 18: model.parameters()
# model.parameters() returns an iterator over all learnable parameters
# Learnable parameters are parameters are the parameters nn defines in each layer to obtain transformation.
# Having too much low number of parameters makes the model less reliable to new data.
# Also, we don't want to feed network too much redundant data.
# A good model uses a sweet spot between these two i.e. neither too high parameters nor too low
 
# example code to show key ideas:
model = nn.Sequential(nn.Linear(1,50), nn.Tanh(), nn.Linear(50,1)) # define a model
# to count total parameters in a model, we can the syntax use p.numel()
total = sum(p.numel() for p in model.parameters())
print(total)
# Alternative code if u have been using loop in non compact way
"""total = 0
for j in model.parameters():
    total = total +j.numel()
print (total)
print(model.parameters())"""

# Let us check if each parameter has requires_grad = True
for p in model.parameters():
    print(p.shape,p.requires_grad)
# The output is:
"""torch.Size([50, 1]) True # weight of layer 1
torch.Size([50]) True # basis of layer 1
torch.Size([1, 50]) True # weight of layer 2
torch.Size([1]) True # basis of layer 2""" 

# This is the basic course on neural network. 
# Next tutorial gives you the training part which optimizes the neural network for given data. 
# Thank you for your patience.