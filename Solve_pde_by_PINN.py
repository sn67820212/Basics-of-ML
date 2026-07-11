# part 5: use of nn to solve poisson equation
# The difference with normal nn is we include pde in PINN 
# PINN is simply a neural network with pde as input along with BCs 
# Let's see how we do it.
# to solve u''(x) = -1, u(0) = u(1) = 0 . the analytical solution is simple:
# u'(x) = c/2-x or u(x) = x(c-x**2)/2 + a, BCs give a=0, c = 1
# u(x) = x(1-x)/2 i.e. u(0.5) = 0.125, let's solve it by NN.

import torch
import torch.nn as nn

# step 1: Model and optimizer
model = nn.Sequential(nn.Linear(1,20),nn.Tanh(),nn.Linear(20,1))
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# step 2: sample points and boundary conditions
x   = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
x_b = torch.tensor([[0.0],[1.0]])
u_b = torch.tensor([[0.],[0.]])

# Training loop:
for step in range(2000):
    u = model (x)

    # derivatives
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]
    
    # Losses
    loss_pde = (u_xx+1).pow(2).mean()
    loss_bc = (model(x_b)-u_b).pow(2).mean()
    loss = loss_bc + loss_pde

    # training 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check error at steps{0,500,1000,1500,2000,2500}
    if step % 500 == 0:
        print(f"step {step}: loss = {loss.item()}")
    
# Test the model at x = 0.5
with torch.no_grad():
    test = model(torch.tensor([[0.5]]))
    print(f"u(0.5)={test.item()}")

# This tutorial was designed to help learners to be able understand PINNs
# The last part shows how nn connects in physics where input is pde and BCs
# The objective of course is served. Users are encouraged to do further exploration by themselves.
# If I find something really great for PINN, i will let u know.
# Thank you very much. Have a wonderful time

# Additional: comparision with exact solution via graphical representation
x = torch.linspace(0,1,100)
y1 = x*(1-x)/2
y2 = model(x.reshape(-1,1)).detach().numpy()
import matplotlib.pyplot as plt
plt.plot(x,y1,label='exact')
plt.plot(x,y2,label='PINN')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
