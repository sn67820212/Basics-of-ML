# part 4 of series: training
# The model we developed so far wouldn't give accurate output.
# It happens because we built a random model with layers.
# We need to train our model using existing data.
# The neural network usesback propagation technique for training a model.
# To get more intuitive knowledge you can watch 3 Blue 1 Brown's video available on youtube.

import torch.nn as nn
import torch

# Level 19: Adam optimizer
# Adam optimizer uses robust discrete size for steps to reduce loss function in our model.
# loss function is the measure of error of our model with respect to existing data
# learning rate (lr) gives step size but Adam decides which parameter needs more step size.
# the overall learning rate is decided by Adam and lr both. Mathematically, 
# p_(n+1) = p_n - lr* adjusted_gradient/adaptive_scaling. Where, p is a parameter element. 
# This is the mental picture u need to know to use it is:
# optimizer changes the value of parameters in model so that the loss gets minimized. 
# Everything else will become clear ehen we use it in combo with loss.backward() later
 
# e.g.
"""model = nn.Sequential(nn.Linear(1,20),nn.Tanh(),nn.Linear(20,1))
optimizer = torch.optim.Adam(
    model.parameters(), # what to update
    lr = 1e-3           # learning rate
)
print (optimizer) """# shows config of adam optimizer
# The key things to mention before training a model using optimizer:
# We do not want to keep lr neither too low nor to high because:
# Too low lr makes learning too slow and requires more time and memory to obtain good result.
# If lr is too high model gets a problem known as overshooting.
# Overshooting forces our updated value to extend beyond the true minima as step size was too large.
# If gradient is larger then high lr increases step size further then the next step keeps going to the reason with large error  rather than minimizing it.
# Similarly, there is another thing to mention:
# If the number of parameter is too low or too high then there arise two unwanted scenarios:
# If it's too low, the model can't model the non lineaar data properly giving rise to underfit the data.
# It it's too high, the model fits the noises thus giving rise to overfitting which makes our model useless to new data.

# Level 20: The resolution of gradient accumulation bug
# We had discussed gradient accumulation previously, and saw it's useful in our training as, 
# we want to minize overall loss due to parameters.
# But, we want to reset the gradient to zero many times if:
# we forget to do so, we will get a bug in our training process.
# Let's see simplest code to highlight this bug and then propose the solution to the bug.

"""x = torch.tensor(3.14, requires_grad=True)
y = torch.sin(x) # y = sin x
# first time grad calculation
y.backward() # dy/dx , this stores gradient as grad = -1. and never gets reset in new .backward() call
print(x.grad) # grad is -1
# Bug: forgot to reset grad and do derivative second time
y = -x**2 
y.backward() # derivative is -6.28 but let's check grad output
print(x.grad) # output is:-6.28+(-1)=-7.28 where,-1 is from previous step
# Fix: to obtain the exaxt value for the latest derivatives only we do
x.grad = None # reset gradient of 1 parameter i.e. x 
# to reset gradient of parameters inside the optimization block we use syntax: optimizer.zero_grad()
y = x**2
y.backward()
print(x.grad) # desired output"""

# Level 21: loss.backward() combo with optimizer.step()
# these are the steps that actually train a model so we will do it in well ordered way rather than showing syntaxes only
# step 1: Define model and optimizer
model = nn.Sequential(nn.Linear(1,10),nn.Tanh(),nn.Linear(10,1))
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# step 2: enter Input and output
"""x = torch.tensor([[1.0],[2.0],[3.0]])
target = torch.tensor([[2.0],[4.0],[6.0]]) # y = 2*x 
# step 3: training using loss.backward() and optimizer.step()
# train model 5 times by keeping track of error to show if it works or not 
for step in range(5): 
    pred = model(x) # pred means prediction from model
    loss = (pred - target).pow(2).mean() # average square loss
    optimizer.zero_grad() # clear value stored in grad in previous loop
    loss.backward() # compute gradient of loss
    optimizer.step() # automatically reduces loss using gradient descent technique
    print (f"step: {step},loss={loss.item()}")"""
# Note: .item() converts 1*1 tensor into scalar. That's it, u can try it yourselfby removing it and cross checking outputs.
# The loss got reduced in each code but u may have observed loss is still large after 5 loop training.
# In real problems, we use thousands of loop to improve model efficiently.

# This is the core of neural network. I recommend you to increase loop and test the model by:
# inserting new inputs and testing accuracy 
#
# The next part of my series is designated to PINNs (Physics Infromed Neural Networks where I will solve few problems)