from lr_scheduler import ReduceLROnPlateau

# select the target class
target_class =

# fix the network weights
# ....

# means and std used for the image normalization
mean = 0.1307
std = 0.3081

# allocate memory for the image variable
# NOTE: we need the gradients with respect to the input images
# This requires certain changes in the initialization of the Pytorch Variable
imagevar =

# calculate the gradients of the objective function
# notice that we will use gradient descend algorithm below, so we need to change the
# sign of the objective function
grad = torch.zeros(1, 10).type(torch.FloatTensor)
grad[0, target_class] =

# set learning parameters of the gradien descend
LR = 10  # worked well for me
NUM_ITER =
# start with a black image
imagevar

# you may use SGD optimizer from pytorch or implement the update step by yourself
optimizer = torch.optim.SGD([...], lr=LR, momentum=0.9)
# scheduler track the changes of the objective function and reduces LR when needed
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, min_lr=0.5e-6)

for i in range(NUM_ITER):
    print('[{0:05d}/{1:d}]'.format(i+1, NUM_ITER), end='\r')

    # set gradients of the optimizer to zero
    # ...

    # obtain the softmax avtivations of the last layer
    act_value = ...
    # backpropagate to image domain
    act_value.backward(grad)

    # transmit the current value of the objective function to the scheduler
    scheduler.step(act_value.data[0, target_class-1])
    # make step toward the negative gradients:
    # ....

    # we clip the values of the updated image to the feasible region
    imagevar.data = torch.clamp(imagevar.data, -mean/std, (1-mean)/std)

# show image
# ...
