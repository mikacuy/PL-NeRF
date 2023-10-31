import numpy as np

def update_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

### Learning rate decay
# LR Decay
def get_learning_rate(init_learning_rate, iteration_num, decay_step, decay_rate, staircase=True):
    p = iteration_num / decay_step
    if staircase:
        p = int(np.floor(p))
    learning_rate = init_learning_rate * (decay_rate ** p)
    return learning_rate