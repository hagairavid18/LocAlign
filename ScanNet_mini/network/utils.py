from keras.layers import Layer
from keras.constraints import Constraint
from keras.initializers import Initializer
import numpy as np
from keras import ops

class Init2Value(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None):
        return self.value.astype(np.float32)    


class ConstraintBetween(Constraint):
    def __init__(self, minimum=-1, maximum=+1):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, w):
        return ops.clip(w, self.minimum, self.maximum)



class FixedNorm(Constraint):
    def __init__(self, value=1.0, axis=0):
        self.axis = axis
        self.value = value

    def __call__(self, w):
        return w * ops.array(self.value, dtype='float32') / (
            1e-7 + ops.sqrt(
                ops.sum(
                    ops.square(w), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis, 'value': self.value}




def embeddings_initializer(shape, dtype=None):
    if shape[0] == shape[1] + 1:
        init = np.zeros(shape, dtype=np.float32)
        for i in range(1, shape[0]):
            init[i, i - 1] = np.sqrt(shape[1])
    else:
        init = np.random.randn(*shape).astype(np.float32)
        init[0, :] *= 0
        init[1:, :] /= np.sqrt((init[1:, :]**2).mean(0))[np.newaxis, :]
    return ops.array(init,dtype=dtype)


