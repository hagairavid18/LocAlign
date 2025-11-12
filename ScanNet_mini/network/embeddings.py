from keras import ops
from keras.src import backend
from keras.constraints import NonNeg
from keras.initializers import Zeros,TruncatedNormal
import numpy as np
from sklearn.mixture import GaussianMixture
import keras
from network.utils import Init2Value,FixedNorm,ConstraintBetween
from keras.layers import Layer

class GaussianKernel(Layer):
    def __init__(self, N, initial_values, covariance_type='diag', eps=1e-1, **kwargs):
        super(GaussianKernel, self).__init__(**kwargs)
        self.support_masking = True
        self.eps = eps
        self.N = N
        self.initial_values = initial_values
        self.covariance_type = covariance_type
        assert self.covariance_type in ['diag', 'full']

    def build(self, input_shape):
        self.nbatch_dim = len(input_shape) - 1
        self.d = input_shape[-1]

        self.center_shape = [self.d, self.N]

        self.centers = self.add_weight(shape=self.center_shape, name='centers',
                                       initializer=Init2Value(
                                           self.initial_values[0]),
                                       regularizer=None,
                                       constraint=None)

        if self.covariance_type == 'diag':
            self.width_shape = [self.d, self.N]

            self.widths = self.add_weight(shape=self.width_shape,
                                          name='widths',
                                          initializer=Init2Value(
                                              self.initial_values[1]),
                                          regularizer=None,
                                          constraint=NonNeg())

        elif self.covariance_type == 'full':
            self.sqrt_precision_shape = [self.d, self.d, self.N]

            self.sqrt_precision = self.add_weight(shape=self.sqrt_precision_shape,
                                                  name='sqrt_precision',
                                                  initializer=Init2Value(
                                                      self.initial_values[1]),
                                                  regularizer=None,
                                                  constraint=ConstraintBetween(-1/self.eps,1/self.eps))

        # Set input spec
        super(GaussianKernel, self).build(input_shape)

    def call(self, inputs, mask=None):
        if self.covariance_type == 'diag':
            activity = ops.exp(- 0.5 * ops.sum(
                (
                    (
                        ops.expand_dims(inputs, axis=-1)
                        - ops.reshape(self.centers,
                                    [1 for _ in range(self.nbatch_dim)] + self.center_shape)
                    ) / ops.reshape(self.eps + self.widths, [1 for _ in range(self.nbatch_dim)] + self.width_shape)
                )**2, axis=-2))

        elif self.covariance_type == 'full':
            # intermediate = ops.expand_dims(inputs, axis=-1) - ops.reshape(self.centers,
            #                                                           [1 for _ in range(self.nbatch_dim)] + self.center_shape)  # B X d X n_centers

            # intermediate2 = ops.sum(
            #     ops.expand_dims(intermediate, axis=-3) *
            #     ops.expand_dims(self.sqrt_precision, axis=0),
            #     axis=- 2)
            
            
            intermediate2 = ops.einsum('...i,lij->...lj', inputs,self.sqrt_precision) - ops.reshape(ops.einsum('ij,lij->lj',self.centers, self.sqrt_precision),[1 for _ in range(self.nbatch_dim)] + self.center_shape)

            activity = ops.exp(- 0.5 * ops.sum(intermediate2**2, axis=-2))

        return activity

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[:-1]) + [self.N]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def get_config(self):
        config = {'N': self.N,
                  'initial_values': keras.saving.serialize_keras_object(self.initial_values),
                  'covariance_type': self.covariance_type}
        base_config = super(
            GaussianKernel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        initial_values_config = config.pop('initial_values')
        initial_values = keras.saving.deserialize_keras_object(initial_values_config)
        return cls(initial_values=initial_values, **config)



def inv_root_matrix(H):
    lam, v = np.linalg.eigh(H)
    return np.dot(v,  1 / np.sqrt(lam)[:, np.newaxis] * v.T)


def initialize_GaussianKernel(points, N,covariance_type='diag',reg_covar=1e-1,n_init=10):
    GMM = GaussianMixture(n_components=N, covariance_type=covariance_type, verbose=1,
                          reg_covar=reg_covar,n_init=n_init)
    GMM.fit(points)
    centers = GMM.means_
    covariances = GMM.covariances_
    probas = GMM.weights_
    order = np.argsort(probas)[::-1]
    centers = centers[order]
    covariances = covariances[order]
    probas = probas[order]
    if covariance_type == 'diag':
        widths = np.sqrt(covariances)
    elif covariance_type == 'full':
        sqrt_precision_matrix = np.array(
            [inv_root_matrix(covariance) for covariance in covariances])
    if covariance_type == 'diag':
        return centers.T, widths.T
    elif covariance_type == 'full':
        return centers.T, sqrt_precision_matrix.T


def initialize_GaussianKernelRandom(xlims, N, covariance_type):
    xlims = np.array(xlims,dtype=np.float32)
    coordinates_dimension = xlims.shape[0]

    centers = np.random.rand(coordinates_dimension, N).astype(np.float32)
    centers = centers * (xlims[:,1]-xlims[:,0])[:,np.newaxis] + xlims[:,0][:,np.newaxis]

    widths = np.ones([coordinates_dimension, N], dtype=np.float32)
    widths = widths * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] / (N / 4)

    if covariance_type == 'diag':
        initial_values = [centers,widths]
    else:
        sqrt_precision_matrix = np.stack([np.diag( 1.0/(1e-4+widths[:,n]) ).astype(np.float32) for n in range(N)],axis=-1)
        initial_values = [centers,sqrt_precision_matrix]
    return initial_values





class OuterProduct(Layer):
    def __init__(self, n_filters, 
                 use_single1=True,
                 use_single2=True,
                 use_bias=True,
                 fixednorm=None,
                 kernel_regularizer=None,
                 single1_regularizer=None,
                 single2_regularizer=None,
                 sum_axis=None, **kwargs):
        super(OuterProduct, self).__init__(**kwargs)
        self.support_masking = True
        self.n_filters = n_filters
        self.use_single1 = use_single1
        self.use_single2 = use_single2
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.single1_regularizer = single1_regularizer
        self.single2_regularizer = single2_regularizer
        self.fixednorm = fixednorm
        self.sum_axis = sum_axis


    def build(self, input_shape):
        if self.fixednorm is not None:
            constraint_kernel = FixedNorm(value=self.fixednorm, axis=[0, 1])
        else:
            constraint_kernel = None
            
        self.n1 = input_shape[0][-1]
        self.n2 = input_shape[1][-1]
        if self.fixednorm is not None:
            stddev = self.fixednorm / np.sqrt(self.n1 * self.n2)
        else:
            stddev = 1.0 / np.sqrt(self.n1 * self.n2)


        initializer = TruncatedNormal(mean=0., stddev=stddev)

        self.kernel12 = self.add_weight(
            shape=[self.n1, self.n2, self.n_filters],
            name='kernel12',
            initializer=TruncatedNormal(
                mean=0., stddev=stddev),
            constraint=constraint_kernel,
            regularizer=self.kernel_regularizer
        )

        if self.use_single1:
            stddev = 1.0 / np.sqrt(self.n1)
            initializer = TruncatedNormal(mean=0., stddev=stddev)

            self.kernel1 = self.add_weight(
                shape=[self.n1, self.n_filters],
                name='kernel1',
                initializer=initializer,
                constraint=None,
                regularizer = self.single1_regularizer
            )
        if self.use_single2:
            stddev = 1.0 / np.sqrt(self.n2)
            initializer = TruncatedNormal(mean=0., stddev=stddev)

            self.kernel2 = self.add_weight(
            shape=[self.n2, self.n_filters],
            name='kernel2',
            initializer=initializer,
            constraint=None,
            regularizer = self.single2_regularizer
            )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.n_filters, ],
                name='bias',
                initializer=Zeros(),
                constraint=None
            )

        # Set input spec
        super(OuterProduct, self).build(input_shape)

    def call(self, inputs, mask=None):
        first_input = inputs[0]
        second_input = inputs[1]
        bias_shape = [1 for _ in first_input.shape[:-1]] + [self.n_filters]
        if self.sum_axis is not None:
            del bias_shape[self.sum_axis]

        if self.sum_axis is not None:
            if self.sum_axis == 2:
                activity = ops.einsum('...ki,...kj,ijl->...l',first_input,second_input, self.kernel12)
            else:
                outer_product = ops.sum(ops.expand_dims(
                    first_input, axis=-1) * ops.expand_dims(second_input, axis=-2), axis=self.sum_axis)
                activity = ops.tensordot(outer_product, self.kernel12, [[-2, -1], [0, 1]])
        else:
            outer_product = ops.expand_dims(
                first_input, axis=-1) * ops.expand_dims(second_input, axis=-2)
            
            # if self.sum_axis is not None:
            #     if self.sum_axis == 2:
            #         outer_product=ops.einsum('...ki,...kj->...ij',first_input,second_input)
            #     else:                    
            #         outer_product = ops.sum(ops.expand_dims(
            #             first_input, axis=-1) * ops.expand_dims(second_input, axis=-2), axis=self.sum_axis)

            # activity = ops.tensordot(
            #     outer_product, self.kernel12, [[-2, -1], [0, 1]])

        if self.use_single1:
            if self.sum_axis is not None:
                activity += ops.dot(ops.sum(first_input,
                                                axis=self.sum_axis), self.kernel1)
            else:
                activity += ops.dot(first_input, self.kernel1)
        if self.use_single2:
            if self.sum_axis is not None:
                activity += ops.dot(ops.sum(second_input,
                                                axis=self.sum_axis), self.kernel2)
            else:
                activity += ops.dot(second_input, self.kernel2)
        if self.use_bias:
            activity += ops.reshape(self.bias, bias_shape)
        return activity

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0]] + [max(shape1, shape2) for shape1, shape2 in zip(input_shape[0][1:-1],
                                                                                            input_shape[1][1:-1])] + [self.n_filters]
        if self.sum_axis is not None:
            del output_shape[self.sum_axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if isinstance(mask,list):
            condition = mask[0] is not None
        else:
            condition = mask is not None
        if condition:
            if self.sum_axis is not None:
                return mask[0][..., 0]
            else:
                return mask[0]
        else:
            return mask

    def get_config(self):
        config = {'n_filters': self.n_filters,
                  'use_single1': self.use_single1,
                  'use_single2': self.use_single2,
                  'use_bias': self.use_bias,
                  'fixednorm': self.fixednorm,
                  'sum_axis': self.sum_axis,                  
                 'kernel_regularizer':self.kernel_regularizer,
                 'single1_regularizer':self.single1_regularizer,
                 'single2_regularizer':self.single2_regularizer,
                  }
        base_config = super(
            OuterProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedDense(keras.layers.Dense):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * ops.expand_dims(ops.cast(mask, inputs.dtype),-1)
        outputs = super().call(inputs)
        return outputs        
                        
    def compute_mask(self, inputs, mask=None):
        return mask

class MaskedConcatenate(keras.layers.Concatenate):
    def compute_mask(self, inputs, mask=None):
        if isinstance(mask,list):
            return mask[0]
        else:
            return mask
    
class MaskedActivation(keras.layers.Activation):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * ops.expand_dims(ops.cast(mask, inputs.dtype),-1)
        outputs = super().call(inputs)
        return outputs
                        
    def compute_mask(self, inputs, mask=None):
        return mask
    

class MaskedRescaling(keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(MaskedRescaling, self).__init__(**kwargs)
        self.support_masking = True
        self.scale = scale
        # self.scale = ops.array(scale,dtype='float32')

    def call(self, inputs, mask=None):
        # outputs = inputs * self.scale
        outputs = inputs * ops.array(self.scale, dtype='float32')
        if mask is not None:
            outputs *= ops.expand_dims(ops.cast(mask, outputs.dtype),-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def compute_mask(self, inputs, mask=None):
        return mask    

    def get_config(self):
        config = super(MaskedRescaling, self).get_config()
        config.update({
            'scale': self.scale
        })
        return config
    
    
class MaskandClip(keras.layers.Layer):
    def __init__(self, mask_value=-1,clip_min=-1,clip_max=1024, **kwargs):
        super(MaskandClip, self).__init__(**kwargs)
        self.support_masking = True
        self.mask_value = mask_value
        self.clip_min = clip_min
        self.clip_max = clip_max
        # self._build_at_init()
        
    def call(self, inputs):        
        outputs = ops.cast(ops.clip( inputs, self.clip_min,self.clip_max), inputs.dtype)
        boolean_mask = ops.any(ops.not_equal(inputs, self.mask_value), axis=-1)
        # backend.set_keras_mask(outputs, mask=boolean_mask)
        return outputs

    # def call(self, inputs):
    #     boolean_mask = ops.any(
    #         ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True
    #     )
    #     # Set masked outputs to 0
    #     outputs = inputs * backend.cast(boolean_mask, dtype=inputs.dtype)
    #     # Compute the mask and outputs simultaneously.
    #     backend.set_keras_mask(outputs, mask=ops.squeeze(boolean_mask, axis=-1))
    #     return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
    
    # @property
    # def compute_dtype(self):
    #     return 'int32'
    
    # def compute_output_spec(self, input_spec):
    #     # Define the output shape and dtype
    #     return input_spec
    
    def compute_mask(self, inputs, mask=None):
        boolean_mask = ops.any(ops.not_equal(inputs, self.mask_value), axis=-1)
        return boolean_mask

    def get_config(self):
        config = {'mask_value': self.mask_value,'clip_min':self.clip_min,'clip_max':self.clip_max}
        base_config = super(MaskandClip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))