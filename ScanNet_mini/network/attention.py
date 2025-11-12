from keras.layers import Layer
from keras import ops
from network.utils import FixedNorm
from keras.initializers import Zeros

class AttentionPooling(Layer):
    def __init__(self, N_values = 32, N_heads = 1,
                 kernel_regularizer=None,kernel_norm=None,**kwargs):
        self.support_masking = True
        self.N_values = N_values
        self.N_heads = N_heads
        self.kernel_regularizer = kernel_regularizer
        self.kernel_norm = kernel_norm
        self.epsilon = ops.array(1e-6)
        super(AttentionPooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        neighborhoods_shape, features_shape = input_shape
        batch_size,N_after,K = neighborhoods_shape
        batch_size,N_before,N_features = features_shape

        self.value_kernel = self.add_weight(
            shape=[N_features, self.N_values],name='value_kernel',
            constraint=FixedNorm(self.kernel_norm,axis=0) if self.kernel_norm is not None else None,
            regularizer=self.kernel_regularizer,
            
        )
        self.key_kernel = self.add_weight(
            shape=[N_features, self.N_heads],name='key_kernel',
            regularizer=self.kernel_regularizer,
            initializer = Zeros()   
        )
        
        
    def call(self,inputs,mask=None):
        neighborhoods, features = inputs
        if mask is None:
            mask_token_after, mask_token_before = None, None
        else:
            mask_token_after,mask_token_before = mask
                                            
        batch_size,N_after,K = neighborhoods.shape
        batch_size,N_before,N_features = features.shape
        if batch_size is None:
            batch_size = -1
                    
                
        values = ops.dot(features, self.value_kernel)
        keys = ops.dot(features, self.key_kernel)        
        values_and_keys = ops.concatenate([values,keys],axis=-1)
        
        values_and_keys_stacked = ops.reshape(
        ops.take_along_axis(
            values_and_keys,ops.reshape( neighborhoods,
                        [batch_size,N_after*K,1])
                        ,axis=1),
        
        [batch_size,N_after,K, self.N_values+self.N_heads]
        )
        
        values_stacked = ops.reshape(
                    values_and_keys_stacked[:,:,:,:self.N_values],
                    [batch_size,N_after,K,self.N_heads, self.N_values//self.N_heads]
        )
        
        keys_stacked = values_and_keys_stacked[:,:,:,self.N_values:]
        
        mask_stacked =  ops.not_equal(neighborhoods , -1)
        if mask_token_before is not None:
            mask_stacked &= ops.reshape(
        ops.take_along_axis(
            mask_token_before,ops.reshape( neighborhoods,
                        [batch_size,N_after*K]),axis=1),
        [batch_size,N_after,K])
            
        # attention_coefficients =  ops.exp( keys_stacked - ops.amax(keys_stacked,axis=-2,keepdims=True)  )
        # attention_coefficients = attention_coefficients * ops.expand_dims(ops.cast(mask_stacked,dtype="float32"),axis=-1)
        
        attention_coefficients =  ops.exp( keys_stacked - ops.amax(keys_stacked,axis=-2,keepdims=True) )        
        attention_coefficients = attention_coefficients * ops.expand_dims(ops.cast(mask_stacked,dtype="float32"),axis=-1)        
        attention_coefficients = attention_coefficients / (ops.sum(attention_coefficients,axis=-2,keepdims=True) + self.epsilon)
        
        output = ops.reshape(
            ops.einsum('bikhv,bikh->bihv', values_stacked,attention_coefficients),
            [batch_size,N_after,self.N_values])
        
        if mask_token_after is not None:
            output *= ops.expand_dims(ops.cast(mask_token_after,dtype="float32"),axis=-1)            
        return output
    
    
    def compute_output_shape(self, input_shape):
        neighborhoods_shape = input_shape[0]
        batch_size,N_after,K = neighborhoods_shape
        return (batch_size,N_after,self.N_values)

    def compute_mask(self, input, mask=None):
        if isinstance(mask,list):
            return mask[0]
        else:
            return mask
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'N_values': self.N_values,
            'N_heads': self.N_heads,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_norm': self.kernel_norm,
            })
        return config
            

class AttentionLayer(Layer):
    def __init__(self, self_attention=True,beta=True,**kwargs):
        self.support_masking = True
        self.self_attention = self_attention
        self.beta = beta
        self.epsilon = ops.array(1e-6)
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.beta & self.self_attention:
            beta_shape, self_attention_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        elif self.beta & (~self.self_attention):
            beta_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        elif (~self.beta) & self.self_attention:
            self_attention_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        else:
            attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape

        self.Lmax = graph_weights_shape[1]
        self.Kmax = graph_weights_shape[2]
        self.nheads = attention_coefficient_shape[-1]
        assert self.nheads == graph_weights_shape[-1]
        self.nfeatures_output = node_activity_shape[-1] // self.nheads
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs,mask=None):
        if self.beta & self.self_attention:
            beta, self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        elif self.beta & (not self.self_attention):
            beta, attention_coefficients, node_outputs, graph_weights = inputs
        elif (not self.beta) & self.self_attention:
            self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        else:
            attention_coefficients, node_outputs, graph_weights = inputs

        # if self.beta:
        #     beta = ops.reshape(
        #         beta, [-1, self.Lmax, self.nfeatures_graph, self.nheads])
        # if self.self_attention:
        #     self_attention = ops.reshape(
        #     self_attention, [-1, self.Lmax, self.nfeatures_graph, self.nheads])
        # attention_coefficients = ops.reshape(
        #     attention_coefficients, [-1, self.Lmax, self.Kmax, self.nfeatures_graph, self.nheads])
        # node_outputs = ops.reshape(
        #     node_outputs, [-1, self.Lmax, self.Kmax, self.nfeatures_output, self.nheads ])
        node_outputs = ops.reshape(
            node_outputs, [-1, self.Lmax, self.Kmax, self.nheads , self.nfeatures_output ])
        # Add self-attention coefficient.
        if self.self_attention:
            # (attention_coefficients_self, attention_coefficient_others) = ops.split(
            #     attention_coefficients, [1, self.Kmax], axis=2)
            attention_coefficients_self = ops.take(attention_coefficients,[0],axis=2)
            attention_coefficient_others = ops.take(attention_coefficients,range(1,self.Kmax),axis=2)
            attention_coefficients_self = attention_coefficients_self+ops.expand_dims(self_attention, axis=2)
            attention_coefficients = ops.concatenate(
                [attention_coefficients_self, attention_coefficient_others], axis=2)
        # Multiply by inverse temperature beta.
        if self.beta:
            attention_coefficients = attention_coefficients * ops.expand_dims(beta + self.epsilon, axis=2)

        ##
        # attention_coefficients = attention_coefficients - ops.amax(
        #     attention_coefficients, axis=[-3, -2], keepdims=True)
        # attention_coefficients_final = ops.sum(ops.expand_dims(
        #     graph_weights, axis=-1) * ops.exp(attention_coefficients), axis=-2)
        # attention_coefficients_final = attention_coefficients_final/ ( ops.sum(
        #     ops.abs(attention_coefficients_final), axis=-2, keepdims=True) + self.epsilon )

        attention_coefficients_final = graph_weights * ops.exp( attention_coefficients - ops.amax(attention_coefficients, axis=2, keepdims=True) )
        attention_coefficients_final = attention_coefficients_final/ ( ops.sum(ops.abs(attention_coefficients_final), axis=2, keepdims=True) + self.epsilon )
        
        output_final = ops.reshape(
            ops.einsum('bikhf,bikh->bihf', node_outputs, attention_coefficients_final),
            [-1, self.Lmax, self.nfeatures_output * self.nheads])        
        # output_final = ops.reshape(ops.sum(node_outputs * ops.expand_dims(
        #     attention_coefficients_final, axis=-2), axis=2), [-1, self.Lmax, self.nfeatures_output * self.nheads])
        return [output_final, attention_coefficients_final]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.Lmax, self.nfeatures_output * self.nheads),
                (input_shape[0][0], self.Lmax, self.Kmax, self.nheads)]

    def compute_mask(self, input, mask=None):
        if mask not in [None,[None for _ in input]]:
            if self.beta | self.self_attention:
                return [mask[0], None]
            else:
                return [mask[0][...,0],None]
        else:
            return mask




