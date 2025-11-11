from keras.models import Model
from keras.layers import Input, Masking, Concatenate, Activation, Embedding, Dropout, Lambda,BatchNormalization,Reshape
from keras.initializers import Zeros, Ones
import numpy as np
from keras.regularizers import Regularizer
from keras import ops
from network import neighborhoods, attention, embeddings, utils,loss_and_metrics
from utilities import wrappers,io_utils
from keras.optimizers import Adam

class SparseRegularizer(Regularizer):
    def __init__(self, 
                 mode = 'Hoyer',
                 coefficient=0., ndims=2, order='filter_last'):
        assert mode in ['L1','L12','L12_group','Hoyer']
        assert coefficient>0
        assert ndims in [2,3]
        assert order in ['filter_first','filter_last']
        self.mode = mode
        self.coefficient = coefficient
        self.ndims = ndims
        self.order = order
        
    def __call__(self,W):
        if self.mode == 'L1':
            return self.coefficient * ops.sum(ops.abs(W) )
        
        elif self.mode == 'L12':        
            if self.order == 'filter_first':  # Order of the tensor indices.
                if self.ndims == 2: # For gaussian-dependent bias
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[1], 'float32') * ops.sum(ops.square(ops.mean(ops.abs(W), axis=1)))
                elif self.ndims == 3:
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[1] * ops.shape(W)[2], 'float32') * ops.sum(
                        ops.square(ops.mean(ops.abs(W), axis=(1, 2))))
            elif self.order == 'filter_last':  # Default order the tensor indices.
                if self.ndims == 2:
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[0], 'float32') * ops.sum(ops.square(ops.mean(ops.abs(W), axis=0)))
                elif self.ndims == 3:
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[0] * ops.shape(W)[1], 'float32') * ops.sum(
                        ops.square(ops.mean(ops.abs(W), axis=(0, 1))))
                    
        elif self.mode == 'L12_group':            
            if self.ndims == 2: # For gaussian-dependent bias
                return self.coefficient / 2 * ops.cast(ops.shape(W)[1], 'float32') * ops.sum(ops.square(ops.mean(ops.abs(W), axis=1)))
            else:        
                if self.order == 'filter_first': # Order of the tensor indices.
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[1] * ops.shape(W)[2], 'float32') * ops.sum(
                        ops.square(ops.mean(ops.sqrt(ops.mean(ops.square(W), axis=-1)), axis=-1)))
                elif self.order == 'filter_last': # Order of the tensor indices.
                    return self.coefficient / 2 * ops.cast(ops.shape(W)[0] * ops.shape(W)[1], 'float32') * ops.sum(
                ops.square(ops.mean(ops.sqrt(ops.mean(ops.square(W), axis=1)), axis=0)))
                    
        
        elif self.mode == 'Hoyer':            
            if self.order == 'filter_first':  # Order of the tensor indices.
                if self.ndims == 2: # For gaussian-dependent bias
                    return self.coefficient / 2 * ops.sum( ops.square( ops.mean(ops.abs(W),axis=1 ) ) / ( ops.mean( ops.square(W) , axis=1 )  + 1e-4 ) )
                elif self.ndims ==3:
                    return self.coefficient / 2 * ops.sum( ops.square( ops.mean(ops.abs(W),axis=(1,2) ) ) / (ops.mean( ops.square(W) , axis=(1,2) ) + 1e-4 ) )
            elif self.order == 'filter_last':  # Default order the tensor indices.
                if self.ndims == 2: # For gaussian-dependent bias
                    return self.coefficient / 2 * ops.sum( ops.square( ops.mean(ops.abs(W),axis=0 ) ) / (ops.mean( ops.square(W) , axis=0 ) + 1e-4 ) )
                elif self.ndims ==3:
                    return self.coefficient / 2 * ops.sum( ops.square( ops.mean(ops.abs(W),axis=(0,1) ) ) / (ops.mean( ops.square(W) , axis=(0,1) ) + 1e-4 ) )            
        else:
            raise ValueError

    def get_config(self):
        return {'mode': self.mode,
                'coefficient':self.coefficient,
                'ndims': self.ndims,
                'order':self.order}
        
    
  


def addNonLinearity(input_layer, activation, name=None):
    if name is None:
        name = 'activity_nonlinear'
    if activation == 'tanh':
        center = True
        scale = True
    elif activation == 'relu':
        center = True
        scale = False
    elif activation == 'elu':
        center = True
        scale = True
    elif activation in [None,'linear']:
        center = False
        scale = False
    elif activation == 'linear_light_attention':
        center = True
        scale = True
        activation = 'linear'  # Use linear activation, but with a custom attention layer.
    else:
        center = True
        scale = True

    input_layer_normalized = BatchNormalization(
        epsilon=1e-3, axis=-1, center=center, scale=scale, name=name + '_normalization')(input_layer) # Custom batch norm layer that takes into account the mask.

    output_layer = embeddings.MaskedActivation(
        activation, name=name + '_masked')(input_layer_normalized)
    return output_layer


def attribute_embedding(attribute_layer,output_dim, activation,name=None):
    if name is None:
        name = 'attribute_embedding'

    if output_dim is None: # Does nothing. Just passes through mask-preserving identity layer to change name.
        return embeddings.MaskedActivation('linear',name=name)(attribute_layer)
    else:
        projected_attribute_layer = embeddings.MaskedDense(output_dim, use_bias=False, activation=None,
                                                 name=name+'_projection')(attribute_layer)
        embedded_attribute_layer = addNonLinearity(projected_attribute_layer, activation,
                                                  name=name)
        return embedded_attribute_layer



def neighborhood_embedding(
        frames,
        attributes,
        neighbors,
        coordinates='euclidean',
        Ngaussians=32,
        covariance_type = 'full',
        initial_gaussian_values = None,
        regularization_strength = 0.,
        regularization_mode = 'L12',
        nfilters=64,
        activation = 'relu',
        skip_connections = True,
        scale='atom'):
    # We need a unique name for the layers for every call of this function.
    neighborhood_embedding.counter += 1
    # Compute local neighborhoods.
    local_coordinates, local_attributes = neighborhoods.LocalNeighborhood(coordinates=coordinates,
                                                                          name=f'neighborhood_{scale}_{neighborhood_embedding.counter}')([frames,neighbors,attributes])

    # Gaussian embedding of local coordinates.
    embedded_local_coordinates = embeddings.GaussianKernel(Ngaussians,
                                                           initial_values=initial_gaussian_values,
                                                           covariance_type=covariance_type,
                                                           name=f'embedded_local_coordinates_{scale}_{neighborhood_embedding.counter}')(local_coordinates)

    # Apply Spatio-chemical filters.
    if regularization_strength>0:
        kernel_regularizer = SparseRegularizer(coefficient=regularization_strength,mode=regularization_mode,ndims=3)
        single1_regularizer = SparseRegularizer(coefficient=regularization_strength,mode=regularization_mode,ndims=2)
        
        if regularization_mode in ['L1','L12','L12_group']:
            fixednorm = np.sqrt(Ngaussians / neighbors.shape[-1] )
        elif regularization_mode == 'Hoyer':
            fixednorm = None
        else:
            raise ValueError
    else:
        kernel_regularizer,single1_regularizer,fixed_norm = None, None, None
                

    spatiochemical_filters_input = embeddings.OuterProduct(nfilters,
                                                           use_single1=True, use_single2=False,
                                                            use_bias=False,
                                                            kernel_regularizer=kernel_regularizer,
                                                            single1_regularizer=single1_regularizer,
                                                            fixednorm=fixednorm,
                                                           sum_axis=2,
                                                            name=f'SCAN_filter_input_{scale}_{neighborhood_embedding.counter}')(
        [embedded_local_coordinates, local_attributes])

    if skip_connections:
        spatiochemical_filters_input=embeddings.MaskedConcatenate(axis=-1,name=f'SCAN_filter_concat_{scale}_{neighborhood_embedding.counter}')([spatiochemical_filters_input,attributes])
    spatiochemical_filters_activity = addNonLinearity(spatiochemical_filters_input, activation,
                                                          name=f'SCAN_filter_activity_{scale}_{neighborhood_embedding.counter}')  # Add non-linearity.
    return spatiochemical_filters_activity
    
neighborhood_embedding.counter = 0


def ScanNet(
        Lmax_aa=800,
        Lmax_atom=None,
        Lmax_aa_points=None,
        Lmax_atom_points=None,
        with_atom=True,
        nfeatures_atom=12,
        nembedding_atom=12,
        coordinates_atom='euclidean',        
        K_atom=16,
        N_atom=32,
        covariance_type_atom='full',        
        nfilters_atom=128,
        regularization_strength_atom = 2.5e-3,        
        dense_pooling=64,
        nattentionheads_pooling=64,        
        regularization_strength_pool = 2.5e-3,        
        nfeatures_aa=20,
        nembedding_aa=32,
        frame_aa='triplet_sidechain',        
        coordinates_aa='euclidean',        
        K_aa=16,
        N_aa=32,
        covariance_type_aa='full',        
        nfilters_aa=128,
        regularization_strength_aa = 2.5e-3,        
        filter_MLP=[32],
        coordinates_graph='graph',        
        index_distance_max_graph=8,        
        nattentionheads_graph=1,
        K_graph=32,
        N_graph=32,
        covariance_type_graph='full',        
        nfilters_graph=2,
        nclasses = 2,
        noutput_heads=1,
        name_heads = None,
        weight_heads = None,
        label_smoothing = 0.,
        initial_values={'GaussianKernel_atom': None, 'GaussianKernel_aa': None, 'GaussianKernel_graph': None,
                        'dense_graph': None},
        activation='relu',
        regularization_mode = 'Hoyer',
        dropout=0.,
        output = 'classification',        
        optimizer='adam',
        jit_compile='auto'):


    assert output in ['classification', 'distribution', 'multioutput_classification','autoregressive','scalar']
    if output in ['classification','distribution','scalar']:
        assert noutput_heads == 1
    elif output in ['multioutput_classification','autoregressive']:
        if name_heads is None:        
            name_heads = [f'output_{k}' for k in range(1,noutput_heads+1)]
        if weight_heads is None:
            weight_heads = np.array([1. for _ in range(noutput_heads)])            

    if frame_aa == 'triplet_backbone':
        order_aa = '3'
    elif frame_aa in ['triplet_sidechain', 'triplet_cbeta']:
        order_aa = '2'
    else:
        print('Incorrect frame_aa')
        return

    frame_atom = 'covalent'
    order_atom = '2'
    
    
    
    # Check initial values for GaussianKernel, and initialize to random uniform if needed.
    for component in ['aa', 'atom', 'graph']:
        if initial_values.get('GaussianKernel_%s' % component) is None:
            print('Initial values for %s GaussianKernel not found, random initialization...' % component)
            coordinates = locals()['coordinates_%s' % component]
            if coordinates == 'euclidean':
                xlims = [[-8.,8.],[-8.,8.],[-8.,8.]]
            elif coordinates == 'graph':
                xlims = [
                    [0,12.],
                    [-1.,1.],
                    [-1.,1.],
                    [-1.,1.],
                    [0,16]]
            else:
                assert ValueError,coordinates                                
            N = locals()[f'N_{component}']
            covariance_type = locals()['covariance_type_%s' % component]
            key = 'GaussianKernel_%s' % component
            print(N, covariance_type, key)
            initial_values[key] = embeddings.initialize_GaussianKernelRandom(xlims, N, covariance_type)

    if initial_values.get('dense_graph') is None:
        print('Initial values for graph dense not found, random initialization...')
        initial_values['dense_graph'] = [np.random.rand(N_graph, nattentionheads_graph)]
    else:
        if initial_values['dense_graph'][0].shape != (N_graph,nattentionheads_graph):
            print(f"Broadcasting dense_graph from {initial_values['dense_graph'][0].shape} to {(N_graph,nattentionheads_graph)}")
            initial_values['dense_graph'][0] = np.repeat( initial_values['dense_graph'][0] ,  nattentionheads_graph,axis=1)

    # Given maximum number of amino acids, define other maximum values.
    if Lmax_atom is None:
        Lmax_atom = 9 * Lmax_aa
    if Lmax_aa_points is None:
        if frame_aa == 'triplet_backbone':
            Lmax_aa_points = Lmax_aa + 2
        elif frame_aa == 'triplet_sidechain':
            Lmax_aa_points = 2 * Lmax_aa + 1
        elif frame_aa == 'quadruplet':
            Lmax_aa_points = 2 * Lmax_aa + 2
    if Lmax_atom_points is None:
        Lmax_atom_points = 11 * Lmax_aa

    ### Input layers.

    # The 3 inputs arrays at amino acid scale.
    frame_indices_aa = Input(shape=[Lmax_aa, 3], name='frame_indices_aa', dtype="int32") # List of triplet/quadruplet of the indices of the points for building the frames.
    attributes_aa = Input(shape=[Lmax_aa, nfeatures_aa], name='attributes_aa', dtype='float32') # The attributes of the amino acids (PWM/ one-hot encoded sequence).
    point_clouds_aa = Input(shape=[Lmax_aa_points, 3], name='point_clouds_aa', dtype='float32') # 3D coordinates of the amino acids point clouds. (Calpha coordinates, side chain center of mass,...).

    # Same, after masking.
    masked_frame_indices_aa = embeddings.MaskandClip(mask_value=-1, clip_min=-1, clip_max= Lmax_aa_points-1,  name='masked_frame_indices_aa')(frame_indices_aa)
    masked_attributes_aa = Masking(mask_value=0.0, name='masked_attributes_aa')(attributes_aa)
    masked_point_clouds_aa = Masking(mask_value=0.0, name='masked_point_clouds_aa')(point_clouds_aa)

    # Same at atomic scale.
    if with_atom:
        frame_indices_atom = Input(shape=[Lmax_atom, 3], name='frame_indices_atom', dtype="int32")
        attributes_atom = Input(shape=[Lmax_atom,1], name='attributes_atom', dtype="int32")
        point_clouds_atom = Input(shape=[Lmax_atom_points, 3], name='point_clouds_atom', dtype='float32')
        
        atom_to_aa_indices = Input(shape=[Lmax_atom, 1], name='atom_to_aa_indices', dtype="int32") # The mapping from atom to amino acid indices.
        aa_to_atom_indices = Input(shape=[Lmax_aa, 14], name='aa_to_atom_indices', dtype="int32") # The mapping from amino acid to atoms indices.

        masked_frame_indices_atom = embeddings.MaskandClip(mask_value=-1, clip_min=-1,clip_max=Lmax_atom_points-1,name='masked_frame_indices_atom')(frame_indices_atom)
        masked_point_clouds_atom = Masking(mask_value=0.0, name='masked_point_clouds_atom')(point_clouds_atom)
        masked_attributes_atom = Reshape((Lmax_atom,))(attributes_atom)
        masked_atom_to_aa_indices = embeddings.MaskandClip(mask_value=-1, clip_min=-1,clip_max=Lmax_aa-1, name='masked_atom_to_aa_indices')(atom_to_aa_indices)
        masked_aa_to_atom_indices = embeddings.MaskandClip(mask_value=-1, clip_min=-1,clip_max=Lmax_atom-1, name='masked_aa_to_atom_indices')(aa_to_atom_indices)        

    frames_aa = neighborhoods.FrameBuilder(order=order_aa,name=f'frames_aa')([masked_point_clouds_aa,masked_frame_indices_aa])
    
    if with_atom:
        frames_atom = neighborhoods.FrameBuilder(order=order_atom,name=f'frames_atom')([masked_point_clouds_atom,masked_frame_indices_atom])
        [neighbors_atom,neighbors_aa,neighbors_graph] = neighborhoods.NearestNeighborSearch(
        K_atoms=[K_atom],K_aas=[K_aa,K_graph],from_frames=True)([frames_atom, frames_aa, masked_atom_to_aa_indices, masked_aa_to_atom_indices])        
    else:
        [neighbors_aa,neighbors_graph] = neighborhoods.NearestNeighborSearch(K_atoms=[],K_aas=[K_aa,K_graph],from_frames=True)(frames_aa) # To implement

    ## Embed attributes.
    if nembedding_aa is not None: # Apply point-wise dense embedding.
        embedded_attributes_aa = attribute_embedding(masked_attributes_aa,nembedding_aa,activation,name='embedded_attributes_aa')
    else: # Normalize such that that variance is approximately one.
        normalizer = np.sqrt(nfeatures_aa)
        embedded_attributes_aa =  embeddings.MaskedRescaling(scale=normalizer,name='embedded_attributes_aa')(masked_attributes_aa)
        nembedding_aa = nfeatures_aa

    if with_atom:
        if nfeatures_atom == nembedding_atom:
            trainable = False  # Same number of categories as output dimension. Use one-hot encoding.
        else:
            trainable = True
        embedded_attributes_atom = Embedding(
            nfeatures_atom + 1, nembedding_atom, mask_zero=True, embeddings_initializer=utils.embeddings_initializer,
            embeddings_constraint=utils.FixedNorm(axis=0, value=np.sqrt(nfeatures_atom)), trainable=trainable,
            name='embedded_attributes_atom')(masked_attributes_atom)

        ## If atomic coordinates are included, compute embeddings of atomic neighborhoods.
        SCAN_filters_atom = neighborhood_embedding(
            frames_atom,
            embedded_attributes_atom,
            neighbors_atom,
            coordinates=coordinates_atom,
            Ngaussians=N_atom,
            covariance_type=covariance_type_atom,
            initial_gaussian_values=initial_values['GaussianKernel_atom'],
            regularization_mode=regularization_mode,
            regularization_strength=regularization_strength_atom,
            nfilters=nfilters_atom,
            activation=activation,
            scale='atom')

        # Pool at amino acid level by attention-pooling.
        if dense_pooling is None:
            dense_pooling = nfilters_atom
            
        if regularization_strength_pool>0:
            kernel_regularizer = SparseRegularizer(mode=regularization_mode,coefficient=regularization_strength_pool,ndims=2)
            if regularization_mode in ['L1','L12','L12_group']:
                kernel_norm = 1.0
            elif regularization_mode in ['Hoyer']:
                kernel_norm = None
        else:
            kernel_norm,kernel_regularizer = None,None
            
        
        SCAN_filters_atom_aggregated_input = attention.AttentionPooling(N_values=dense_pooling,
                                   N_heads=nattentionheads_pooling,
                                   kernel_regularizer=kernel_regularizer,
                                   kernel_norm=kernel_norm,
                                   name='atom_to_aa_pooling')([aa_to_atom_indices,SCAN_filters_atom])
        
        # Attention-based aggregation of atom features to amino acid scale.
        SCAN_filters_atom_aggregated_activity = addNonLinearity(SCAN_filters_atom_aggregated_input, activation,name='SCAN_filters_atom_aggregated_activity')
        # Add final non-linearity.


    if with_atom:
        all_embedded_attributes_aa = embeddings.MaskedConcatenate(name='all_embedded_attributes_aa', axis=-1)([embedded_attributes_aa, SCAN_filters_atom_aggregated_activity] )
    else:
        all_embedded_attributes_aa = Activation('linear',name='all_embedded_attributes_aa')(embedded_attributes_aa)



    # Compute embeddings of amino acid neighborhoods.
    SCAN_filters_aa = neighborhood_embedding(
        frames_aa,
        all_embedded_attributes_aa,
        neighbors_aa,
        coordinates=coordinates_aa,
        Ngaussians=N_aa,
        covariance_type=covariance_type_aa,
        initial_gaussian_values=initial_values['GaussianKernel_aa'],
        regularization_mode=regularization_mode,
        regularization_strength=regularization_strength_aa,
        nfilters=nfilters_aa,
        activation=activation,
        scale='aa')

    if dropout > 0:
        SCAN_filters_aa = Dropout(dropout, noise_shape=(
            None, 1, None), name='dropout')(SCAN_filters_aa)

    embedded_filter = SCAN_filters_aa
    if filter_MLP:
        for k, n_feature_filter in enumerate(filter_MLP):
            embedded_filter = attribute_embedding(embedded_filter,n_feature_filter,activation,name='SCAN_filters_aa_embedded_%s'%(k+1))


    # Final graph attention layer. Propagates label information from "hotspots" to passengers to obtain spatially consistent labels.

    beta = embeddings.MaskedDense(nattentionheads_graph, use_bias=True, bias_initializer=Ones(),
                                 kernel_initializer=Zeros(), activation='relu', name='beta')(embedded_filter)

    self_attention = embeddings.MaskedDense(nattentionheads_graph, use_bias=True, bias_initializer=Zeros(),
              kernel_initializer=Zeros(), name='self_attention')(embedded_filter)

    cross_attention = embeddings.MaskedDense(nattentionheads_graph, use_bias=False,
                                            kernel_initializer=Zeros(), name='cross_attention')(embedded_filter)

    if (nattentionheads_graph * nfilters_graph > noutput_heads*nclasses): ## Usually nattentionheads_graph = noutput_heads, and nfilters_graph = nclasses.
        node_features_activation = 'relu'
    else:
        node_features_activation = None

    node_features = embeddings.MaskedDense(
        nattentionheads_graph * nfilters_graph, activation=node_features_activation, use_bias=True,
        name='node_output')(embedded_filter)

    graph_weights, attention_local, node_features_local = neighborhoods.LocalNeighborhood(
        coordinates=coordinates_graph,index_distance_max=index_distance_max_graph, name='neighborhood_graph')(
        [frames_aa, neighbors_graph, cross_attention, node_features])

    embedded_graph_weights = embeddings.GaussianKernel(N_graph, initial_values=initial_values['GaussianKernel_graph']
                                                       , covariance_type=covariance_type_graph,
                                                       name='embedded_local_coordinates_graph')(graph_weights)

    embedded_graph_weights = embeddings.MaskedDense(
        nattentionheads_graph, use_bias=False, name='edges_graph')(embedded_graph_weights)
    if dropout > 0:
        embedded_graph_weights = Dropout(dropout, noise_shape=(None, None, None,1), name='edges_graph_dropout')(embedded_graph_weights)    

    graph_attention_output, attention_coefficients = attention.AttentionLayer(name='attention_layer')(
        [beta, self_attention, attention_local, node_features_local, embedded_graph_weights])
    
    if (nattentionheads_graph * nfilters_graph > noutput_heads * nclasses): ## Usually nattentionheads_graph = noutput_heads, and nfilters_graph = nclasses.
        preactivation_output = embeddings.MaskedDense(noutput_heads *nclasses, use_bias=True, activation=None, name='preactivation_output')(graph_attention_output)
    else:
        preactivation_output = graph_attention_output
        
        
    if output in ['classification','distribution']:
        classifier_output = embeddings.MaskedActivation(activation='softmax', name='classifier_output')(preactivation_output)
    elif output in ['multioutput_classification','autoregressive']:
        reshaped_preactivation_output = Reshape([Lmax_aa,noutput_heads,nclasses],name='reshaped_preactivation_output')(preactivation_output)
        post_softmax_preactivation_output = embeddings.MaskedActivation(activation='softmax',name='reshaped_classifier_output')(reshaped_preactivation_output)
        classifier_output = Reshape([Lmax_aa,noutput_heads*nclasses],name='classifier_output')(post_softmax_preactivation_output)
    elif output == 'scalar':
        classifier_output = embeddings.MaskedActivation(activation='none',name='classifier_output')(preactivation_output)    
    else:
        raise ValueError

    if with_atom:
        inputs = [frame_indices_aa, attributes_aa, aa_to_atom_indices, point_clouds_aa,
                  frame_indices_atom, attributes_atom, atom_to_aa_indices, point_clouds_atom]
    else:
        inputs = [frame_indices_aa, attributes_aa, point_clouds_aa]

    model = Model(inputs=inputs,outputs=classifier_output)
    
    if output in ['classification','distribution']:
        loss = loss_and_metrics.CategoricalCrossentropy_custom(label_smoothing=label_smoothing,
                                               name='categorical_crossentropy')
    elif output == 'multioutput_classification':
        loss = loss_and_metrics.MultiOutputCrossentropy_custom(
                                               noutputs = noutput_heads,
                                               weights = weight_heads,
                                               label_smoothing=label_smoothing,
                                               name='multioutput_crossentropy')
    elif output == 'autoregressive':
        loss = loss_and_metrics.AutoRegressiveCrossentropy_custom(
                                               noutputs = noutput_heads,
                                               weights = weight_heads,
                                               label_smoothing=label_smoothing,
                                               name='autoregressive_crossentropy')        
    elif output == 'scalar':
        loss = 'MSE'
    else:
        raise ValueError
    
    if output == 'classification':
        metrics = [loss_and_metrics.CategoricalCrossentropy_custom(label_smoothing=label_smoothing, name='categorical_crossentropy'),
                   loss_and_metrics.AUCPR_custom(name='aucpr')]        
    elif output == 'distribution':
        metrics = [loss_and_metrics.CategoricalCrossentropy_custom(label_smoothing=label_smoothing, name='categorical_crossentropy'),'MSE']        
    elif output == 'scalar':
        metrics = ['MSE']        
    else:
        raise ValueError            
    
    model.compile(loss=loss,optimizer=Adam(learning_rate=1e-3),metrics=metrics,jit_compile=jit_compile,run_eagerly=False)
    # model.get_layer('edges_graph').set_weights(initial_values['dense_graph'])
    model.summary()
    return model


def initialize_ScanNet(
        inputs,
        outputs,
        with_atom=True,
        Lmax_aa=800,
        nfeatures_atom=12,
        nembedding_atom=12,
        coordinates_atom='euclidean',        
        K_atom=16,
        Dmax_atom=4.,
        N_atom=32,
        covariance_type_atom='full',        
        nfilters_atom=128,
        regularization_strength_atom = 2.5e-3,        
        dense_pooling=64,
        nattentionheads_pooling=64,        
        regularization_strength_pool = 2.5e-3,        
        nfeatures_aa=20,
        nembedding_aa=32,
        frame_aa='triplet_sidechain',        
        coordinates_aa='euclidean',        
        K_aa=16,
        Dmax_aa=11.,        
        N_aa=32,
        covariance_type_aa='full',        
        nfilters_aa=128,
        regularization_strength_aa = 2.5e-3,        
        filter_MLP=[32],
        coordinates_graph='graph',        
        index_distance_max_graph=8,        
        K_graph=32,
        Dmax_graph=13.,        
        N_graph=32,
        covariance_type_graph='full',        
        nattentionheads_graph=1,
        nfilters_graph=2,        
        nclasses = 2,
        noutput_heads=1,
        name_heads = None,
        weight_heads = None,
        label_smoothing = 0.,        
        initial_values={'GaussianKernel_atom': None, 'GaussianKernel_aa': None, 'GaussianKernel_graph': None,
                        'dense_graph': None},
        activation='relu',
        regularization_mode = 'Hoyer',
        dropout=0.,
        optimizer='adam',
        output = 'classification',
        jit_compile='auto',
        initial_values_folder='/specific/netapp5_2/iscb/wolfson/jeromet/Data/InterfacePrediction/initial_values/',
        fresh_initial_values=False,
        save_initial_values=True,
        n_init=10,
        epochs = 100,
        batch_size=1):


    Lmax_atom = 9 * Lmax_aa
    if frame_aa == 'triplet_backbone':
        Lmax_aa_points = Lmax_aa + 2
        order_aa = '3'        
    elif frame_aa in ['triplet_sidechain','triplet_cbeta']:
        Lmax_aa_points = 2 * Lmax_aa + 1
        order_aa = '2'
    else:
        raise ValueError()

    Lmax_atom_points = 11 * Lmax_aa

    initial_values = {'GaussianKernel_aa': None, 'GaussianKernel_atom': None, 'GaussianKernel_graph': None,'dense_graph': None}

    if with_atom:
        input_type = ['triplets', 'attributes', 'indices', 'points', 'triplets', 'attributes', 'indices', 'points']
        Lmaxs = [Lmax_aa, Lmax_aa, Lmax_aa, Lmax_aa_points, Lmax_atom, Lmax_atom, Lmax_atom, Lmax_atom_points]
    else:
        input_type = ['triplets', 'attributes', 'points']
        Lmaxs = [Lmax_aa, Lmax_aa, Lmax_aa_points]


    order_atom = '2'
    frame_atom = 'covalent'

    location_aa = initial_values_folder + 'initial_GaussianKernel_aa_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_cov_%s.data' % (
        N_aa, K_aa, Dmax_aa, frame_aa, coordinates_aa, covariance_type_aa)
    
    location_atom = initial_values_folder + 'initial_GaussianKernel_aa_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_cov_%s.data' % (
        N_atom, K_atom, Dmax_atom, frame_atom, coordinates_atom, covariance_type_atom)
    
    location_graph = initial_values_folder + 'initial_GaussianKernel_graph_N_%s_%s_Kmax_%s_Dmax_%s_coords_%s_indexmax_%s_cov_%s.data' % (
        N_graph, 1, K_graph, Dmax_graph, coordinates_graph, index_distance_max_graph, covariance_type_graph)
    
    try:
        assert not fresh_initial_values
        initial_values = {}
        for location in location_aa,location_atom,location_graph:
            initial_values.update( io_utils.load_pickle(location) )
    except:
        print('At least one initial value is missing, regenerating the data')
        
        initial_values = neighborhoods.initialize_GaussianKernels_and_graph(inputs,
                                                        labels = outputs,
                                                        neighborhood_params =  {
                    'atom': {
                    'frame_order':order_atom,
                    'K':K_atom,
                    'coordinates':coordinates_atom,
                    'index_distance_max' : 16,
                    'Dmax':Dmax_atom,
                    },
                    'aa': {
                    'frame_order':order_aa,
                    'K':K_aa,
                    'coordinates':coordinates_aa,
                    'index_distance_max' : 16,
                    'Dmax':Dmax_aa,
                    },
                    'graph': {
                    'frame_order':order_aa,
                    'K':K_graph,
                    'coordinates':coordinates_graph,
                    'index_distance_max' : index_distance_max_graph,
                    'Dmax':Dmax_graph,
                    },
                },
            
            gaussian_params = {
                'atom': {
                    'covariance_type':covariance_type_atom,
                    'N': N_atom,
                },
                'aa': {
                    'covariance_type':covariance_type_aa,
                    'N': N_aa,
                },
                'graph': {
                    'covariance_type':covariance_type_graph,
                    'N': N_graph,
                },                                
            },
            
            n_samples=100,
            padded=False,
            n_init=n_init
            )
        if save_initial_values:
            io_utils.save_pickle({'GaussianKernel_atom':initial_values['GaussianKernel_atom']},location_atom)
            io_utils.save_pickle({'GaussianKernel_aa':initial_values['GaussianKernel_aa']},location_aa)
            io_utils.save_pickle({'GaussianKernel_graph':initial_values['GaussianKernel_graph'],
                                    'dense_graph':initial_values['dense_graph'],
                                    },
                                    location_graph)
                          
    model = wrappers.grouped_Predictor_wrapper(ScanNet,
                                               with_atom=with_atom,
                                               Lmax_aa=Lmax_aa,
                                               Lmax_atom=Lmax_atom,
                                               Lmax_aa_points=Lmax_aa_points,
                                               Lmax_atom_points=Lmax_atom_points,
                                               K_aa=K_aa,
                                               K_atom=K_atom,
                                               K_graph=K_graph,
                                               N_aa=N_aa,
                                               N_atom=N_atom,
                                               N_graph=N_graph,
                                               nfeatures_aa=nfeatures_aa,
                                               nfeatures_atom=nfeatures_atom,
                                               nembedding_atom=nembedding_atom,
                                               nembedding_aa=nembedding_aa,
                                               dense_pooling=dense_pooling,
                                               nattentionheads_pooling=nattentionheads_pooling,
                                               nfilters_aa=nfilters_aa,
                                               nfilters_atom=nfilters_atom,
                                               nfilters_graph=nfilters_graph,
                                               nattentionheads_graph=nattentionheads_graph,
                                               filter_MLP=filter_MLP,
                                               initial_values=initial_values,
                                               covariance_type_aa=covariance_type_aa,
                                               covariance_type_atom=covariance_type_atom,
                                               covariance_type_graph=covariance_type_graph,
                                               activation=activation,
                                               frame_aa=frame_aa,
                                               coordinates_aa=coordinates_aa,
                                               coordinates_atom=coordinates_atom,
                                               coordinates_graph=coordinates_graph,
                                               index_distance_max_graph=index_distance_max_graph,
                                               regularization_mode = regularization_mode,
                                               regularization_strength_aa = regularization_strength_aa,
                                               regularization_strength_atom = regularization_strength_atom,
                                               regularization_strength_pool = regularization_strength_pool,                                               
                                               dropout=dropout,
                                               optimizer=optimizer,
                                               input_type=input_type,
                                               Lmaxs=Lmaxs,
                                               output = output,
                                               multi_inputs=True,
                                               multi_outputs=False,
                                               nclasses=nclasses,
                                               noutput_heads=noutput_heads,
                                               name_heads = name_heads,
                                               weight_heads = weight_heads,
                                               label_smoothing = label_smoothing,
                                               jit_compile=jit_compile
                                               )
    extra_params = {'epochs': epochs, 'batch_size': batch_size}
    return model, extra_params






if __name__ == '__main__':
    from preprocessing import pipelines
    import numpy as np
    import utilities.dataset_utils as dataset_utils
    (list_origins,# List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
    list_sequences,# List of corresponding sequences.
    list_resids,#List of corresponding residue identifiers.
    list_labels)  = dataset_utils.read_labels('datasets/PPBS/labels_validation_70.txt')

    pipeline = pipelines.ScanNetPipeline(nclasses = 2)    
    
    inputs,outputs, failed_examples = pipeline.build_and_process_dataset(
                                  list_origins,
                                  list_resids=list_resids,
                                  list_labels=list_labels,
                                  biounit=True,
                                  verbose=True,
                                  overwrite=False,
                                  permissive=True,
                                  ncores = 10
                                  )

    model, extra_params = initialize_ScanNet(
        inputs,
        outputs,
        with_atom=True,
        Lmax_aa=1024,
        batch_size=1,
        fresh_initial_values=True,
        save_initial_values=False, 
        n_init=1,
    )
    model.fit(inputs, outputs, batch_size=1, epochs=10)
