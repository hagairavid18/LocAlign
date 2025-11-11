#%%
import numpy as np
from keras.layers import Layer,Input, Masking, Dense
from keras.models import Model, Sequential
from keras import ops
from utilities import wrappers
from preprocessing.pipelines import padd_matrix
from network.embeddings import GaussianKernel,initialize_GaussianKernel,MaskandClip



def distance_pcs(cloud1,cloud2, expand=True,squared=False):
    if expand:
        cloud1 = ops.expand_dims(cloud1,axis=-2)
        cloud2 = ops.expand_dims(cloud2,axis=-3)
    square_distance = (cloud1[...,0] - cloud2[...,0] )**2
    square_distance += (cloud1[...,1] - cloud2[...,1] )**2
    square_distance += (cloud1[...,2] - cloud2[...,2] )**2
    if not squared:
        return ops.sqrt(square_distance)
    else:
        return square_distance



class FrameBuilder(Layer):
    def __init__(self, order='2', **kwargs):
        super(FrameBuilder, self).__init__(**kwargs)
        self.support_masking = True
        self.epsilon = ops.array(1e-6)
        self.order = order
        return

    def build(self, input_shape):
        self.xaxis = ops.array(np.array([[1, 0, 0]], dtype=np.float32))
        self.yaxis = ops.array(np.array([[0, 1, 0]], dtype=np.float32))
        self.zaxis = ops.array(np.array([[0, 0, 1]], dtype=np.float32))

        super(FrameBuilder, self).build(input_shape)

    def call(self, inputs, mask=None):

        points, triplets = inputs
        '''
        For each atom, four cases should be distinguished.
        Case 1: both neighbors exist (i.e. at least two covalent bonds). Construct the frame as usual using Schmidt orthonormalization.
        Case 2: The first neighbor does not exist (i.e. the next atom along the protein tree. Example: Alanine, A_i =Cbeta A_{i-1} = Calpha, A_{i+1} = Cgamma does not exists). 
        The solution is to place a virtual atom such that (A_{i-2}, A_{i-1}, A_{i}, A_{i+1,virtual}) is a parallelogram. For alanine, (N,Calpha,Cbeta,Cgamma_virt) is a parallelogram.
        Case 3: The second neighbor does not exist (i.e. the previous atom along the protein tree.
        Example: N-terminal N along the backbone, or missing residues). Similarly, we build a parallelogram
        (A_{i-1,virtual}, A_i, A_{i+1},A_{i+2}).
        Case 4: None exist (missing atoms). Use the default cartesian frame.
        '''

        triplets = ops.clip(triplets, 0, ops.shape(points)[-2]-1)


        
        delta_10 = ops.take_along_axis(points, triplets[:,:,1:2],axis=1) - ops.take_along_axis(points, triplets[:,:,0:1],axis=1)
        delta_20 = ops.take_along_axis(points, triplets[:,:,2:3],axis=1) - ops.take_along_axis(points, triplets[:,:,0:1],axis=1)
        
        # delta_10 = tf.gather_nd(points, triplets[:, :, 1:2], batch_dims=1) - tf.gather_nd(points, triplets[:, :, 0:1],
        #                                                                                   batch_dims=1)        
        # delta_20 = tf.gather_nd(points, triplets[:, :, 2:3], batch_dims=1) - tf.gather_nd(points, triplets[:, :, 0:1],
        #                                                                                   batch_dims=1)
        
        if self.order in ['2','3']: # Order 1: the second point is on the zaxis and the third in the xz plane. Order 2: the third point is on the zaxis and the second in the xz plane.
            delta_10,delta_20 = delta_20,delta_10

        centers = ops.take_along_axis(points, triplets[:, :, 0:1], axis=1)
        # centers = tf.gather_nd(points, triplets[:, :, 0:1], batch_dims=1)
        zaxis = (delta_10 + self.epsilon * ops.reshape(self.zaxis,[1,1,3])) / (ops.sqrt(ops.sum(delta_10 ** 2, axis=-1, keepdims=True) ) + self.epsilon)

        yaxis = ops.cross(zaxis, delta_20)
        yaxis = (yaxis + self.epsilon * ops.reshape(self.yaxis,[1,1,3]) ) / (ops.sqrt(ops.sum(yaxis ** 2, axis=-1, keepdims=True) ) + self.epsilon)

        xaxis = ops.cross(yaxis, zaxis)
        xaxis = (xaxis + self.epsilon * ops.reshape(self.xaxis,[1,1,3]) ) / (ops.sqrt(ops.sum(xaxis ** 2, axis=-1, keepdims=True)) + self.epsilon)

        if self.order == '3':
            xaxis,yaxis,zaxis = zaxis,xaxis,yaxis


        frames = ops.stack([centers,xaxis,yaxis,zaxis],axis=-2)

        if mask not in [None,[None,None]]:
            frames *= ops.expand_dims(ops.expand_dims(ops.cast(mask[-1],'float32'),axis=-1),axis=-1)
        return frames

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], 4, 3)
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask in [None,[None,None]]:
            return None
        else:
            if mask[1] is not None:
                return ops.tile(ops.expand_dims(mask[1], axis=-1),[1,1,4])
            else:
                return None            
        
        # if isinstance(mask,list):
        #     condition = mask[1] is not None
        # else:
        #     condition = mask is not None  
        # if condition:
        #     return ops.tile(ops.expand_dims(mask[1], axis=-1),[1,1,4])
        # else:
        #     return None


class NearestNeighborSearch(Layer):
    '''
    Layer that takes as input:
    - cloud_atom: point cloud of atoms [batch, Natom , 3]
    - cloud_aa: point cloud of amino acids [batch, Naa , 3]
    - atom_to_aa mapping from atom to amino acid index [batch,Natom]
    - aa_to_atom mapping from amino acid to atom index [batch, Natom, N_atom_per_aa_max]
    
    - K_atoms, list of neighorhood sizes at atomic level.
    - K_aas, list of neighorhood sizes at amino acid level.
        
    And outputs a list of tensor of indices of nearest neighbors, for the atom and amino acid levels    
    [batch,Natom, K_atoms[0]], ..., [batch,Natom, K_atoms[-1]], [batch,Naa, K_aa[0]],...,[batch,Naa, K_aa[-1]]        
    '''
    
    def __init__(self,*args,K_atoms=[16],K_aas=[16,32],
                 from_frames=False,
                 **kwargs):
        self.K_atoms = K_atoms
        self.K_aas = K_aas
        self.from_frames = from_frames
        self.n_neighborhood_atoms = len(K_atoms)
        self.n_neighborhood_aas = len(K_aas)
        self.with_atoms = self.n_neighborhood_atoms>0
        super().__init__(*args,**kwargs)
        self.supports_masking = True        


    def call(self, inputs, mask=None):
        if self.with_atoms:
            if mask is not None:
                mask_atom,mask_aa = mask[0],mask[1]
            else:
                mask_atom,mask_aa = None,None
        else:
            mask_aa = mask
            
        if self.with_atoms:
            if self.from_frames:
                frames_atom, frames_aa, atom_to_aa, aa_to_atom = inputs
                cloud_atom,cloud_aa = frames_atom[:,:,0],frames_aa[:,:,0]            
                if mask_atom is not None:
                    mask_atom = mask_atom[:,:,0]
                if mask_aa is not None:
                    mask_aa = mask_aa[:,:,0]
            else:        
                cloud_atom, cloud_aa, atom_to_aa, aa_to_atom = inputs
        else:
            if self.from_frames:
                frames_aa = inputs
                cloud_aa = frames_aa[:,:,0]            
                if mask_aa is not None:
                    mask_aa = mask_aa[:,:,0]
            else:        
                cloud_aa = inputs            
        
        if self.with_atoms:
            batch_size,N_atom,N_aa, N_atom_per_aa_max = cloud_atom.shape[0], cloud_atom.shape[1], cloud_aa.shape[1], aa_to_atom.shape[-1]
        else:
            batch_size,N_aa = cloud_aa.shape[0], cloud_aa.shape[1]
            
        if batch_size is None:
            batch_size = -1
        factor = 2 # For top-16 closest atoms, search among top-32 closest amino acids. Handles edge cases, like two amino acids whose Calpha are far but side chains are close. Set to 1 for small speed-up.
        K_max = max([factor*K for K in self.K_atoms] + self.K_aas)
        
        # Pairwise amino acid distances
        distance_aa = distance_pcs(cloud_aa,cloud_aa,expand=True,squared=True)
        if mask_aa is not None:
            distance_aa = ops.where(ops.expand_dims(mask_aa,axis=-2) & ops.expand_dims(mask_aa,axis=-1),
                                    distance_aa,np.inf)            
        distance_aa *= -1 # Negative distance for top_k.            
        nearest_neighbors_aa_max = ops.top_k(distance_aa,k=K_max)[1]                
        nearest_neighbors_aa = [nearest_neighbors_aa_max[:,:,:K] for K in self.K_aas]
        
        if self.with_atoms:
            K_atom_max = factor*max(self.K_atoms) # For top-16 closest atoms, search among top-32 closest amino acids.
            neighbors_aa_for_atom = nearest_neighbors_aa_max[:,:,:K_atom_max]
            
            # This casting shouldn't be needed, but apparently necessary for the jax backend...
            aa_to_atom, atom_to_aa = ops.cast(aa_to_atom,'int32'), ops.cast(atom_to_aa,'int32')
            
            
            candidate_atom_neighbors_index = ops.reshape(
            ops.take_along_axis(
            aa_to_atom,
            ops.reshape(
                ops.take_along_axis(
                    neighbors_aa_for_atom, 
                    ops.clip(atom_to_aa , -1, N_aa-1) # This clip is needed for the edge case where we truncate a long protein to N_aa, but some atoms associated to residues N_aa,N_aa+1... appear in the atomic point cloud.
                    ,axis=1),
                        [batch_size,N_atom*K_atom_max,1]),
            axis=1),
            [batch_size,N_atom,K_atom_max*N_atom_per_aa_max]
            )
            
            mask_candidate_atom = ops.not_equal(candidate_atom_neighbors_index,-1)
            if mask_atom is not None:
                mask_candidate_atom &= ops.reshape(
                    ops.take_along_axis(
                    mask_atom, 
                    ops.reshape(candidate_atom_neighbors_index,[batch_size,N_atom*K_atom_max*N_atom_per_aa_max]),
                    axis=1
                ), [batch_size,N_atom,K_atom_max*N_atom_per_aa_max])
                
                
            cloud_candidate_atom = ops.reshape(
                ops.take_along_axis(
                cloud_atom,
                ops.reshape(
                    candidate_atom_neighbors_index,
                    [batch_size,N_atom*K_atom_max*N_atom_per_aa_max,1]
                ),
                axis=1
            ),
                [batch_size,N_atom,K_atom_max*N_atom_per_aa_max,3]
            )
                        
            distance_atom_to_candidates = distance_pcs(ops.expand_dims(cloud_atom,axis=-2),cloud_candidate_atom,expand=False,squared=True)
            distance_atom_to_candidates = ops.where(mask_candidate_atom,distance_atom_to_candidates,np.inf)
            
            distance_atom_to_candidates *= -1 # Negative distance for top_k.            
            nearest_neighbors_atom_among_candidates = ops.top_k(distance_atom_to_candidates,k=K_atom_max)[1]
            
            nearest_neighbors_atom_max = ops.take_along_axis(
                candidate_atom_neighbors_index,            
                nearest_neighbors_atom_among_candidates,axis=-1)
            
            nearest_neighbors_atom = [nearest_neighbors_atom_max[:,:,:K] for K in self.K_atoms]
            return nearest_neighbors_atom + nearest_neighbors_aa
        else:
            return nearest_neighbors_aa
    
    def compute_mask(self, inputs, mask=None):                
        if mask is None:
            return None
        elif self.with_atoms & (mask == [None for _ in inputs]):
            return [None for _ in range(self.n_neighborhood_aas + self.n_neighborhood_atoms)]
        else:
            if self.with_atoms:
                if self.from_frames:
                    return [mask[0][:,:,1] for _ in range(self.n_neighborhood_atoms)] + [mask[1][:,:,1] for _ in range(self.n_neighborhood_aas)]
                else:
                    return [mask[0] for _ in range(self.n_neighborhood_atoms)] + [mask[1] for _ in range(self.n_neighborhood_aa)]
            else:
                if self.from_frames:
                    return [mask[:,:,1] for _ in range(self.n_neighborhood_aas)]
                else:
                    return [mask for _ in range(self.n_neighborhood_aa)]
                
    
    def get_config(self):
        config = {'K_atoms': self.K_atoms,
                  'K_aas': self.K_aas,
                  'from_frames':self.from_frames}
        config.update( super(NearestNeighborSearch, self).get_config() )
        return config



class LocalNeighborhood(Layer):
    def __init__(self, coordinates='euclidean',index_distance_max=16,**kwargs):
        
        assert coordinates in ['euclidean','graph']
        self.coordinates = coordinates
        self.coordinates_dimension = 5 if coordinates == 'graph' else 3
        self.index_distance_max = index_distance_max
        self.epsilon = ops.array(1e-6)
        self.support_masking = True     
        return super(LocalNeighborhood, self).__init__(**kwargs)


    def build(self, input_shape):
        self.nattributes = len(input_shape) - 2
        super(LocalNeighborhood, self).build(input_shape)


    def call(self, inputs, mask=None):
        frames, neighbors = inputs[0],inputs[1]
        attributes = inputs[2:]
        
        batch_size = -1 # Batch size
        N = frames.shape[1] # Number of frames
        K = neighbors.shape[2] # Number of neighbors
        
        
        if self.coordinates == 'euclidean':
            neighbor_coordinates_global = ops.reshape(
                                    ops.take_along_axis(frames[:,:,0], 
                ops.reshape(neighbors, [batch_size, N*K,1])
                ,axis=1), [batch_size,N,K,3]) # B X Lmax X K X 3
            
            neighbors_coordinates_local = ops.einsum('...ki,...ji->...kj',
                                                    neighbor_coordinates_global - frames[:,:,:1], # After removing the center
                                                    frames[:,:,1:4]) # Relative coordinates in the frame.
            
        
        elif self.coordinates == 'graph':
                        
            neighbor_frames = ops.reshape(
                                    ops.take_along_axis(frames, 
                ops.reshape(neighbors, [batch_size, N*K,1,1])
                ,axis = 1), [batch_size,N,K,4,3]
            )
            
            neighbor_vector = neighbor_frames[:,:,:,0] - ops.expand_dims(frames[:,:,0] , axis=2)
            neighbor_distance = ops.sqrt( ops.sum(neighbor_vector**2,axis=-1) )            
            neighbor_unit_vector = neighbor_vector / (ops.expand_dims(neighbor_distance,axis=-1) + self.epsilon)
            neighbor_ZdotZ = ops.einsum('...i,...ki->...k',frames[:,:,-1,:],neighbor_frames[:,:,:,-1,:])
            
            neighbor_ZdotDelta = ops.einsum('...i,...ki->...k',frames[:,:,-1,:],
                                              neighbor_unit_vector)
            
            neighbor_DeltadotZ = ops.sum(neighbor_unit_vector * neighbor_frames[:,:,:,-1,:],axis=-1 )
            
            neighbor_index_distance = ops.clip(
                ops.abs(ops.cast( neighbors - neighbors[:,:,:1], 'float32') ),
                0, self.index_distance_max) # How many amino acid apart along the sequence.
                                                
            neighbors_coordinates_local = ops.stack([neighbor_distance,neighbor_ZdotZ,neighbor_ZdotDelta,neighbor_DeltadotZ,neighbor_index_distance],axis=-1)
                            
        
        neighbors_attributes = [ 
                        ops.reshape(
                            ops.take_along_axis(attribute, 
        ops.reshape(neighbors, [batch_size, K*N,1])
        ,axis=1), [batch_size,N,K,attribute.shape[-1]])
        for attribute in attributes]
        
        if mask is None or mask is [None for _ in inputs]:
            pass
        else:
            mask_points = mask[0][:,:,1]
            neighbors_mask =  ops.expand_dims(ops.cast(ops.expand_dims(mask_points,axis=-1) & ops.reshape(
                            ops.take_along_axis(mask_points, 
                                                ops.reshape(neighbors, [batch_size, K*N]),axis=1),
                            [batch_size,N,K]),
                                      float),axis=-1)
            neighbors_coordinates_local *= neighbors_mask
            for k in range(self.nattributes):
                neighbors_attributes[k] *= neighbors_mask
                                        
        return [neighbors_coordinates_local] + neighbors_attributes


    def compute_output_shape(self, input_shape):
        batch_size =input_shape[0][0] # Batch size
        N = input_shape[0][1] # Number of frames
        K = input_shape[1][2] # Number of neighbors        
        attribute_dims = [attribute_shape[-1] for attribute_shape in input_shape[2:]]
        
        output_shape = [(batch_size, N, K, dim)                        
                        for dim in [self.coordinates_dimension] + attribute_dims
                        ]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if isinstance(mask,list):
            condition = mask[0] is not None
        else:
            condition = mask is not None
        if condition:
            return [mask[0][:,:,1:2] for _ in range(1+self.nattributes)]        
        else:
            return mask

    def get_config(self):
        config = {
                  'coordinates': self.coordinates,
                  'index_distance_max': self.index_distance_max,
                  }
        base_config = super(LocalNeighborhood, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_Frames(inputs,n_samples=None,padded=False,order='2',Lmax=None):
    B = len(inputs[0])
    if n_samples is not None:
        b = min(n_samples, B)
    else:
        b = B

    nindices = inputs[0][0].shape[-1]

    if padded:
        triplets_ = inputs[0][:b]
        clouds_ = inputs[1][:b]
        Lmax = inputs[0].shape[1]
        Lmax2 = inputs[1].shape[1]
    else:
        if Lmax is not None:
            Lmax = min(max([len(input_) for input_ in inputs[0][:b]] ), Lmax)
        else:
            Lmax = max([len(input_) for input_ in inputs[0][:b]] )
        Lmax2 = max([len(input_) for input_ in inputs[1][:b]] )
        Ls = [len(x) for x in inputs[0][:b]]
        triplets_ = np.zeros([b,Lmax, nindices ],dtype=np.int32)
        clouds_ = np.zeros([b,Lmax2,3],dtype=np.float32)

        for b_ in range(b):
            padd_matrix(inputs[0][b_], padded_matrix=triplets_[b_], padding_value=-1)
            padd_matrix(inputs[1][b_], padded_matrix=clouds_[b_], padding_value=0)

    inputs_ = [triplets_,clouds_]

    triplets = Input(shape=[Lmax, nindices], dtype="int32",name='triplets')
    clouds = Input(shape=[Lmax2, 3], dtype='float32',name='clouds')
    masked_triplets = Masking(mask_value=-1, name='masked_triplets')(triplets)
    masked_clouds = Masking(mask_value=0.0, name='masked_clouds')(clouds)
    frames = FrameBuilder(name='frames',order=order)([masked_clouds,masked_triplets])
    first_layer = Model(
        inputs=[triplets,clouds], outputs=frames)
    frames_ = first_layer.predict(inputs_)
    if not padded:
        frames_ = wrappers.truncate_list_of_arrays(frames_,Ls)
    return frames_


def get_LocalNeighborhoods(inputs, 
                           flat=False,
                           n_samples=100,
                           padded=False,                           
                           neighborhood_params =  {
                                    'atom': {
                                    'frame_order':'2',
                                    'K':16,
                                    'coordinates':'euclidean',
                                    'index_distance_max' : 16
                                    },
                                    'aa': {
                                    'frame_order':'2',
                                    'K':16,
                                    'coordinates':'euclidean',
                                    'index_distance_max' : 16
                                    },
                                    'graph': {
                                    'frame_order':'2',
                                    'K':32,
                                    'coordinates':'graph',
                                    'index_distance_max' : 16
                                    },
                                }
                           ):
        
    inputs = [ inputs_[:n_samples] for inputs_ in inputs]
    inputs[1] = [input.astype(np.float32) for input in inputs[1]]
    inputs[5] = [input.astype(np.float32) for input in inputs[5]]
            
    if padded:
        Lmaxs = [inputs_.shape[1] for inputs_ in inputs]
    else:
        Lmaxs = [max([len(input_) for input_ in inputs_] ) for inputs_ in  inputs]        
        padding_values = [-1,0.,-1,0.,-1,0.,-1,0]        
        inputs = [
            np.stack([padd_matrix(input,Lmax=Lmax,padding_value=padding_value) for input in inputs_]
,axis =0)
        for Lmax,padding_value,inputs_, in zip(Lmaxs,padding_values,inputs)]        
            
    dtypes = ["int32","float32","int32","float32","int32","float32","int32","float32"]
    keras_inputs  = [Input(shape=inputs_.shape[1:],dtype=dtype_) for inputs_,dtype_ in zip(inputs,dtypes)]
    
    
    masked_keras_inputs = [
        MaskandClip(mask_value=-1, clip_min=-1, clip_max= Lmaxs[3]-1)(keras_inputs[0]),
        Masking(mask_value=0.)(keras_inputs[1]),
        MaskandClip(mask_value=-1, clip_min=-1, clip_max= Lmaxs[4]-1)(keras_inputs[2]),
        Masking(mask_value=0.)(keras_inputs[3]),
        MaskandClip(mask_value=-1, clip_min=-1, clip_max= Lmaxs[7]-1)(keras_inputs[4]),
        Masking(mask_value=0.)(keras_inputs[5]),
        MaskandClip(mask_value=-1, clip_min=-1, clip_max= Lmaxs[0]-1)(keras_inputs[6]),
        Masking(mask_value=0.)(keras_inputs[7]),
    ]
        
    
    [
            frame_indices_aa,
            attributes_aa,
            aa_to_atom,
            point_clouds_aa,
            frame_indices_atom,
            attributes_atom,
            atom_to_aa,
            point_clouds_atom        
        ] = masked_keras_inputs
    
    
    frames_aa = FrameBuilder(order=neighborhood_params['aa']['frame_order'])([point_clouds_aa,frame_indices_aa])
    frames_atom = FrameBuilder(order=neighborhood_params['atom']['frame_order'])([point_clouds_atom,frame_indices_atom])
    
    [neighbors_atom,neighbors_aa,neighbors_graph] = NearestNeighborSearch(K_atoms=[neighborhood_params['atom']['K']],K_aas=[neighborhood_params['aa']['K'],neighborhood_params['graph']['K']],from_frames=True)([frames_atom, frames_aa, atom_to_aa, aa_to_atom])
    
    neighbor_coordinates_atom, neighbors_attributes_atom = LocalNeighborhood(coordinates=neighborhood_params['atom']['coordinates'],index_distance_max=neighborhood_params['atom']['index_distance_max'])([frames_atom,neighbors_atom,attributes_atom])
    neighbor_coordinates_aa, neighbors_attributes_aa = LocalNeighborhood(coordinates=neighborhood_params['aa']['coordinates'],index_distance_max=neighborhood_params['aa']['index_distance_max'])([frames_aa,neighbors_aa,attributes_aa])
    neighbor_coordinates_graph, neighbors_attributes_graph = LocalNeighborhood(coordinates=neighborhood_params['graph']['coordinates'],index_distance_max=neighborhood_params['graph']['index_distance_max'])([frames_aa,neighbors_graph,attributes_aa])
    
    model = Model(
        inputs = keras_inputs,
        outputs = [neighbor_coordinates_atom,neighbors_attributes_atom,
                   neighbor_coordinates_aa,neighbors_attributes_aa,
                   neighbor_coordinates_graph,neighbors_attributes_graph]
    )
    
    # print(model.summary() )
    # raise ValueError
    

    outputs = model.predict(inputs, batch_size=1)
    
    if flat:
        masks = [outputs[k].max((-1,-2))>0 for k in [0,2,4]]
        outputs = [outputs[j][masks[j//2]] for j in range(6)]
        
    [neighbor_coordinates_atom,neighbors_attributes_atom,
     neighbor_coordinates_aa,neighbors_attributes_aa,
     neighbor_coordinates_graph,neighbors_attributes_graph] = outputs
    
    return [neighbor_coordinates_atom,neighbors_attributes_atom,
            neighbor_coordinates_aa,neighbors_attributes_aa,
            neighbor_coordinates_graph,neighbors_attributes_graph]
    

def initialize_GaussianKernels_and_graph(
    inputs,
    labels = None,            
    neighborhood_params =  {
            'atom': {
            'frame_order':'2',
            'K':16,
            'coordinates':'euclidean',
            'index_distance_max' : 16,
            'Dmax':4,
            },
            'aa': {
            'frame_order':'2',
            'K':16,
            'coordinates':'euclidean',
            'index_distance_max' : 16,
            'Dmax':13,
            },
            'graph': {
            'frame_order':'2',
            'K':32,
            'coordinates':'graph',
            'index_distance_max' : 16,
            'Dmax':13,            
            },
        },
    
    gaussian_params = {
        'atom': {
            'covariance_type':'full',
            'N': 32,
        },
        'aa': {
            'covariance_type':'full',
            'N': 32,
        },
        'graph': {
            'covariance_type':'full',
            'N': 32,
        },                                
    },
    
    n_samples=100,
    padded=False,
    n_init=10    
):
    initial_values = {}
    
    if labels is not None:        
        inputs = [inputs[0],labels] + inputs[2:]
                    
    outputs = get_LocalNeighborhoods(
                                    inputs, 
                                    flat=True,
                                    n_samples=n_samples,
                                    padded=padded,                           
                                    neighborhood_params =  neighborhood_params)
            
    
    for k,scale in enumerate(['atom','aa','graph']):        
        neighbor_coordinates_scale, neighbor_attributes_scale = outputs[2*k:2*k+2]
        if neighborhood_params[scale]['coordinates'] == 'euclidean':
            neighbor_distance_scale = np.sqrt((neighbor_coordinates_scale**2).sum(-1))
        else:
            neighbor_distance_scale = neighbor_coordinates_scale[...,0]
        neighbor_coordinates_scale = neighbor_coordinates_scale[neighbor_distance_scale <= neighborhood_params[scale]['Dmax']]
        neighbor_attributes_scale = neighbor_attributes_scale[neighbor_distance_scale <= neighborhood_params[scale]['Dmax']]
        
        if neighborhood_params[scale]['coordinates'] == 'euclidean':
            reg_covar = 1e-2
        else:
            reg_covar = 1e0
            
        initial_values[f'GaussianKernel_{scale}'] = initialize_GaussianKernel(neighbor_coordinates_scale, gaussian_params[scale]['N'],
                                covariance_type=gaussian_params[scale]['covariance_type'],
                                reg_covar=reg_covar,n_init=n_init)
        
        
        
    if labels is not None:
        neighbor_coordinates_graph, neighbor_attributes_graph = outputs[4],outputs[5]
        initial_values.update(
                        initialize_GraphEdges(
                                        neighbor_coordinates_graph,
                                        neighbor_attributes_graph,
                                        gaussian_params['graph'],                                        
                                        initial_values[f'GaussianKernel_graph'],
                                        )
        )
    return initial_values
        
        

def initialize_GraphEdges(
                    neighbor_coordinates,
                    neighbor_labels,
                    gaussian_params,
                    gaussian_values,
                    epochs=10
                    ):
    
    
    mu_labels = neighbor_labels[:,0,1].mean()
    features = neighbor_coordinates.reshape([-1, neighbor_coordinates.shape[-1]])
    
    target =  ( (neighbor_labels[:,:,1] - neighbor_labels[:,:,1].mean(0,keepdims=True)) * (neighbor_labels[:,:1,1] - mu_labels) ).flatten() / (mu_labels-mu_labels**2)
    target[features[:,0]>=15.] = 0.
    print('check_decay:0', target[features[:,0]==0].mean() )
    print('check_decay:5', target[features[:,0] <= 5].mean() )
    print('check_decay:10', target[features[:,0]>=10].mean() )
    print('check_decay:15', target[features[:,0]>=15].mean() )
    model = Sequential()
    model.add(Input(shape=(5,)))
    model.add(GaussianKernel(gaussian_params['N'], 
                             covariance_type=gaussian_params['covariance_type'],
                      initial_values=gaussian_values, name='GaussianKernel_graph'))
    model.add(Dense(1, activation=None, use_bias=False, name='dense_graph'))
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    model.fit(features, target, epochs=epochs, batch_size=1024)
    
    # plt.scatter( features[:,0], model.predict(features), s=1,alpha=0.25,c=features[:,-1] ); plt.show()
    
    model_params = dict([(layer.name, layer.get_weights())
                         for layer in model.layers])
    return model_params




if __name__ == '__main__':
# %%
    import matplotlib.pyplot as plt
    from preprocessing import pipelines
    import numpy as np
    
    list_origins = ['11as_A',
                '137l_B',
                '13gs_A',
                '1a05_A',
                '1a09_A',
                '1a0d_A',
                '1a0e_A',
                '1a0f_A',
                '1a0g_B',
                '1a0o_B']

    pipeline = pipelines.ScanNetPipeline(
                 nclasses = 2)    
    
    inputs,outputs, failed_examples = pipeline.build_and_process_dataset(
                                  list_origins,
                                  biounit=True,
                                  verbose=True,
                                  overwrite=False,
                                  permissive=True,
                                  ncores = 10
                                  )
    
    frames = get_Frames([inputs[0],inputs[3]] ,padded=False)
    
    
    plt.plot(frames[0][:,0,:]); plt.show() # Check centers.
    plt.plot(frames[0][:, 1, :]); plt.show()  # Check unit vectors.
    for i in range(3): # Check orthonormality.
        for j in range(3):
            print('Dot product',i,j , np.abs( (frames[0][:, 1+i, :] * frames[0][:,1+j,:]).sum(-1)  ).max() )


    [neighbor_coordinates_atom,neighbors_attributes_atom,
            neighbor_coordinates_aa,neighbors_attributes_aa,
            neighbor_coordinates_graph,neighbors_attributes_graph] = get_LocalNeighborhoods(inputs,
                           flat=False,
                           padded=False,
                           neighborhood_params =  {
                                    'atom': {
                                    'frame_order':'2',
                                    'K':16,
                                    'coordinates':'euclidean',
                                    'index_distance_max' : 16
                                    },
                                    'aa': {
                                    'frame_order':'2',
                                    'K':16,
                                    'coordinates':'euclidean',
                                    'index_distance_max' : 16
                                    },
                                    'graph': {
                                    'frame_order':'2',
                                    'K':32,
                                    'coordinates':'graph',
                                    'index_distance_max' : 16
                                    },
                                }
                           )
            
            

    plt.hist( neighbor_coordinates_atom.flatten(),bins=100 ); plt.show() # Check scales.
    plt.matshow(np.sqrt( (neighbor_coordinates_atom[0]**2).sum(-1) ) ,aspect='auto'); plt.colorbar(); plt.show() # Check order and padding.

    plt.hist( neighbor_coordinates_aa.flatten(),bins=100 ); plt.show() # Check scales.
    plt.matshow(np.sqrt( (neighbor_coordinates_aa[0]**2).sum(-1) ) ,aspect='auto'); plt.colorbar(); plt.show() # Check order and padding.
    

    plt.matshow(neighbor_coordinates_graph[0,:,:,0],aspect='auto'); plt.colorbar(); plt.show()
    for i in range(neighbor_coordinates_graph.shape[-1]):
        plt.hist(neighbor_coordinates_graph[...,i].flatten(),bins=100); plt.show()

    plt.matshow(neighbor_coordinates_graph[0,:,:,-1],aspect='auto'); plt.colorbar(); plt.show()

#%%
    import utilities.dataset_utils as dataset_utils
    (list_origins,# List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
    list_sequences,# List of corresponding sequences.
    list_resids,#List of corresponding residue identifiers.
    list_labels)  = dataset_utils.read_labels('datasets/PPBS/labels_validation_70.txt')

    pipeline = pipelines.ScanNetPipeline(
                 nclasses = 2)    
    
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

    initial_values = initialize_GaussianKernels_and_graph(inputs,
                                                          labels = outputs,
                                                          neighborhood_params =  {
            'atom': {
            'frame_order':'2',
            'K':16,
            'coordinates':'euclidean',
            'index_distance_max' : 16,
            'Dmax':4,
            },
            'aa': {
            'frame_order':'2',
            'K':16,
            'coordinates':'euclidean',
            'index_distance_max' : 16,
            'Dmax':13,
            },
            'graph': {
            'frame_order':'2',
            'K':32,
            'coordinates':'graph',
            'index_distance_max' : 16,
            'Dmax':13,            
            },
        },
    
    gaussian_params = {
        'atom': {
            'covariance_type':'full',
            'N': 32,
        },
        'aa': {
            'covariance_type':'full',
            'N': 32,
        },
        'graph': {
            'covariance_type':'full',
            'N': 32,
        },                                
    },
    
    n_samples=10,
    padded=False,
    n_init=1
    )

    plt.plot(initial_values['GaussianKernel_atom'][0].T); plt.show()
    plt.hist(initial_values['GaussianKernel_atom'][1].flatten(), bins=20); plt.show()
    
    plt.plot(initial_values['GaussianKernel_aa'][0].T); plt.show()
    plt.hist(initial_values['GaussianKernel_aa'][1].flatten(), bins=20); plt.show()

    plt.plot(initial_values['GaussianKernel_graph'][0].T); plt.show()
    plt.hist(initial_values['GaussianKernel_graph'][1].flatten(), bins=20); plt.show()    


    #%% Check consistency with wrapper.
    import wrappers


    all_triplets = []
    all_clouds = []

    B = 100
    for b in range(B):
        L = np.random.randint(5,high=21)
        random_direction1 = np.random.randn(3)
        random_direction2 = np.random.randn(3)
        points = np.random.randn(L,3)
        cloud = np.concatenate([
            points,
            points + random_direction1[np.newaxis],
            points + random_direction2[np.newaxis]
        ], axis=0)
        triplet = np.stack([
            np.arange(L),
            np.arange(L)+L,
            np.arange(L)+2*L], axis=-1)
        all_clouds.append(cloud)
        all_triplets.append(triplet)


    def keras_frames(Lmax=20):
        from keras.engine.base_layer import Layer
        from keras import backend as K
        import numpy as np
        from keras.layers import Input, Masking, Dense
        from keras.models import Model
        cloud = Input(shape=(3*Lmax,3),dtype='float32')
        triplet = Input(shape=(Lmax, 3), dtype="int32")
        masked_cloud = Masking(mask_value=0.0, name='masked_indices_atom')(cloud)
        masked_triplets = Masking(mask_value=-1, name='masked_triplets_atom')(triplet)
        frames = FrameBuilder()([cloud,triplet])
        model = Model(inputs=[triplet,cloud],outputs=frames)
        return model


    model = wrappers.grouped_Predictor_wrapper(keras_frames,
                                               Lmax=20,
                                               multi_inputs=True,
                                               input_type=['triplets','points'],
                                               Lmaxs=[20,3*20])


    all_frames = model.predict([all_triplets,all_clouds],return_all=True,batch_size=1)

    for i,frame in enumerate(all_frames):
        print(i,(frame[:,1:].max(0)-frame[:,1:].min(0) ).max()  )
