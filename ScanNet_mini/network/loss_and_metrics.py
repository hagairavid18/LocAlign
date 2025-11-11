from keras.metrics import Metric,AUC
from keras.losses import Loss,CategoricalCrossentropy
from keras import ops

class CategoricalCrossentropy_custom(Loss):
    def __init__(self, label_smoothing=0., name = 'categorical_crossentropy',  **kwargs):
        self.label_smoothing = label_smoothing
        self._cce = CategoricalCrossentropy(from_logits=False, reduction='mean')
        super().__init__(name=name,**kwargs)

    def call(self, y_true, y_pred):
        if self.label_smoothing>0.:
            nclasses = y_true.shape[-1]
            y_true = self.label_smoothing / nclasses * ops.sum(y_true,axis=-1,keepdims=True) + (1-self.label_smoothing) * y_true
            ## In ScanNet, for a given token, y_true does not sum to 1 but to the token weight. Need to correct for this.
        return self._cce(y_true,y_pred)


    def get_config(self):
        config = super().get_config()
        config['label_smoothing'] = self.label_smoothing
        return config

class MultiOutputCrossentropy_custom(Loss):
    def __init__(self,label_smoothing=0.,
        noutputs=1, weights=None,
        name='multioutput_crossentropy',**kwargs):
        self.label_smoothing = label_smoothing
        self._cce = CategoricalCrossentropy(from_logits=False, reduction='mean')

        assert isinstance(noutputs,int)
        self.noutputs = noutputs
        self.weights = ops.array(weights) if weights is not None else None
        super().__init__(name=name,**kwargs)


    def call(self,y_true, y_pred):
        nclasses = y_true.shape[-1] // self.noutputs
        y_true = ops.reshape(y_true, list(y_true.shape[:-1]) + [self.noutputs, nclasses] )

        if self.label_smoothing>0.:
            y_true = self.label_smoothing / y_true.shape[-1] * ops.sum(y_true,axis=-1,keepdims=True) + (1-self.label_smoothing) * y_true
            ## In ScanNet, for a given token, y_true does not sum to 1 but to the token weight. Need to correct for this.

        if self.weights is not None:
            y_true = y_true * ops.reshape(self.weights, [1 for _ in y_true.shape[:-2]] + [self.noutputs,1] )

        y_true = ops.reshape(y_true, list(y_true.shape[:-2]) + [self.noutputs*nclasses] )
        return self._cce(y_true,y_pred)


class AutoRegressiveCrossentropy_custom(Loss):
    def __init__(self,label_smoothing=0.,
        noutputs=1, weights=None,
        name='autoregressive_crossentropy',**kwargs):
        self.label_smoothing = label_smoothing

        assert isinstance(noutputs,int)
        self.noutputs = noutputs
        self.weights = ops.array(weights) if weights is not None else None
        self._cce = CategoricalCrossentropy(from_logits=False, reduction=None,axis=-1)

        super().__init__(name=name,**kwargs)

    def call(self,y_true, y_pred):
        nclasses = y_true.shape[-1] // self.noutputs
        y_true = ops.reshape(y_true, list(y_true.shape[:-1]) + [self.noutputs, nclasses] )
        y_pred = ops.reshape(y_pred, list(y_pred.shape[:-1]) + [self.noutputs, nclasses] )

        if self.label_smoothing>0.:
            y_true = self.label_smoothing / y_true.shape[-1] * ops.sum(y_true,axis=-1,keepdims=True) + (1-self.label_smoothing) * y_true
            ## In ScanNet, for a given token, y_true does not sum to 1 but to the token weight. Need to correct for this.

        summed_loss_per_output = ops.sum( self._cce(y_true,y_pred), axis=(0,1) )
        summed_counts_per_output =  ops.sum( y_true, axis=(0,1,3) ) # For the autoregressive loss, not all heads are trained on all tokens.

        loss_per_output = summed_loss_per_output / (summed_counts_per_output + ops.array(1e-4) )
        if self.weights is not None:
            return ops.dot(loss_per_output,self.weights)
        else:
            return ops.sum(loss_per_output)



class CategoricalAccuracy_custom(Metric):
    def __init__(self, nclasses=2,name='categorical_accuracy',**kwargs):
        self.nclasses = nclasses
        super().__init__(name=name , **kwargs)
        self.correct = self.add_variable(name="correct", initializer="zeros")
        self.total = self.add_variable(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = ops.cast(ops.sum( ops.take_along_axis(y_true, ops.argmax(y_pred,axis=-1),axis=-1) ),'float')
        total = ops.cast(ops.sum(y_true),'float')
        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return self.correct / self.total    

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)
    

class MultiOutputAccuracy_custom(Metric):
    def __init__(self, noutputs=2,weights=None, name='multioutput_accuracy',**kwargs):

        self.noutputs = noutputs
        if weights is None:
            weights = ops.ones(noutputs)
        self.weights = weights

        super().__init__(name=name , **kwargs)
        self.correct = self.add_variable(name="correct", initializer="zeros",shape=(self.noutputs))
        self.total = self.add_variable(name="total", initializer="zeros",shape=(self.noutputs))


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.reshape(
            y_pred,list(y_pred.shape[:-1]) + [self.noutputs,2])

        y_true = ops.reshape(
            y_true,list(y_true.shape[:-1]) + [self.noutputs,2])        

        correct = ops.sum( y_true * ops.cast(ops.greater_equal(y_pred, 0.5), 'float'), axis=(0,1,3) )
        total = ops.sum( y_true , axis=(0,1,3) )
        self.correct.assign_add(correct)
        self.total.assign_add(total)        

    def result(self):
        accuracy_per_output = self.correct / self.total
        return ops.dot( self.weights, accuracy_per_output) / ops.sum(self.weights)

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)


AutoRegressiveAccuracy_custom = MultiOutputAccuracy_custom



class AUCPR_custom(AUC):
    def __init__(self, nclasses=2,name='AUCPR',**kwargs):
        self.nclasses = nclasses
        super().__init__(num_thresholds=200,
                    summation_method = 'interpolation',
                    curve='PR',
                    name=name,
                    multi_label = False,
                    from_logits=False,
                    **kwargs
                    )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._built:
            shape = y_pred.shape if self.nclasses>2 else y_pred[:,:,:1].shape
            self._build(shape)

        if self.nclasses>2:
            predictions = y_pred
            positive_labels = y_true
            negative_labels = ops.sum(y_true,axis=-1,keepdims=True) - y_true
        else:
            predictions = y_pred[:,:,1]
            positive_labels = y_true[:,:,1]
            negative_labels = y_true[:,:,0]

        binarized_predictions = ops.cast(ops.greater_equal(ops.expand_dims(predictions,axis=-1) , self._thresholds) ,'float32')

        true_positives = ops.sum( binarized_predictions * ops.expand_dims(positive_labels,axis=-1), axis=(0,1) )
        false_positives = ops.sum( binarized_predictions * ops.expand_dims(negative_labels,axis=-1), axis=(0,1) )
        true_negatives = ops.sum( (1-binarized_predictions) * ops.expand_dims(negative_labels,axis=-1), axis=(0,1) ) 
        false_negatives = ops.sum( (1-binarized_predictions) * ops.expand_dims(positive_labels,axis=-1), axis=(0,1) )
        
        if self.nclasses>2:
            true_positives = ops.transpose(true_positives)
            false_positives = ops.transpose(false_positives)
            true_negatives = ops.transpose(true_negatives)
            false_negatives = ops.transpose(false_negatives)

        self.true_positives.assign_add(true_positives)
        self.true_negatives.assign_add(true_negatives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)
        return


class MultiOutputAUCPR_custom(AUC):
    def __init__(self, name='multioutput_AUCPR',
        noutputs=1,weights=None,**kwargs):

        self.noutputs = noutputs
        if weights is None:
            weights = ops.ones(noutputs)
        self.weights = weights

        super().__init__(num_thresholds=200,
                    summation_method = 'interpolation',
                    curve='PR',
                    name=name,
                    num_labels=noutputs,
                    multi_label = True,
                    label_weights = self.weights,
                    from_logits=False,
                    **kwargs
                    )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self._built:
            self._build(y_pred[...,:self.noutputs].shape)
        y_pred = ops.reshape(
            y_pred,list(y_pred.shape[:-1]) + [self.noutputs,2])

        y_true = ops.reshape(
            y_true,list(y_true.shape[:-1]) + [self.noutputs,2])

        y_pred_positive = y_pred[...,1]
        y_pred_binarized =  ops.cast(ops.greater_equal( ops.expand_dims(y_pred_positive,axis=-1), self._thresholds),'float32')

        y_true_positive = y_true[...,1]
        y_true_negative = y_true[...,0]
                        
        true_positives = ops.transpose( ops.sum( y_pred_binarized * ops.expand_dims(y_true_positive,axis=-1), axis=(0,1) ) )
        false_positives = ops.transpose( ops.sum( y_pred_binarized * ops.expand_dims(y_true_negative,axis=-1), axis=(0,1) ) )        
        true_negatives = ops.transpose( ops.sum( (1-y_pred_binarized) * ops.expand_dims(y_true_negative,axis=-1), axis=(0,1) ) )
        false_negatives = ops.transpose( ops.sum( (1-y_pred_binarized) * ops.expand_dims(y_true_positive,axis=-1), axis=(0,1) ) )
        self.true_positives.assign_add(true_positives)
        self.true_negatives.assign_add(true_negatives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)
        return


AutoRegressiveAUCPR_custom = MultiOutputAUCPR_custom

    
    
