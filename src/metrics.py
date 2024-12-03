import tensorflow as tf
from keras.metrics import Metric

class SparseCategoricalAccuracyIgnoreClass(Metric):
    def __init__(self, ignore_class, **kwargs):
        super().__init__(name="accuracy", **kwargs)
        self.ignore_class = tf.constant(ignore_class, dtype=tf.int64)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.correct = self.add_weight(name="correct", initializer="zeros")
    
    def update_state(
        self, 
        y_true, # (N,1)
        y_pred, # (N,n_classes)
        sample_weight=None
    ):
        # Cast y_true to int64 to ensure compatibility
        y_true = tf.cast(tf.squeeze(y_true,axis=-1), tf.int64) # (N,)
        
        # Find the mask of values not equal to ignore_class
        mask = tf.not_equal(y_true, self.ignore_class) # (N,)
        
        # Mask the true labels and predictions
        y_true = tf.boolean_mask(y_true, mask) # (N',)
        y_pred = tf.boolean_mask(y_pred, mask) # (N', n_classes)
        
        # Compute the predicted classes
        y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int64) # (N')
        
        # Count matches
        matches = tf.cast(tf.equal(y_true, y_pred_classes), self.dtype) # (N')
        
        # Update total and correct counts
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), self.dtype))
    
    def result(self):
        # Compute accuracy
        return tf.math.divide_no_nan(self.correct, self.total)
    
    def reset_state(self):
        # Reset state variables
        self.total.assign(0)
        self.correct.assign(0)