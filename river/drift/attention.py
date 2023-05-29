from __future__ import annotations
import numpy
import scipy
from tensorflow import matmul, math, cast, float32
from tensorflow.python.keras.layers import Layer
from keras.backend import softmax

from river.base import DriftDetector

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

class ATTENTION(DriftDetector):

    def __init__(self):
        super().__init__()
        self._reset()

        self.value_size = 10
        self.value = [0 for i in range(self.value_size)]

    def _reset(self):
        super()._reset()

    def update(self, x):
        print('attention updating')
        if self.drift_detected:
            self._reset()
        raise()
        self.value.append([x])
        self.value.pop(0)
 
        d_k = 1  # Dimensionality of the linearly projected queries and keys
        
        queries = numpy.array([self.value])
        keys = numpy.array([self.value])
        values = numpy.array([self.value])
        
        print(values)

        attention = DotProductAttention()
        print(attention(queries, keys, values, d_k))
        
        return self
    
    
