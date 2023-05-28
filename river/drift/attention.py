from __future__ import annotations
import numpy
import scipy

from river.base import DriftDetector

class ATTENTION(DriftDetector):

    def __init__(self):
        super().__init__()
        self._reset()

        self.value = []
        self.value_size = 10

    def _reset(self):
        super()._reset()

    def update(self, x):

        if self.drift_detected:
            self._reset()

        self.value.append(x)
        if len(self.value) > self.value_size:
            self.value.pop(0)

        attention_score = numpy.matmul(self.value, numpy.transpose(self.value, (1, 2)))

        attention_prob = scipy.special.softmax(attention_score)

        attention_values = numpy.matmul(attention_prob, self.value)

        print(attention_values)

        return self
