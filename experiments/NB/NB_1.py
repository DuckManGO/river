"""
Test 1: Naive Bayes Predictor without Drift Detection
This is a simple Naive Bayes Predictor tested on the 
datasets from the river package. These datasets are 
selected to garentee drift in order to see how this model 
will do without drift detection. In general, it seems this 
model does poorly overall even without drift.
"""

###########################################################

from river import naive_bayes, stream, drift, metrics, datasets
import numpy as np
import matplotlib.pyplot as plt

model = naive_bayes.GaussianNB()
dataset = drift.datasets.AirlinePassengers()

metric = metrics.MAE()

file = open(dataset.path, 'r')
content = file.readlines()[1:]
file.close()

k = 10
h = []
_X = []
_Y = []

error = []

for i in range(k):
    h.append(float(content[i].split(',')[1]))
for i in range(k, len(content)):
    c = float(content[i].split(',')[1])
    _X.append(h)
    _Y.append(c)
    h.append(c)
    h.pop(0)

X = np.array(_X)
Y = np.array(_Y)

for x, y in stream.iter_array(X, Y):
    e = metric.update(y, (model.predict_one(x) or 0.)).get()
    error.append(e)
    model.learn_one(x, y)

print(error)
plt.plot(error)
plt.show()

plt.plot(Y)
plt.show()