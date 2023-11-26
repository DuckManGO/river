"""
Test 2: Using river inbuilt datasets \w Retraining and Varying delta
A few drifting datasets from the river package is used for 
evaluation. In this situation the default setting was used. 
In general, it would look as if ATWIN is doing better, but 
it is mainly due to the extra sensitivity. How it compares 
against different parameters are to be tested next.
"""

###########################################################

from river import naive_bayes, stream, drift, metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

###########################################################

# Dataset
dataset = drift.datasets.AirlinePassengers()
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

###########################################################

# Test NB Models
class Model:
    def __init__(self, detector, metric, r=100):
        self.model = naive_bayes.GaussianNB()
        self.detector = detector
        self.metric = metric
        self.errors = []
        self.drifts = []
        self.retrain_set = []
        self.r = r

    def foward(self, x, y, i):
        self.model.learn_one(x, y)

        self.retrain_set.append((x, y))
        if len(self.retrain_set) > self.r:
            self.retrain_set.pop(0)

        e = self.metric.update(y, (self.model.predict_one(x) or 0.)).get()
        self.detector.update(e)
        self.errors.append(e)
        if self.detector.drift_detected:
            self.model = naive_bayes.GaussianNB()
            self.drifts.append(i)
            for _x, _y in self.retrain_set:
                    self.model.learn_one(x, y)
        
        return self.errors, self.drifts
    
    def get_eval(self):
        return (self.errors, self.drifts)

###########################################################

# Evaluation
def plot_data(dist_a, dist_b):
    fig = plt.figure(figsize=(7,6), tight_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 3])
    ax1, ax2, ax3 = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

    ax1.grid()
    ax1.plot(dist_a[0], label='ADWIN')
    ax2.grid()
    ax2.plot(dist_b[0], label='ATWIN')
    ax3.grid()
    ax3.plot(dist_a[0], label='Compare')
    ax3.plot(dist_b[0], label='Compare')

    if dist_a[1] is not None:
        for drift_detected in dist_a[1]:
            ax1.axvline(drift_detected, color='red')
            ax3.axvline(drift_detected, color='red')
    if dist_b[1] is not None:
        for drift_detected in dist_b[1]:
            ax2.axvline(drift_detected, color='red')
            ax3.axvline(drift_detected, color='red')
    plt.show()

###########################################################

for d in [0.0005, 0.001, 0.002, 0.005, 0.01]:
    NBAdwin = Model(drift.ADWIN(delta=d), metrics.MSE())
    NBAtwin = Model(drift.atwin.ATWIN(delta=d, temp=1/1000), metrics.MSE())

    i = 0
    for x, y in stream.iter_array(X, Y):
        NBAdwin.foward(x, y, i)
        NBAtwin.foward(x, y, i)
        i += 1

    print(f"\nd = {d}")
    print(sum(NBAdwin.get_eval()[0]) / len(NBAdwin.get_eval()[0]))
    print(sum(NBAtwin.get_eval()[0]) / len(NBAtwin.get_eval()[0]))

    plot_data(NBAdwin.get_eval(), NBAtwin.get_eval())

###########################################################



