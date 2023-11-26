"""
Test 3: Using Elec2 dataset
Elec2 is a real life dataset from the Australian New South 
Wales Electricity Market. We do not know if there is a 
drift or not.
"""

###########################################################

from river import naive_bayes, stream, drift, metrics, datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

###########################################################

# Dataset
print("\rDownloading Dataset...", end = '\r')
dataset = datasets.Elec2()
if not dataset.is_downloaded:
    dataset.download()
file = open(dataset.path, 'r')
content = file.readlines()[1:]
file.close()

print("Dataset Downloaded      ")

k = 48
h = []
_X = []
_Y = []

error = []

# 3-6
print("\rFormatting Dataset...", end = '\r')
for i in range(k):
    h = h + [float(j) for j in content[i].split(',')[3:7]]
for i in range(k, len(content)):
    c = [float(j) for j in content[i].split(',')[3:7]]
    _X.append(h)
    if content[i].split(',')[-1] == "UP\n":
        _Y.append(True)
    else:
        _Y.append(False)
    h = h[4:] + c

X = np.array(_X)
Y = np.array(_Y)

print("Dataset Formatted         ")

###########################################################

# Test NB Models

print("\rCreating Models...", end = '\r')

class Model:
    def __init__(self, detector):
        #self._model = naive_bayes.GaussianNB()
        self.model = naive_bayes.GaussianNB() #copy.copy(self._model)
        self._detector = detector
        self.detector = copy.copy(self._detector)
        self.acc = 0
        self.accL = 48
        self.errors = []
        self.drifts = []
        self.retrain_set = []
        self.k = 48 * 7

    def foward(self, x, y, i):
        # Training
        self.model.learn_one(x, y)

        # Retraining Set
        self.retrain_set.append((x, y))
        if len(self.retrain_set) > self.k:
            self.retrain_set.pop(0)

        # Accuracy
        if y == self.model.predict_one(x):
            self.acc += 1
        
        # Every day (48)
        if i % self.accL == 0:
            e = self.acc / self.accL
            self.acc = 0

            self.errors.append(e)

            if self.detector:
                self.detector.update(e)

                # Drift Detected
                if self.detector.drift_detected:
                    self.model = naive_bayes.GaussianNB() #copy.copy(self._model)
                    self.drifts.append(i)

                    # Retrain
                    for _x, _y in self.retrain_set:
                        self.model.learn_one(_x, _y)

            return e
        return 0
    
    def get_eval(self):
        return (self.errors, self.drifts)

NB = Model(None)    
NBAdwin = Model(drift.ADWIN())
NBAtwin = Model(drift.atwin.ATWIN(delta=0.0005, temp=1/10))

print("Models Created          ")

###########################################################

# Training
i = 0

for x, y in stream.iter_array(X, Y):
    NB.foward(x, y, i)
    NBAdwin.foward(x, y, i)
    NBAtwin.foward(x, y, i)
    i += 1
    if i % 100 == 0:
        print(f"\rTraining Progress: {i}/{len(X)}", end = '\r')
print("Training Completed             ")

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

def cram_plot(nb, da, db):
    plt.plot(nb[0], color='black')
    plt.plot(da[0], color="blue")
    plt.plot(db[0], color='red')

    """
    if da[1] is not None:
        for drift_detected in da[1]:
            plt.axvline(drift_detected, color='red')
    if db[1] is not None:
        for drift_detected in db[1]:
            plt.axvline(drift_detected, color='blue')
    """

    plt.legend(["NB", "NBAdwin", "NBAtwin"], loc ="lower right") 
    plt.xlabel("Examples")
    plt.ylabel("Accuracy")

    plt.show()

#plot_data(NBAdwin.get_eval(), NBAtwin.get_eval())

Nb = NB.get_eval()
Ad = NBAdwin.get_eval()
At = NBAtwin.get_eval()

cram_plot(Nb, Ad, At)

print(f'Control: {sum(Nb[0])/len(Nb[0])}')
print(f'Adwin: {sum(Ad[0])/len(Ad[0])}')
print(f'    Drifts: {len(Ad[1])}')
print(f'Atwin: {sum(At[0])/len(At[0])}')
print(f'    Drifts: {len(At[1])}')

"""
plt.plot(Y)
plt.show()
"""