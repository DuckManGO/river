import numpy as np
from time import time

from datashapes import testpile

from river import drift

##############################################################################

import matplotlib.pyplot as plt
from matplotlib import gridspec
def plot_data(stream, drifts=None):
    fig = plt.figure(figsize=(7,3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.grid()
    ax1.plot(stream, label='Stream')
    ax2.grid(axis='y')
    ax2.hist(stream)
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red')
    plt.show()

##############################################################################

# Parameters
u = 0.1
o = 0.3
n = 10000

# Generate data for 3 distributions
print("Generating Stream...")
random_state = np.random.RandomState(seed=42)
stream = random_state.normal(0.2, u, n)
print("Stream Generated")

##############################################################################

print("Creating ATWIN1 Instance...")
drift_detector_1 = drift.atwin.ATWIN(delta=o)
drift_count_1 = 0
print("ATWIN2 Instance Created")

print("Creating ATWIN2 Instance")
drift_detector_2 = drift.atwin.ATWIN2(delta=o, temp=10)
drift_count_2 = 0
print("ATWIN2 Instance Created")

##############################################################################

start = time()
print("Running ATWIN1...")
for i, val in enumerate(stream):
    if i % 1000 == 0 :
        print(f"\rProgress: {i}", end = '\r')
    drift_detector_1.update(val)   # Data is processed one sample at a time

    if drift_detector_1.drift_detected:
        drift_count_1 += 1
print("ATWIN1 Complete     ")
end = time() - start

print("Results:")
print(f"ATWIN1 False Drifts: {drift_count_1}")
print(f"False Positive Rate: {drift_count_1 / n}")
print(f"Running Time: {end}")

##############################################################################

drift = []

start = time()
print("Running ATWIN2...")
for i, val in enumerate(stream):
    if i % 1000 == 0 :
        print(f"\rProgress: {i}", end = '\r')
    drift_detector_2.update(val)

    if drift_detector_2.drift_detected:
        drift_count_2 += 1
        drift.append(i)
print("ATWIN2 Complete     ")
end = time() - start

print("Results:")
print(f"ATWIN2 False Drifts: {drift_count_2}")
print(f"False Positive Rate: {drift_count_2 / n}")
print(f"Running Time: {end}")

##############################################################################

#plot_data(stream, drift)