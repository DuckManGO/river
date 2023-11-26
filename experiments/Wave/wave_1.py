# Wave 1: Square

import numpy as np

import matplotlib.pyplot as plt

from river import drift

# Parameters
u = 0.05
o = 0.002
n = 10000
k = 10
d = k * 2

# Generate data for 3 distributions
print("Generating Stream...")
random_state = np.random.RandomState(seed=42)
stream = np.array([])
for i in range(k):
    stream = np.concatenate((stream, random_state.normal(0.2, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.3, u, n)))
print("Stream Generated")

plt.plot(stream)
plt.show()

print("Creating ATWIN Instance...")
test_drift_detector = drift.atwin.ATWIN(delta=0.001, temp=1/100)
test_drift_count = 0
test_drifts = []
print("ATWIN Instance Created")

print("Creating ADWIN Instance")
drift_detector = drift.ADWIN(delta=0.002)
drift_count = 0
drifts = []
print("ATWIN Instance Created")

print("Running ADWIN...")
for i, val in enumerate(stream):
    if i % 1000 == 0 :
        print(f"\rProgress: {i}", end = '\r')
    drift_detector.update(val)   # Data is processed one sample at a time

    if drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        #print(f'Change detected at index {i} by ADWIN')
        #drifts.append(i)
        drift_count += 1
        drifts.append(i % n)
print("ADWIN Complete     ")

print("Results:")
print(f"ADWIN Drifts: {drift_count}")
print(f"Detection Rate: {drift_count / d}")
print(f"Average Detection Time {sum(drifts)/len(drifts)}")

print("Running ATWIN...")
for i, val in enumerate(stream):
    if i % 1000 == 0 :
        print(f"\rProgress: {i}", end = '\r')
    test_drift_detector.update(val)

    if test_drift_detector.drift_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        #print(f'Change detected at index {i} by ATWIN')
        #test_drifts.append(i)
        test_drift_count += 1
        test_drifts.append(i % n)
print("ATWIN Complete     ")

print("Results:")
print(f"ATWIN Drifts: {test_drift_count}")
print(f"Detection Rate: {test_drift_count / d}")
print(f"Average Detection Time {sum(test_drifts)/len(drifts)}")