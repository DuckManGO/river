# Wave 3: Stairs

import numpy as np

import matplotlib.pyplot as plt

from river import drift

# Parameters
u = 0.05
n = 250
k = 100
d = k * 2

# Generate data for 3 distributions
print("Generating Stream...")
random_state = np.random.RandomState(seed=42)
stream = np.array([])
for i in range(k):
    stream = np.concatenate((stream, random_state.normal(0.2, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.225, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.25, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.275, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.3, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.275, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.25, u, n)))
    stream = np.concatenate((stream, random_state.normal(0.225, u, n)))
print("Stream Generated")

plt.plot(stream)
plt.show()

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

for o in [0.0005, 0.001, 0.002, 0.005, 0.01]:

    print(f"Creating ATWIN({o}) Instance...")
    test_drift_detector = drift.atwin.ATWIN(delta=o, temp=1/1000)
    test_drift_count = 0
    test_drifts = []
    print("ATWIN Instance Created")

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
