# Wave 0: Slope based tests

# Imports
import numpy as np
from river import drift
import statistics
import matplotlib.pyplot as plt

###########################################################

# Parameters
u = 0.05
n = 1000
k = 10
d = k * 2
slope = [0.0001, 0.0002, 0.0004, 0.001, 0.1] 
change = 0.1

###########################################################

# Generate data for 3 distributions
print("Generating Streams...")
random_state = np.random.RandomState(seed=42)

streams = {}

for s in slope:
    stream = np.array([])
    width = int(change / s)

    for i in range(k):
        up = [stream]

        for i in range(width):
            up.append(random_state.normal(0.2+change*(i/width), u, 1))

        up.append(random_state.normal(0.2 + change, u, n - width))

        for i in reversed(range(width)):
            up.append(random_state.normal(0.2+change*(i/width), u, 1))

        up.append(random_state.normal(0.2, u, n - width))

        stream = np.concatenate(up)
    
    streams[s] = stream
print("Stream Generated")

###########################################################

# Functions

def average(data):
    data = [i for i in data if i!=0]
    mean = round(statistics.mean(data), 1)
    sd = round(statistics.stdev(data), 1)
    out = f"{mean} ± {sd}"
    return out

###########################################################

slope_result = []
slope_result2 = []

print("Varying slope...")

for s in slope:
    print(f"Slope = {s}")

    stream = streams[s]

    # Adwin
    drift_detector = drift.ADWIN(delta=0.05)
    drift_count = 0
    drifts = [0] * d

    for i, val in enumerate(stream):
        if i % n == 0 :
            print(f"\rProgress: {i}", end = '\r')
        drift_detector.update(val)   # Data is processed one sample at a time

        if drift_detector.drift_detected:
            # The drift detector indicates after each sample if there is a drift in the data
            #print(f'Change detected at index {i} by ADWIN')
            #drifts.append(i)
            drift_count += 1
            current = i // n
            if drifts[current] == 0:
                drifts[current] = i % n
    print("ADWIN Complete     ")

    print("Results:")
    print(f"ADWIN Drifts: {drift_count}")
    print(f"Average Detection Time {average(drifts)}")

    slope_result.append(drifts)

    # Atwin
    test_drift_detector = drift.atwin.RTWIN(delta=0.05, temp=1/10)
    test_drift_count = 0
    test_drifts = [0] * d

    for i, val in enumerate(stream):
        if i % n == 0 :
            print(f"\rProgress: {i}", end = '\r')
        test_drift_detector.update(val)

        if test_drift_detector.drift_detected:
            test_drift_count += 1
            current = i // n
            if test_drifts[current] == 0:
                test_drifts[current] = i % n
    print("ATWIN Complete     ")

    print("Results:")
    print(f"ATWIN Drifts: {test_drift_count}")
    print(f"Average Detection Time {average(test_drifts)}")

    slope_result2.append(test_drifts)

print("ADWIN & " + " & ".join([average(t) for t in slope_result]) + " \\\\")
print("ATWIN & " + " & ".join([average(t) for t in slope_result2]) + " \\\\")