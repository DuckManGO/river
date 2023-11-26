"""
Test Stationary 0: Comprehensive Delta Test
"""

print("Starting Test: Stationary 3")

# Imports
import numpy as np
from river import drift
import statistics

###########################################################

# Parameters

#delta = [0.0005, 0.001, 0.002, 0.005, 0.01]
#delta = [0.001, 0.002, 0.05, 0.1, 0.5]
delta = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
mu = 1
n = 100000
temperature = [1, 1/2, 1/5, 1/10, 1/100, 1/1000]

###########################################################

# Generate data for each u
print("Generating Stream...")
random_state = np.random.RandomState(seed=69)
streams = [random_state.normal(0.2, mu, n) for i in range(10)]
print("Stream Generated")

###########################################################

# Functions

def average(data):
    mean = round(statistics.mean(data), 1)
    sd = round(statistics.stdev(data), 1)
    out = f"{mean} ± {sd}"
    return out

###########################################################

delta_result = []

print("Varying delta...")

for o in delta:

    print(f"Delta = {o}")

    temp_results = []

    temp_result = []

    print("Testing ADWIN (t=null)")

    for i, stream in enumerate(streams):
        print(f"\rCreating ADWIN({i}) Instance...", end = '\r')
        drift_detector = drift.ADWIN(delta=o)
        drift_count = 0

        print(f"\rRunning ADWIN({i})...", end = '\r')
        for j, val in enumerate(stream):
            if j % 1000 == 0 :
                print(f"\rProgress: {j}                ", end = '\r')
            drift_detector.update(val) 

            if drift_detector.drift_detected:
                drift_count += 1
        #print(f"False Drifts: {drift_count}           ")
        temp_result.append(drift_count)
    print(f"False Drifts: {average(temp_result)}           ")
    temp_results.append(temp_result)

    for t in temperature:
        temp_result = []

        print(f"Testing ATWIN (t={t})")

        for i, stream in enumerate(streams):
            print(f"\rCreating ATWIN({i}) Instance...", end = '\r')
            drift_detector = drift.atwin.ATWIN(delta=o, temp=t)
            drift_count = 0

            print(f"\rRunning ATWIN({i})...", end = '\r')
            for j, val in enumerate(stream):
                if j % 1000 == 0 :
                    print(f"\rProgress: {j}                  ", end = '\r')
                drift_detector.update(val) 

                if drift_detector.drift_detected:
                    drift_count += 1
            #print(f"False Drifts: {drift_count}         ")
            temp_result.append(drift_count)
        print(f"False Drifts: {average(temp_result)}           ")
        temp_results.append(temp_result)

    delta_result.append(temp_results)

for o, d in zip(delta, delta_result):
    print(f"{o} & " + " & ".join([average(t) for t in d]) + " \\\\")