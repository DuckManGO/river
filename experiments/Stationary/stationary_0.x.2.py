"""
Test Stationary 0: Comprehensive Mu Test
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
delta = 0.05 #[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
mu = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1]
n = 100000
temperature = [1, 1/2, 1/5, 1/10, 1/100, 1/1000]

###########################################################

# Generate data for each u
print("Generating Stream...")
random_state = np.random.RandomState(seed=69)
streams = [[random_state.normal(0.2, u, n) for i in range(10)] for u in mu]
print("Stream Generated")

###########################################################

# Functions

def average(data):
    mean = round(statistics.mean(data), 1)
    sd = round(statistics.stdev(data), 1)
    out = f"{mean} ± {sd}"
    return out

###########################################################

mu_result = []

print("Varying mu...")

for u, _streams in zip(mu, streams):

    print(f"Mu = {u}")

    temp_results = []

    temp_result = []

    print("Testing ADWIN (t=null)")

    for i, stream in enumerate(_streams):
        print(f"\rCreating ADWIN({i}) Instance...", end = '\r')
        drift_detector = drift.ADWIN(delta=delta)
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

        for i, stream in enumerate(_streams):
            print(f"\rCreating ATWIN({i}) Instance...", end = '\r')
            drift_detector = drift.atwin.RTWIN(delta=delta, temp=t)
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

    mu_result.append(temp_results)

for u, d in zip(mu, mu_result):
    print(f"{u} & " + " & ".join([average(t) for t in d]) + " \\\\")