# Test 1: t against o

import numpy as np
from river import drift

# Parameters
u = 0.5
n = 10000

# Generate data for each u
print("Generating Stream...")
random_state = np.random.RandomState(seed=69)
streams = [random_state.normal(0.2, u, n) for i in range(10)]
print("Stream Generated")

print("Begin Tests...")

ave_drift = 0
for i, stream in enumerate(streams):
    print(f"Creating ADWIN Instance...")
    test_drift_detector = drift.ADWIN(delta=0.002)
    test_drift_count = 0
    print("ADWIN Instance Created")

    print("Running ADWIN...")
    for j, val in enumerate(stream):
        if j % 1000 == 0 :
            print(f"\rProgress: {j}", end = '\r')
        test_drift_detector.update(val)

        if test_drift_detector.drift_detected:
            test_drift_count += 1
    print("ADWIN Complete     ")
    print(f"ADWIN False Drifts: {test_drift_count}\n")
    ave_drift += test_drift_count

print(f"Average Drift: {ave_drift}")

results = []

T = [1, 1/2, 1/10, 1/50, 1/100, 1/1000]
O = [0.001, 0.002, 0.005, 0.01, 0.05]

for o in O:
    results.append([])
    for t in T:
        ave_drift = 0
        for i, stream in enumerate(streams):
            print(f"Creating ATWIN({o}, {t}, {i}) Instance...")
            test_drift_detector = drift.atwin.ATWIN(delta=o, temp=t)
            test_drift_count = 0
            print("ATWIN Instance Created")

            print("Running ATWIN...")
            for j, val in enumerate(stream):
                if j % 1000 == 0 :
                    print(f"\rProgress: {j}", end = '\r')
                test_drift_detector.update(val)

                if test_drift_detector.drift_detected:
                    test_drift_count += 1
            print("ATWIN Complete     ")
            print(f"ATWIN False Drifts: {test_drift_count}\n")
            ave_drift += test_drift_count

        results[-1].append(ave_drift)

print("Test Complete")
print("Differences:")
print(results)

#testpile.plot_data([[results]])