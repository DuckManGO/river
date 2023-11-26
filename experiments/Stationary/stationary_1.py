import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from time import time

from datashapes import testpile

from river import drift

# Parameters
u = 0.5
o = 0.3
n = 1000000

# Generate data for 3 distributions
print("Generating Stream...")
random_state = np.random.RandomState(seed=42)
stream = random_state.normal(0.2, u, n)
print("Stream Generated")

print("Creating ATWIN Instance...")
test_drift_detector = drift.ATWIN(delta=o)
test_drift_count = 0
print("ATWIN Instance Created")

print("Creating ADWIN Instance")
drift_detector = drift.ADWIN(delta=o)
drift_count = 0
print("ATWIN Instance Created")

start = time()
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
print("ADWIN Complete")
end = time() - start

print("Results:")
print(f"ADWIN False Drifts: {drift_count}")
print(f"False Positive Rate: {drift_count / n}")
print(f"Running Time: {end}")

start = time()
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
print("ATWIN Complete")
end = time() - start

print("Results:")
print(f"ATWIN False Drifts: {test_drift_count}")
print(f"False Positive Rate: {test_drift_count / n}")
print(f"Running Time: {end}")