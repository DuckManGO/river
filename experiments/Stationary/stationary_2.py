import numpy as np
from river import drift
from datashapes import testpile

def test(stream):
    print("Creating ATWIN Instance...")
    test_drift_detector = drift.atwin.ATWIN(delta=o)
    test_drift_count = 0
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
    print("ATWIN Complete")

    print("Results:")
    print(f"ATWIN False Drifts: {test_drift_count}")
    print(f"False Positive Rate: {test_drift_count / n}")

    results.append(drift_count - test_drift_count)

# Parameters
u = 0.5
o = 0.3
n = 1000000

# Generate data for each u
print("Generating Stream...")
random_state = np.random.RandomState(seed=69)
streams = [random_state.normal(0.2, u, n) for i in range(10)]
print("Stream Generated")

print("Begin Tests...")
results = []
ave_drift = 0

for stream in streams:
    print("Creating ADWIN Instance")
    drift_detector = drift.ADWIN(delta=o)
    drift_count = 0
    print("ATWIN Instance Created")

    print("Running ADWIN...")
    for i, val in enumerate(stream):
        if i % 1000 == 0 :
            print(f"\rProgress: {i}", end = '\r')
        drift_detector.update(val)   # Data is processed one sample at a time

        if drift_detector.drift_detected:
            drift_count += 1
    print("ADWIN Complete     ")
    print(f"ADWIN False Drifts: {drift_count}")
    ave_drift += drift_count

results.append(ave_drift)

for i in range(1, 10):
    ave_drift = 0

    for stream in streams:
        print(f"Creating ATWIN({i}) Instance...")
        test_drift_detector = drift.ATWIN(delta=o, temp=1/i)
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
        print(f"ATWIN False Drifts: {test_drift_count}")
        ave_drift += test_drift_count

    results.append(ave_drift)

print("Test Complete")
print("Differences:")
print(results)

testpile.plot_data([[results]])