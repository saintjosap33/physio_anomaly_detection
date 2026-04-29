import numpy as np
from scipy.stats import skew, kurtosis

def extract_features_from_csv(data):
    """
    data shape = (60, 4)
    columns = [bvp, eda, temp, acc]
    """

    bvp = data[:, 0]
    eda = data[:, 1]
    temp = data[:, 2]
    acc = data[:, 3]

    features = []

    # ======================
    # 1️⃣ BASIC STATS (6 × 4 = 24)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        features += [
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            skew(signal),
            kurtosis(signal)
        ]

    # ======================
    # 2️⃣ SIGNAL ENERGY (4)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        features.append(np.sum(signal ** 2))

    # ======================
    # 3️⃣ ZERO CROSSINGS (4)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        zero_crossings = np.where(np.diff(np.sign(signal)))[0].size
        features.append(zero_crossings)

    # ======================
    # 4️⃣ DIFFERENCE FEATURES (8)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        diff = np.diff(signal)
        features += [
            np.mean(diff),
            np.std(diff)
        ]

    # ======================
    # 5️⃣ RANGE + MEDIAN (8)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        features += [
            np.ptp(signal),        # range
            np.median(signal)
        ]

    # ======================
    # 6️⃣ PERCENTILES (12)
    # ======================
    for signal in [bvp, eda, temp, acc]:
        features += [
            np.percentile(signal, 25),
            np.percentile(signal, 50),
            np.percentile(signal, 75)
        ]

    # ======================
    # 7️⃣ CORRELATIONS (6)
    # ======================
    signals = [bvp, eda, temp, acc]
    for i in range(4):
        for j in range(i+1, 4):
            corr = np.corrcoef(signals[i], signals[j])[0, 1]
            features.append(corr)

    # ======================
    # FINAL CHECK
    # ======================
    features = np.array(features)

    # Ensure exactly 69 features
    if len(features) < 69:
        features = np.pad(features, (0, 69 - len(features)))
    elif len(features) > 69:
        features = features[:69]

    return features.reshape(1, -1)