
import pickle
import numpy as np
# Load one subject file
file_path = r"D:\somma\research\dataset\ppg+dalia\data\PPG_FieldStudy\S1\S1.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f,encoding="latin1")

# Check keys
print("Keys in dataset:", data.keys())
print(type(data))
print(data['signal'].keys())
print(data['signal']['wrist'].keys())
# Extract signals
bvp = data['signal']['wrist']['BVP']
acc = data['signal']['wrist']['ACC']
eda = data['signal']['wrist']['EDA']
temp = data['signal']['wrist']['TEMP']

# Labels (heart rate per window)
hr = data['label']

print("BVP shape:", bvp.shape)
print("ACC shape:", acc.shape)
print("EDA shape:", eda.shape)
print("TEMP shape:", temp.shape)
print("HR labels:", len(hr))
print(type(bvp), type(acc))

bvp = np.array(bvp)
acc = np.array(acc)
eda = np.array(eda)
temp = np.array(temp)
import numpy as np

bvp = bvp.flatten()
eda = eda.flatten()
temp = temp.flatten()
def create_windows(signal, window_size, step):
    windows = []
    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i+window_size])
    return np.array(windows)
# BVP (64 Hz)
bvp_windows = create_windows(bvp, 512, 128)

# ACC (32 Hz)
acc_windows = create_windows(acc, 256, 64)

# EDA (4 Hz)
eda_windows = create_windows(eda, 32, 8)

# TEMP (4 Hz)
temp_windows = create_windows(temp, 32, 8)
print("BVP windows:", len(bvp_windows))
print("ACC windows:", len(acc_windows))
print("EDA windows:", len(eda_windows))
print("TEMP windows:", len(temp_windows))
print("HR labels:", len(hr))
min_len = min(
    len(bvp_windows),
    len(acc_windows),
    len(eda_windows),
    len(temp_windows),
    len(hr)
)

bvp_windows = bvp_windows[:min_len]
acc_windows = acc_windows[:min_len]
eda_windows = eda_windows[:min_len]
temp_windows = temp_windows[:min_len]
hr = hr[:min_len]

print("after aligning")

print("BVP windows:", len(bvp_windows))
print("ACC windows:", len(acc_windows))
print("EDA windows:", len(eda_windows))
print("TEMP windows:", len(temp_windows))
print("HR labels:", len(hr))
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, welch

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def entropy(signal):
    prob = np.abs(signal) / (np.sum(np.abs(signal)) + 1e-6)
    return -np.sum(prob * np.log(prob + 1e-6))

def band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(psd[mask])

# -----------------------------
# BVP FEATURES (29 FEATURES)
# -----------------------------
def extract_bvp_features(window, fs=64):
    features = []

    # --- Basic (1–7)
    features += [
        np.mean(window),                # 1
        np.std(window),                 # 2
        np.min(window),                 # 3
        np.max(window),                 # 4
        np.sum(window**2),              # 5
        skew(window),                   # 6
        kurtosis(window)                # 7
    ]

    # --- Entropy (8)
    features.append(entropy(window))    # 8

    # --- Peaks (9–14)
    peaks, properties = find_peaks(window, distance=20)

    if len(peaks) > 2:
        ibi = np.diff(peaks) / fs

        features += [
            len(peaks),                     # 9
            np.mean(ibi),                   # 10
            np.std(ibi),                    # 11
            60 / (np.mean(ibi)+1e-6),       # 12 HR
            np.sqrt(np.mean(np.diff(ibi)**2)), # 13 RMSSD
            np.sum(np.abs(np.diff(ibi)) > 0.05) / len(ibi) # 14 pNN50
        ]

        peak_vals = window[peaks]

        # --- Peak Shape (15–18)
        features += [
            np.mean(peak_vals),             # 15
            np.std(peak_vals),              # 16
            np.mean(np.diff(peaks)),        # 17 width approx
            np.mean(properties.get("prominences", np.zeros(len(peaks)))) # 18
        ]
    else:
        features += [0]*10

    # --- Frequency (19–20)
    freqs, psd = welch(window, fs=fs)
    features.append(freqs[np.argmax(psd)])   # 19
    features.append(entropy(psd))            # 20

    # --- HRV Bands (21–23)
    lf = band_power(freqs, psd, 0.04, 0.15)
    hf = band_power(freqs, psd, 0.15, 0.4)
    features += [
        lf,              # 21
        hf,              # 22
        lf/(hf+1e-6)     # 23
    ]

    # --- Energy Bands (24–25)
    features += [
        band_power(freqs, psd, 0.5, 2),  # 24
        band_power(freqs, psd, 2, 5)     # 25
    ]

    # --- Dynamics (26–27)
    zcr = np.sum(np.diff(np.sign(window)) != 0)
    snr = np.mean(np.abs(window)) / (np.std(np.diff(window)) + 1e-6)

    features += [
        zcr,  # 26
        snr   # 27
    ]

    # --- Signal Quality (28–29)
    smooth = np.convolve(window, np.ones(5)/5, mode='same')
    features += [
        np.var(smooth),                        # 28
        np.sum(window**2)/(np.sum(np.abs(window))+1e-6) # 29
    ]

    return features


# -----------------------------
# ACC FEATURES (19 FEATURES)
# -----------------------------
def extract_acc_features(window):
    window = np.array(window)

    mag = np.sqrt(np.sum(window**2, axis=1))
    jerk = np.diff(mag)

    features = [
        np.mean(mag),   # 30
        np.std(mag),    # 31
        np.max(mag),    # 32
        np.sum(mag**2), # 33
        np.mean(jerk),  # 34
        np.std(jerk),   # 35
        np.percentile(mag, 25), # 36
        np.percentile(mag, 75)  # 37
    ]

    # Axis-specific (38–43)
    features += [
        np.mean(window[:,0]), # 38
        np.mean(window[:,1]), # 39
        np.mean(window[:,2]), # 40
        np.std(window[:,0]),  # 41
        np.std(window[:,1]),  # 42
        np.std(window[:,2])   # 43
    ]

    # Correlations (44–46)
    features += [
        np.nan_to_num(np.corrcoef(window[:,0], window[:,1])[0,1]), # 44
        np.nan_to_num(np.corrcoef(window[:,1], window[:,2])[0,1]), # 45
        np.nan_to_num(np.corrcoef(window[:,0], window[:,2])[0,1])  # 46
    ]

    # Frequency (47–48)
    freqs, psd = welch(mag)
    features += [
        freqs[np.argmax(psd)], # 47
        entropy(psd)           # 48
    ]

    return features


# -----------------------------
# EDA FEATURES (9 FEATURES)
# -----------------------------
def extract_eda_features(window):
    peaks, _ = find_peaks(window)

    features = [
        np.mean(window), # 49
        np.std(window),  # 50
        np.min(window),  # 51
        np.max(window),  # 52
        len(peaks)       # 53
    ]

    if len(peaks) > 0:
        peak_vals = window[peaks]
        features += [
            np.mean(peak_vals),            # 54
            np.mean(np.diff(peaks)) if len(peaks)>1 else 0, # 55
            np.mean(window)-np.min(window), # 56
            np.sum(peak_vals)              # 57
        ]
    else:
        features += [0]*4

    return features


# -----------------------------
# TEMP FEATURES (6 FEATURES)
# -----------------------------
def extract_temp_features(window):
    slope = window[-1] - window[0]

    return [
        np.mean(window),           # 58
        np.std(window),            # 59
        slope,                     # 60
        slope / len(window),       # 61
        np.max(window)-np.min(window), # 62
        np.var(window)             # 63
    ]


# -----------------------------
# CROSS FEATURES (64–66)
# -----------------------------
def cross_features(bvp, acc, eda, temp):
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))

    min_len = min(len(bvp), len(acc_mag))
    c1 = np.corrcoef(bvp[:min_len], acc_mag[:min_len])[0,1]

    c2 = np.corrcoef(eda, bvp[:len(eda)])[0,1]
    c3 = np.corrcoef(temp, eda[:len(temp)])[0,1]

    return [
        np.nan_to_num(c1), # 64
        np.nan_to_num(c2), # 65
        np.nan_to_num(c3)  # 66
    ]


# -----------------------------
# FINAL FEATURE MATRIX
# -----------------------------
X = []
prev_hr = 0

for i in range(len(hr)):
    features = []

    features += extract_bvp_features(bvp_windows[i])
    features += extract_acc_features(acc_windows[i])
    features += extract_eda_features(eda_windows[i])
    features += extract_temp_features(temp_windows[i])
    features += cross_features(bvp_windows[i], acc_windows[i], eda_windows[i], temp_windows[i])

    # Temporal features (67–69)
    delta_hr = hr[i] - prev_hr if i > 0 else 0
    rolling_mean = np.mean(hr[max(0,i-5):i+1])
    rolling_std = np.std(hr[max(0,i-5):i+1])

    features += [
        delta_hr,     # 67
        rolling_mean, # 68
        rolling_std   # 69
    ]

    prev_hr = hr[i]

    X.append(features)

X = np.array(X)
y = np.array(hr)

print("Final Feature Shape:", X.shape)
print("Total Features:", X.shape[1])
print("Any NaN:", np.isnan(X).any())
# -----------------------------
# SAVE FEATURES
# -----------------------------
np.save("features.npy", X)
np.save("labels.npy", y)
# -----------------------------
# ACTIVITY LABEL CREATION (FIXED)
# -----------------------------
activity_labels = np.array(data['activity']).flatten()

# Align with feature windows
activity_labels = activity_labels[:min_len]

np.save("activity.npy", activity_labels)

print("Activity labels shape:", activity_labels.shape)
print("Features saved successfully!")