# ==========================================================
# MULTI-SUBJECT FEATURE EXTRACTION PIPELINE
# PPG-DaLiA Dataset
# ==========================================================

import os
import pickle
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, welch, correlate
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ==========================================================
# DATASET PATH
# ==========================================================
dataset_path = r"D:\somma\research\dataset\ppg+dalia\data\PPG_FieldStudy"


# ==========================================================
# GLOBAL CONTAINERS
# ==========================================================
all_X = []
all_y = []
all_activity = []
all_subjects = []


# ==========================================================
# WINDOWING
# ==========================================================
def create_windows(signal, window_size, step):
    windows = []

    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i + window_size])

    return np.array(windows)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def safe_entropy(signal):
    prob = np.abs(signal) / (np.sum(np.abs(signal)) + 1e-9)
    return -np.sum(prob * np.log(prob + 1e-9))


def band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(psd[mask]))


def safe_corr(a, b):

    n = min(len(a), len(b))

    a = a[:n]
    b = b[:n]

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def z_norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def downsample(signal, factor):

    n = (len(signal) // factor) * factor

    return signal[:n].reshape(-1, factor).mean(axis=1)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_bvp_features(window, fs=64):

    features = []

    features += [
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        np.sum(window ** 2),
        skew(window),
        kurtosis(window)
    ]

    features.append(safe_entropy(window))

    peaks, properties = find_peaks(window, distance=20)

    if len(peaks) > 2:

        ibi = np.diff(peaks) / fs

        features += [
            len(peaks),
            np.mean(ibi),
            np.std(ibi),
            60 / (np.mean(ibi) + 1e-6),
            np.sqrt(np.mean(np.diff(ibi) ** 2)),
            np.sum(np.abs(np.diff(ibi)) > 0.05) / len(ibi)
        ]

        peak_vals = window[peaks]

        features += [
            np.mean(peak_vals),
            np.std(peak_vals),
            np.mean(np.diff(peaks)),
            np.mean(properties.get("prominences", np.zeros(len(peaks))))
        ]

    else:
        features += [0] * 10

    freqs, psd = welch(window, fs=fs)

    features.append(freqs[np.argmax(psd)])
    features.append(safe_entropy(psd))

    lf = band_power(freqs, psd, 0.04, 0.15)
    hf = band_power(freqs, psd, 0.15, 0.40)

    features += [
        lf,
        hf,
        lf / (hf + 1e-6),
        band_power(freqs, psd, 0.5, 2),
        band_power(freqs, psd, 2, 5)
    ]

    zcr = np.sum(np.diff(np.sign(window)) != 0)

    snr = np.mean(np.abs(window)) / (
        np.std(np.diff(window)) + 1e-6
    )

    features += [zcr, snr]

    smooth = np.convolve(
        window,
        np.ones(5) / 5,
        mode='same'
    )

    features += [
        np.var(smooth),
        np.sum(window ** 2) / (
            np.sum(np.abs(window)) + 1e-6
        )
    ]

    return features


def extract_acc_features(window):

    window = np.array(window)

    mag = np.sqrt(np.sum(window ** 2, axis=1))

    jerk = np.diff(mag)

    features = [
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.sum(mag ** 2),
        np.mean(jerk),
        np.std(jerk),
        np.percentile(mag, 25),
        np.percentile(mag, 75)
    ]

    features += [
        np.mean(window[:, 0]),
        np.mean(window[:, 1]),
        np.mean(window[:, 2]),
        np.std(window[:, 0]),
        np.std(window[:, 1]),
        np.std(window[:, 2])
    ]

    features += [
        np.nan_to_num(
            np.corrcoef(window[:, 0], window[:, 1])[0, 1]
        ),

        np.nan_to_num(
            np.corrcoef(window[:, 1], window[:, 2])[0, 1]
        ),

        np.nan_to_num(
            np.corrcoef(window[:, 0], window[:, 2])[0, 1]
        )
    ]

    freqs, psd = welch(mag)

    features += [
        freqs[np.argmax(psd)],
        safe_entropy(psd)
    ]

    return features


def extract_eda_features(window):

    peaks, _ = find_peaks(window)

    features = [
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        len(peaks)
    ]

    if len(peaks) > 0:

        peak_vals = window[peaks]

        features += [
            np.mean(peak_vals),
            np.mean(np.diff(peaks))
            if len(peaks) > 1 else 0,
            np.mean(window) - np.min(window),
            np.sum(peak_vals)
        ]

    else:
        features += [0] * 4

    return features


def extract_temp_features(window):

    slope = window[-1] - window[0]

    return [
        np.mean(window),
        np.std(window),
        slope,
        slope / len(window),
        np.max(window) - np.min(window),
        np.var(window)
    ]


def cross_features_original(
        bvp,
        acc,
        eda,
        temp):

    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    return [
        safe_corr(bvp, acc_mag[:len(bvp)]),
        safe_corr(eda, bvp[:len(eda)]),
        safe_corr(temp, eda[:len(temp)])
    ]


# ==========================================================
# SIMPLE ADVANCED CROSS FEATURES
# ==========================================================
def extract_advanced_cross_signal(
        bvp_w,
        acc_w,
        eda_w,
        temp_w):

    bvp_ds = downsample(bvp_w, 16)

    acc_mag = np.sqrt(np.sum(acc_w ** 2, axis=1))

    acc_ds = downsample(acc_mag, 8)

    eda_ds = eda_w[:32]

    temp_ds = temp_w[:32]

    features = []

    signals = [
        bvp_ds,
        acc_ds,
        eda_ds,
        temp_ds
    ]

    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):

            a = signals[i]
            b = signals[j]

            features += [
                safe_corr(a, b),
                np.mean(np.abs(a - b)),
                np.std(a - b),
                np.sum(a * b),
                safe_entropy(a / (b + 1e-6))
            ]

    while len(features) < 120:
        features.append(0)

    return features[:120]


# ==========================================================
# SUBJECT LOOP
# ==========================================================
subjects = sorted(
    [s for s in os.listdir(dataset_path)
     if s.startswith("S")],
    key=lambda x: int(x.replace("S", ""))
)

print("\nSubjects Found:")
print(subjects)


for subject in subjects:

    print(f"\nProcessing {subject}...")

    try:

        subject_file = os.path.join(
            dataset_path,
            subject,
            f"{subject}.pkl"
        )

        with open(subject_file, "rb") as f:
            data = pickle.load(
                f,
                encoding="latin1"
            )

        # ==================================================
        # LOAD SIGNALS
        # ==================================================
        bvp = np.array(
            data['signal']['wrist']['BVP']
        ).flatten()

        acc = np.array(
            data['signal']['wrist']['ACC']
        )

        eda = np.array(
            data['signal']['wrist']['EDA']
        ).flatten()

        temp = np.array(
            data['signal']['wrist']['TEMP']
        ).flatten()

        hr = np.array(
            data['label']
        ).flatten()

        # ==================================================
        # WINDOWING
        # ==================================================
        bvp_windows = create_windows(
            bvp,
            512,
            128
        )

        acc_windows = create_windows(
            acc,
            256,
            64
        )

        eda_windows = create_windows(
            eda,
            32,
            8
        )

        temp_windows = create_windows(
            temp,
            32,
            8
        )

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

        activity_labels = np.array(
            data['activity']
        ).flatten()[:min_len]

        prev_hr = 0

        # ==================================================
        # FEATURE EXTRACTION
        # ==================================================
        for i in range(min_len):

            row = []

            # ORIGINAL FEATURES
            row += extract_bvp_features(
                bvp_windows[i]
            )

            row += extract_acc_features(
                acc_windows[i]
            )

            row += extract_eda_features(
                eda_windows[i]
            )

            row += extract_temp_features(
                temp_windows[i]
            )

            row += cross_features_original(
                bvp_windows[i],
                acc_windows[i],
                eda_windows[i],
                temp_windows[i]
            )

            delta_hr = hr[i] - prev_hr \
                if i > 0 else 0

            rolling_mean = np.mean(
                hr[max(0, i-5):i+1]
            )

            rolling_std = np.std(
                hr[max(0, i-5):i+1]
            )

            row += [
                delta_hr,
                rolling_mean,
                rolling_std
            ]

            prev_hr = hr[i]

            # ADVANCED FEATURES
            row += extract_advanced_cross_signal(
                bvp_windows[i],
                acc_windows[i],
                eda_windows[i],
                temp_windows[i]
            )

            # STORE
            all_X.append(row)
            all_y.append(hr[i])
            all_activity.append(activity_labels[i])
            all_subjects.append(subject)

        print(f"{subject} done -> {min_len} windows")

    except Exception as e:
        print(f"ERROR in {subject}: {e}")


# ==========================================================
# FINAL ARRAYS
# ==========================================================
X = np.array(all_X, dtype=np.float32)

y = np.array(all_y, dtype=np.float32)

activity = np.array(all_activity)

subjects_array = np.array(all_subjects)


# ==========================================================
# CLEAN NaN / INF
# ==========================================================
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ==========================================================
# STANDARDIZATION
# ==========================================================
scaler = StandardScaler()

X = scaler.fit_transform(X)


# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n======================================")
print("FINAL DATASET SUMMARY")
print("======================================")

print(f"Feature Matrix Shape : {X.shape}")
print(f"Labels Shape         : {y.shape}")
print(f"Activities Shape     : {activity.shape}")
print(f"Subjects Shape       : {subjects_array.shape}")

print(f"Total Features       : {X.shape[1]}")

print(f"Any NaN Left         : {np.isnan(X).any()}")
print(f"Any INF Left         : {np.isinf(X).any()}")

print("======================================")


# ==========================================================
# SAVE FILES
# ==========================================================
np.save("features_upgraded.npy", X)

np.save("labels.npy", y)

np.save("activity.npy", activity)

np.save("subjects.npy", subjects_array)


print("\nSaved Files:")
print("features_upgraded.npy")
print("labels.npy")
print("activity.npy")
print("subjects.npy")