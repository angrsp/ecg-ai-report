"""Classical signal processing + a light ML classifier for ECG beats.

Scope is deliberately small: a band-pass filter, a simple R-peak detector
(compared against the dataset's expert annotations), per-beat feature
extraction, and a scikit-learn classifier for normal vs. abnormal beats.
Not a diagnostic tool — see the disclaimer in report.py.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_signal import EcgRecord

# MIT-BIH annotation symbols, grouped into two classes (AAMI-style simplification).
# Non-beat symbols (rhythm/quality markers) are ignored.
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_SYMBOLS = {"A", "a", "J", "S", "V", "r", "F", "E", "/", "f", "Q"}


def bandpass_filter(x: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Standard ECG band-pass (removes baseline wander and high-freq noise)."""
    nyq = fs / 2
    b, a = sps.butter(N=3, Wn=[low / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def detect_r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    """Simplified Pan-Tompkins: derivative -> square -> moving-average -> peak pick."""
    diff = np.diff(x, prepend=x[0])
    squared = diff ** 2
    window = max(1, int(0.08 * fs))  # ~80 ms integration window
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    min_distance = int(0.25 * fs)  # refractory period: no two beats within 250 ms
    height = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = sps.find_peaks(integrated, height=height, distance=min_distance)
    return peaks


def peak_detection_accuracy(detected: np.ndarray, reference: np.ndarray, fs: int, tol_ms: int = 80) -> float:
    """Fraction of reference (annotated) beats matched by a detected peak within tol_ms."""
    tol = int(tol_ms / 1000 * fs)
    if len(reference) == 0:
        return float("nan")
    matched = 0
    for r in reference:
        if np.any(np.abs(detected - r) <= tol):
            matched += 1
    return matched / len(reference)


@dataclass
class BeatFeatures:
    rr_prev: np.ndarray    # interval to previous beat (s)
    rr_next: np.ndarray    # interval to next beat (s)
    amplitude: np.ndarray  # local signal amplitude at the beat
    labels: np.ndarray     # 0 = normal, 1 = abnormal


def extract_beat_features(record: EcgRecord) -> BeatFeatures:
    """Use the dataset's expert-annotated beat locations (ground truth) and
    build per-beat features from the filtered signal around each one."""
    filtered = bandpass_filter(record.signal, record.fs)
    samples = record.ann_samples
    symbols = record.ann_symbols

    rows, labels = [], []
    for i in range(1, len(samples) - 1):
        sym = symbols[i]
        if sym in NORMAL_SYMBOLS:
            label = 0
        elif sym in ABNORMAL_SYMBOLS:
            label = 1
        else:
            continue  # skip non-beat annotations (rhythm/quality markers)

        rr_prev = (samples[i] - samples[i - 1]) / record.fs
        rr_next = (samples[i + 1] - samples[i]) / record.fs
        window = filtered[max(0, samples[i] - 10): samples[i] + 10]
        amplitude = float(np.ptp(window)) if len(window) else 0.0

        rows.append((rr_prev, rr_next, amplitude))
        labels.append(label)

    arr = np.array(rows)
    return BeatFeatures(
        rr_prev=arr[:, 0], rr_next=arr[:, 1], amplitude=arr[:, 2],
        labels=np.array(labels),
    )


def train_and_evaluate(features: BeatFeatures, seed: int = 42) -> dict:
    X = np.column_stack([features.rr_prev, features.rr_next, features.amplitude])
    y = features.labels

    if len(np.unique(y)) < 2:
        return {"accuracy": None, "note": "only one class present in this record; can't train/evaluate"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "n_beats": len(y),
        "n_abnormal": int(y.sum()),
        "pct_abnormal": float(y.mean() * 100),
    }


def summarize_record(record: EcgRecord) -> dict:
    """End-to-end: filter, detect R-peaks (compared to annotations), extract
    features from annotated beats, train/evaluate the classifier, and return
    a compact dict of findings for the report step."""
    filtered = bandpass_filter(record.signal, record.fs)
    detected_peaks = detect_r_peaks(filtered, record.fs)
    peak_acc = peak_detection_accuracy(detected_peaks, record.ann_samples, record.fs)

    features = extract_beat_features(record)
    clf_result = train_and_evaluate(features)

    mean_rr = float(np.mean(features.rr_prev)) if len(features.rr_prev) else float("nan")
    heart_rate_bpm = 60.0 / mean_rr if mean_rr else float("nan")

    return {
        "record_name": record.record_name,
        "duration_s": len(record.signal) / record.fs,
        "n_detected_peaks": len(detected_peaks),
        "n_annotated_beats": len(record.ann_samples),
        "peak_detection_accuracy": peak_acc,
        "heart_rate_bpm": heart_rate_bpm,
        **clf_result,
    }


if __name__ == "__main__":
    from load_signal import load_record

    rec = load_record("100")
    result = summarize_record(rec)
    for k, v in result.items():
        print(f"{k}: {v}")
