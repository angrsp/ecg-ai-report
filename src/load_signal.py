"""Load a record from the MIT-BIH Arrhythmia Database (PhysioNet) via wfdb.

Streams directly from PhysioNet's PN_DIR (no manual download needed) and returns
the signal, its sampling rate, and the reference beat annotations.
"""

from dataclasses import dataclass

import numpy as np
import wfdb

MITDB = "mitdb"


@dataclass
class EcgRecord:
    record_name: str
    signal: np.ndarray       # 1D array, first channel (MLII lead in most mitdb records)
    fs: int                  # sampling frequency in Hz
    ann_samples: np.ndarray  # sample index of each annotated beat
    ann_symbols: list        # annotation symbol per beat ('N' = normal, others = abnormal)


def load_record(record_name: str = "100") -> EcgRecord:
    """Load one MIT-BIH record by name (e.g. '100', '200', '203')."""
    record = wfdb.rdrecord(record_name, pn_dir=MITDB)
    ann = wfdb.rdann(record_name, "atr", pn_dir=MITDB)

    signal = record.p_signal[:, 0]  # first channel
    fs = record.fs

    return EcgRecord(
        record_name=record_name,
        signal=signal,
        fs=fs,
        ann_samples=np.asarray(ann.sample),
        ann_symbols=list(ann.symbol),
    )


if __name__ == "__main__":
    rec = load_record("100")
    print(f"Record {rec.record_name}: {len(rec.signal)} samples at {rec.fs} Hz "
          f"({len(rec.signal) / rec.fs:.1f} s), {len(rec.ann_samples)} annotated beats")
