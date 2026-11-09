# ECG signal analysis + AI report (personal prototype)

A small, honest prototype built to actually learn the basics of applied ML and the Claude API on a
biomedical signal — not a polished product, and **not a medical device**.

## What it does

1. Streams a record from the public **MIT-BIH Arrhythmia Database** (PhysioNet) via `wfdb` — no
   manual download needed.
2. Classical signal processing: band-pass filter, a simplified Pan-Tompkins R-peak detector, checked
   against the dataset's expert annotations.
3. Per-beat feature extraction (RR intervals, local amplitude) and a small `scikit-learn`
   (`RandomForestClassifier`) model distinguishing normal vs. abnormal beats, using the dataset's
   expert labels as ground truth.
4. The numeric findings are sent to the **Claude API** with a prompt that turns them into a short,
   plain-English technical note.

## Why

Built while applying to an R&D role that combines biomedical software with applied AI (ML/DL/GenAI,
LLM APIs). Rather than claim skills I don't have on a CV, I built something small enough to fully
understand and explain — picking up again on a neural-networks-for-ECG thesis topic from my
bachelor's degree, now adding a real LLM-API integration on top.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY
```

An API key is only needed for the report step — get one at <https://console.anthropic.com/>.

## Run

```bash
python main.py                  # record 100, full pipeline incl. the Claude API report
python main.py --record 203     # a noisier record with more arrhythmias
python main.py --dry-run        # skip the Claude API call, just print the numeric findings
```

## Example output (record 203, noisier than 100 — more realistic numbers)

```
record_name: 203
duration_s: 1805.6
n_detected_peaks: 2908
n_annotated_beats: 3108
peak_detection_accuracy: 0.930
heart_rate_bpm: 101.4
accuracy: 0.898
n_beats: 2979
n_abnormal: 451
pct_abnormal: 15.1
```

(Record 100 is much cleaner and gives a near-100% classifier accuracy — worth running both to see
how noise and arrhythmia variety change the results.)

## What this is not

- Not clinically validated, not a diagnostic tool, not CE-marked or regulated software.
- The classifier uses 3 simple features (RR intervals + local amplitude) — a real system would use
  far more signal morphology and almost certainly a different architecture.
- Built to learn and to have something true to discuss, not as a production pitch.

## Data & license

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), PhysioNet, ODC-By 1.0.
Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50, 2001.
