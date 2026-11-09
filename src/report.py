"""Turn the numeric analysis results into a short natural-language report
using the Claude API. This is the one part of the pipeline that needs
ANTHROPIC_API_KEY set (see .env.example).
"""

import os

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")

SYSTEM_PROMPT = (
    "You are drafting a short technical note for an engineer reviewing the output "
    "of a personal, non-clinical ECG signal-processing prototype. You will be given "
    "numeric findings (heart rate, beat counts, a classifier's accuracy on a public "
    "research dataset). Write 3-5 concise sentences in plain English summarising what "
    "the numbers show. Do not invent any number not given to you. "
    "Always state explicitly that this is a personal prototype on a public research "
    "dataset (MIT-BIH Arrhythmia Database), not a medical device, and not clinically "
    "validated — it must never be read as a diagnosis or clinical finding."
)


def build_prompt(findings: dict) -> str:
    lines = [f"{key}: {value}" for key, value in findings.items()]
    return "Findings from the ECG analysis pipeline:\n" + "\n".join(lines)


def generate_report(findings: dict, model: str = DEFAULT_MODEL) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key "
            "(see https://console.anthropic.com/), or run main.py with --dry-run to "
            "skip this step and just see the computed findings."
        )

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(findings)}],
    )
    return message.content[0].text


if __name__ == "__main__":
    demo_findings = {
        "record_name": "100",
        "duration_s": 1805.6,
        "heart_rate_bpm": 75.3,
        "n_annotated_beats": 2273,
        "peak_detection_accuracy": 0.97,
        "accuracy": 0.94,
        "pct_abnormal": 2.1,
    }
    print(generate_report(demo_findings))
