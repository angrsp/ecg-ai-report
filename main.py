"""End-to-end pipeline: load an MIT-BIH record -> analyze it -> generate a
short report with the Claude API.

Usage:
    python main.py                  # record 100, full pipeline incl. API call
    python main.py --record 200     # a different record
    python main.py --dry-run        # skip the Claude API call, print findings only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv  # noqa: E402

from load_signal import load_record  # noqa: E402
from analyze import summarize_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="100", help="MIT-BIH record name (e.g. 100, 200, 203)")
    parser.add_argument("--dry-run", action="store_true", help="skip the Claude API call")
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading record {args.record} from PhysioNet (mitdb)...")
    record = load_record(args.record)

    print("Running signal analysis + classifier...")
    findings = summarize_record(record)

    print("\n--- Findings ---")
    for k, v in findings.items():
        print(f"{k}: {v}")

    if args.dry_run:
        print("\n(--dry-run: skipping the Claude API report step)")
        return

    from report import generate_report  # imported late so --dry-run never needs the API key

    print("\n--- Generating report with Claude ---")
    report_text = generate_report(findings)
    print(report_text)


if __name__ == "__main__":
    main()
