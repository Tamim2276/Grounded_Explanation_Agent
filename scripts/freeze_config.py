# Day 7.6: freeze the settings for the test run. Writes eval/frozen_config.json -- every
# setting that changes an answer, the prompt version, the model files and the question
# splits -- and scripts/run_eval.py refuses to run the test split if anything differs.
#
#   python scripts/freeze_config.py            # then: bash scripts/checkpoint.sh "freeze"
#
# Refuses to overwrite an existing freeze unless --again (a re-freeze means re-running ALL
# systems on the test set, and saying so in the thesis).
import argparse
import sys

from gea.runs import FROZEN, freeze
from gea.safeio import atomic_write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--again", action="store_true", help="replace an existing freeze (re-run every system)")
    args = ap.parse_args()
    if FROZEN.exists() and not args.again:
        print(f"{FROZEN.name} exists already; a re-freeze needs --again (and every system re-run on test)")
        return 1
    frozen = freeze()
    atomic_write_json(FROZEN, frozen)
    s = frozen["settings"]
    print(f"frozen at {frozen['frozen_at']} (commit {frozen['git_commit']}): {len(s)} settings, prompt "
          f"{s['PROMPT_VERSION']}, model files {list(s['model_files'])}, splits {s['splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
