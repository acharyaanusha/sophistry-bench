import argparse
import json
from pathlib import Path

from sophistry_bench.eval import compare_leaderboards


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    args = p.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    deltas = compare_leaderboards(before, after)
    print(json.dumps(deltas, indent=2))


if __name__ == "__main__":
    main()
