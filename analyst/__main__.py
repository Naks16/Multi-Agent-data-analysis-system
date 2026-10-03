"""CLI: python -m analyst --csv data/sales.csv --question "Why did sales drop in March?" """

import argparse
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from analyst.graph import run_analysis
from analyst.llm import LLM, LLMJsonError


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent data analyst")
    parser.add_argument("--csv", required=True, help="path to the CSV file")
    parser.add_argument("--question", required=True, help="question about the data")
    args = parser.parse_args()

    if not Path(args.csv).is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    load_dotenv()  # reads ANTHROPIC_API_KEY and MODEL_NAME from .env if present
    try:
        final = run_analysis(args.csv, args.question, LLM())
    except (LLMJsonError, RuntimeError, anthropic.APIError) as e:
        # The SDK's "Connection error." hides the real reason; show it.
        cause = f" (cause: {e.__cause__!r})" if e.__cause__ else ""
        print(f"Analysis failed: {e}{cause}", file=sys.stderr)
        return 1

    run_dir = Path(final["run_dir"])
    ok = sum(r["success"] for r in final["step_results"])
    print(f"Steps succeeded: {ok}/{len(final['step_results'])}")
    print(f"Critic approved: {final['critic_feedback']['approved']} "
          f"(revisions: {final['revision_count']})")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Trace:  {run_dir / 'trace.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
