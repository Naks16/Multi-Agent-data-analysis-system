"""Eval cases and deterministic scoring (no LLM judge).

A case passes only if every check passes:
- keyword checks: at least one of the regexes matches the answer (case-insensitive)
- number checks:  a number within `rel_tol` of the true value appears in the answer.
  The true value is computed from the CSV with pandas, so it is exact.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from eval.datasets import DATA_DIR

PROJECT_DIR = Path(__file__).parent.parent

# Phrasings that mean "no real change". Used for the null case.
NO_EFFECT = [
    r"not (statistically )?significant", r"no (statistically )?significant", r"insignificant",
    r"no (clear|meaningful|real|notable|substantial|detectable) (evidence|change|difference|effect|impact)",
    r"no evidence", r"(did not|didn't|does not|doesn't) (meaningfully |significantly |materially )?change",
    r"(remained|stayed|was|is) (roughly |about |essentially |largely )?(the same|flat|stable|unchanged)",
]


@dataclass
class NumberCheck:
    name: str
    true_value: Callable[[pd.DataFrame], float]
    rel_tol: float = 0.02
    is_rate: bool = False   # also accept the value written as a percentage (0.3 -> 30%)


@dataclass
class Case:
    id: str
    csv: Path
    question: str
    keywords: dict[str, list[str]]          # check name -> regexes (any one must match)
    numbers: list[NumberCheck] = field(default_factory=list)


CASES = [
    Case("sales_drop", PROJECT_DIR / "data" / "sales.csv",
         "Why did sales drop in March?",
         keywords={"names West region": [r"\bwest\b"]}),
    Case("churn_by_plan", DATA_DIR / "churn.csv",
         "What is the churn rate of each subscription plan, and which plan churns the most?",
         keywords={"names Basic plan": [r"\bbasic\b"]},
         numbers=[NumberCheck("Basic churn rate",
                              lambda df: df[df.plan == "Basic"].churned.mean(), is_rate=True)]),
    Case("top_aov", DATA_DIR / "orders.csv",
         "Which product category has the highest average order value, and what is that average?",
         keywords={"names Electronics": [r"\belectronics\b"]},
         numbers=[NumberCheck("Electronics average order value",
                              lambda df: df[df.category == "Electronics"].order_value.mean())]),
    Case("delivery_delay", DATA_DIR / "deliveries.csv",
         "Why were average delivery times higher in October than in September?",
         keywords={"names FastShip": [r"\bfastship\b"]}),
    Case("discount_effect", DATA_DIR / "discounts.csv",
         "Is the discount percentage associated with the number of units sold?",
         keywords={"says positive association": [r"\bpositive", r"more units", r"units.{0,40}(increase|rise|higher)",
                                                 r"(increase|rise|higher).{0,40}units"]}),
    Case("redesign_null", DATA_DIR / "conversions.csv",
         "Did the website redesign launched in week 7 change the conversion rate?",
         keywords={"says no real change": NO_EFFECT}),
]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def numbers_in(text: str) -> list[float]:
    """All numbers in the text, e.g. '$1,234.5' -> 1234.5, '30.2%' -> 30.2."""
    return [float(m.replace(",", "")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def number_matches(text: str, value: float, rel_tol: float, is_rate: bool) -> bool:
    targets = [value, value * 100] if is_rate else [value]
    return any(abs(n - t) <= rel_tol * abs(t) for n in numbers_in(text) for t in targets)


def score(case: Case, answer: str) -> dict[str, bool]:
    """check name -> passed."""
    results = {name: any(re.search(p, answer, re.IGNORECASE) for p in patterns)
               for name, patterns in case.keywords.items()}
    if case.numbers:
        df = pd.read_csv(case.csv)
        for check in case.numbers:
            results[check.name] = number_matches(answer, check.true_value(df), check.rel_tol, check.is_rate)
    return results
