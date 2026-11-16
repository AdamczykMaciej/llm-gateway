"""Zero-cost AI quality evaluation toolkit (pure stdlib, no model calls by default).

The method: deterministic checks decide pass/fail first; a blind, shuffled judge
(for example a subagent, which costs nothing extra) scores the rest; a private key file maps
anonymous ids back to cases; the aggregator reports per-case, per-tag and A/B numbers;
human labels calibrate the judge. Install with ``pip install llm-gateway[evals]``
(the extra adds no dependencies; it marks the opt-in).
"""

from .calibration import (
    HumanLabel,
    agreement,
    cohen_kappa,
    judge_vs_human,
    load_labels,
    save_labels,
    verdicts_from_judgments,
)
from .checks import (
    CheckResult,
    CheckSpec,
    check_case,
    get_check,
    register_check,
    registered_checks,
    set_language_detector,
)
from .judge import (
    Dimension,
    Judgment,
    PacketSet,
    Rubric,
    build_packets,
    collect_judgments,
    default_rubric,
)
from .runlog import RunLog
from .runner import run_cases
from .schema import Case, Sample, Turn, dump_cases, dump_samples, load_cases, load_samples
from .stats import CaseStats, Comparison, aggregate, compare

__all__ = [
    "Case",
    "CaseStats",
    "CheckResult",
    "CheckSpec",
    "Comparison",
    "Dimension",
    "HumanLabel",
    "Judgment",
    "PacketSet",
    "Rubric",
    "RunLog",
    "Sample",
    "Turn",
    "agreement",
    "aggregate",
    "build_packets",
    "check_case",
    "cohen_kappa",
    "collect_judgments",
    "compare",
    "default_rubric",
    "dump_cases",
    "dump_samples",
    "get_check",
    "judge_vs_human",
    "load_cases",
    "load_labels",
    "load_samples",
    "register_check",
    "registered_checks",
    "run_cases",
    "save_labels",
    "set_language_detector",
    "verdicts_from_judgments",
]
