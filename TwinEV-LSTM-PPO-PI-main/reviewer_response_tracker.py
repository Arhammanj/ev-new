"""Generate and update the reviewer-response tracker.

The tracker is intentionally data-driven so each reviewer point can be updated
without editing the report layout. Run this file after changing a point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "reviewer_response_tracker.json"
MARKDOWN_PATH = ROOT / "reviewer_response_tracker.md"


def point(reviewer: str, number: int, title: str, status: str, response: str = "") -> dict:
    return {
        "id": f"{reviewer}-{number}",
        "reviewer": reviewer,
        "number": number,
        "title": title,
        "status": status,
        "response": response,
        "artifacts": [],
        "reproduction": [],
    }


POINTS = [
    point("R1", 1, "Distinguish the criterion from flexibility, demand response, and prior forecasting/MPC work.", "open"),
    point("R1", 2, "Validate the criterion on additional real-world datasets or independent scenarios.", "addressed", "Caltech and JPL were evaluated with requested-energy complete cases; Office001 was retained as unsupported because requested energy is absent."),
    point("R1", 3, "Quantify how the score changes with transformer capacity.", "partial", "Capacity is configurable and prior capacity-sensitivity infrastructure exists, but a complete score-versus-capacity audit remains to be run."),
    point("R1", 4, "Justify the median/binary-slack formulation and compare continuous or energy-based flexibility.", "open"),
    point("R1", 5, "Compare against safety-filtered RL, constrained optimization, or MPC.", "partial", "MPC and safety-filter infrastructure exists; the complete comparative experiment is not yet reported."),
    point("R1", 6, "Increase seeds and distinguish no significance from no practical benefit.", "partial", "Reproducibility and multi-seed aggregation support were added; expanded independent runs remain pending."),
    point("R1", 7, "Evaluate one-, three-, and six-hour forecasts in the controller.", "open"),
    point("R1", 8, "Discuss charging-rate, SOC, deadlines, and finer control intervals.", "partial", "Real Caltech 15-minute loading and forecast-resolution support were verified; SOC/deadline and rate sensitivity remain pending."),
    point("R1", 9, "Provide joint distributions of availability and slack.", "addressed", "Event-level records and condition-distribution figures now report A, S, their product, and persistent zero-product risk hours."),
    point("R1", 10, "Clarify whether the criterion is a screening metric, necessary condition, or performance predictor.", "partial", "The cross-dataset report frames it as a screening/necessary-condition diagnostic; prospective controller selection is not yet demonstrated."),
    point("R2", 1, "State what information is needed to compute the criterion.", "open"),
    point("R2", 2, "Quantify rapid deployment or grid stress in the introduction.", "open"),
    point("R2", 3, "Explain why forecasting is widely adopted before criticizing its premise.", "open"),
    point("R2", 4, "Link each contribution to the supporting questions.", "open"),
    point("R2", 5, "Signpost the distinction from prediction-assisted and safe-RL studies.", "open"),
    point("R2", 6, "Expand the motivation for the actionability gap.", "open"),
    point("R2", 7, "Describe ACN-Data usage patterns and site context.", "partial", "Dataset characteristics and arrival/departure/energy figures were generated; manuscript prose remains to be updated."),
    point("R2", 8, "Justify hourly resolution.", "addressed", "The README and resolution comparison document hourly versus real 15-minute support."),
    point("R2", 9, "Contrast measured capacity with synthetic environments.", "open"),
    point("R2", 10, "Explain why the selected aggregate target is more forecastable.", "open"),
    point("R2", 11, "Provide the full reward function and learning trade-offs.", "partial", "PPO configuration and evaluation behavior are documented; full manuscript reward exposition remains pending."),
    point("R2", 12, "Explain residual violations under reward penalties.", "partial", "The implementation documents penalty-based evaluation and safety-filter infrastructure; manuscript discussion remains pending."),
    point("R2", 13, "Improve Figure 3 axis label and horizon markers.", "open"),
    point("R2", 14, "Justify median slack instead of mean slack.", "addressed", "The report preserves the median criterion and explains its distributional interpretation."),
    point("R2", 15, "Discuss why capacity findings persist across capacities.", "partial", "Capacity sensitivity infrastructure exists; a complete cross-capacity interpretation remains pending."),
    point("R2", 16, "Give concrete next steps for wider datasets and adaptive horizons.", "partial", "Cross-dataset evaluation and multi-resolution support provide concrete next steps; adaptive horizons remain future work."),
    point("R2", 17, "Change 'controllable demand' to 'controllable EV demand'.", "open"),
    point("R3", 1, "Add learning curves and MPC with BiLSTM and oracle forecasts.", "partial", "Finite-horizon MPC and rollout infrastructure were added; the requested synthetic-site comparison is not yet reported."),
    point("R3", 2, "Test whether the forecast adds information beyond calendar features.", "open"),
    point("R3", 3, "Count lead time from saturated-period onset.", "open"),
    point("R3", 4, "Explain the median and positive-slack threshold; report full distributions.", "addressed", "Event-level statistical diagnostics now report full product distributions and A/S condition frequencies."),
    point("R3", 5, "Run controllers with and without forecast and score JPL/Office001.", "partial", "JPL was scored using requested energy; Office001 is scientifically unsupported for requested-demand scoring; synthetic controller comparison remains pending."),
    point("R3", 6, "Add aggregate draw to the PPO observation.", "open"),
    point("R3", 7, "Increase seeds because the confidence interval spans the capacity-sweep room.", "partial", "Multi-seed aggregation support exists; additional independent runs remain pending."),
    point("R3", 8, "Use a safety layer to isolate forecast effects.", "partial", "Safety-filtered controller support was added; the complete comparison is not yet reported."),
    point("R3", 9, "Reconcile inconsistent p-values and capacity-sweep statements.", "open"),
    point("R3", 10, "Document PPO hyperparameters, deterministic actions, and train/test forecast separation.", "addressed", "PPO settings, sampled versus deterministic evaluation, timestamp alignment, and leakage safeguards are documented."),
    point("R3", 11, "Correct or run the promised one- and three-hour controller forecasts.", "partial", "Forecast generation supports multiple horizons; the controller comparison for all promised horizons remains pending."),
    point("R3", 12, "Restore random-policy and reward-weight-sweep reporting.", "open"),
    point("R3", 13, "Explain the synthetic ramp arrival model and imposed curve.", "partial", "Synthetic-ramp infrastructure was audited; a fuller manuscript explanation remains pending."),
    point("R4", 1, "Justify 50 kW, per-EV power, aggregation, and capacity sensitivity.", "partial", "The configuration is explicit and capacity sensitivity is supported; infrastructure-specific justification and full sweep remain pending."),
    point("R4", 2, "Justify hourly resolution and repeat at 15-minute intervals.", "addressed", "Real Caltech 15-minute feasibility and hourly-versus-15-minute comparison artifacts were generated."),
    point("R4", 3, "Explain variables mathematically and report joint/event-level distributions.", "addressed", "The criterion implementation, event records, statistical diagnostics, and condition figures are available."),
    point("R4", 4, "Justify six-hour aggregate demand and verify no forecast input/target leakage.", "partial", "Input/target leakage was removed and alignment audits were added; peak/violation forecasting targets remain future work."),
    point("R4", 5, "Add rule-based, optimization, or MPC benchmarks.", "partial", "MPC and cap-only infrastructure exists; a complete benchmark table is not yet reported."),
    point("R4", 6, "Increase runs, statistical comparisons, temporal periods, and PPO reproducibility details.", "partial", "Reproducibility audits, configuration capture, and multi-seed support were added; expanded periods/runs remain pending."),
    point("R4", 7, "Validate on an independent real-world dataset.", "addressed", "JPL was evaluated independently; Office001 was transparently excluded from requested-demand scoring because coverage is zero."),
    point("R4", 8, "Reconcile forecasting horizons and capacity-sweep claims.", "partial", "Resolution and forecast documentation were updated; manuscript tables and all controller horizon runs remain pending."),
]

DEFAULT_ARTIFACTS = {
    "R1-2": ["cross_dataset_lead_time\\outputs\\lead_time_scores.csv", "cross_dataset_lead_time\\CROSS_DATASET_REPORT.md"],
    "R1-9": ["cross_dataset_lead_time\\outputs\\lead_time_statistical_diagnostics.csv", "cross_dataset_lead_time\\outputs\\paper_figures\\lead_time_condition_diagnostics.png"],
    "R2-8": ["resolution_comparison.py", "paper_figures\\resolution_comparison.png"],
    "R2-14": ["cross_dataset_lead_time\\CROSS_DATASET_REPORT.md"],
    "R3-4": ["cross_dataset_lead_time\\outputs\\lead_time_statistical_diagnostics.csv", "cross_dataset_lead_time\\outputs\\risk_hour_zero_persistence.csv"],
    "R3-10": ["verify_reproducibility.py", "README.md"],
    "R4-2": ["resolution_comparison.py", "paper_figures\\resolution_comparison.png"],
    "R4-3": ["cross_dataset_lead_time\\outputs\\lead_time_event_records.csv", "cross_dataset_lead_time\\outputs\\paper_figures\\lead_time_condition_diagnostics.png"],
    "R4-7": ["cross_dataset_lead_time\\outputs\\dataset_verification.csv", "cross_dataset_lead_time\\CROSS_DATASET_REPORT.md"],
}


def write_outputs(points: list[dict]) -> None:
    JSON_PATH.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")
    lines = [
        "# Reviewer Response Tracker",
        "",
        "Statuses: `open`, `partial`, `addressed`.",
        "Update a point with `python reviewer_response_tracker.py --id R1-2 "
        "--status addressed --response \"...\" --artifacts path/to/file`.",
        "",
        "| ID | Status | Reviewer point | Response / evidence | Artifacts |",
        "|---|---|---|---|---|",
    ]
    for item in points:
        artifacts = ", ".join(item["artifacts"]) or "-"
        response = item["response"] or "Pending."
        lines.append(
            f"| {item['id']} | {item['status']} | {item['title']} | "
            f"{response} | {artifacts} |"
        )
    lines.extend([
        "",
        "## Reproduction commands",
        "",
        "```powershell",
        "python cross_dataset_lead_time\\cross_dataset_lead_time.py",
        "python -m pytest -q cross_dataset_lead_time\\test_cross_dataset_lead_time.py",
        "python resolution_comparison.py",
        "python verify_reproducibility.py --forecast-csv acn_target_a_bilstm_full.csv --label hourly",
        "```",
    ])
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_points() -> list[dict]:
    if not JSON_PATH.exists():
        return [dict(item) for item in POINTS]
    saved = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    saved_points = saved.get("points")
    if not isinstance(saved_points, list) or len(saved_points) != len(POINTS):
        raise SystemExit(f"Invalid tracker contents: {JSON_PATH}")
    return saved_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Point ID, for example R1-2")
    parser.add_argument("--status", choices=("open", "partial", "addressed"))
    parser.add_argument("--response")
    parser.add_argument("--artifacts", nargs="*", default=None)
    args = parser.parse_args()
    points = load_points()
    for item in points:
        if not item["artifacts"]:
            item["artifacts"] = DEFAULT_ARTIFACTS.get(item["id"], [])
    if args.id:
        matches = [item for item in points if item["id"] == args.id]
        if len(matches) != 1:
            raise SystemExit(f"Unknown reviewer point: {args.id}")
        item = matches[0]
        if args.status:
            item["status"] = args.status
        if args.response is not None:
            item["response"] = args.response
        if args.artifacts is not None:
            item["artifacts"] = args.artifacts
    write_outputs(points)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
