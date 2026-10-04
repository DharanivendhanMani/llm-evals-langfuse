# Test results

Saved outputs from the Arusuvai support evals, kept as a reference. Each file's name says what happened.

| File | What it ran | Result |
|---|---|---|
| `01-deterministic-run-PASSED.txt` | `run_experiment.py --local --no-judge --repeat 1` | Deterministic checks passed in all groups. Judge checks not run. |
| `02-judge-calibration.txt` | `calibrate_judge.py` | 16/16 agreement with AI-written labels (smoke test only). |
| `03-full-run-GATE-CLOSED-88.9pct.txt` | `run_experiment.py --repeat 3` with judge | Gate closed. Unanswerable 88.9% (needs 100%); UNA-03 failed. |
| `04-full-run-GATE-CLOSED-83.3pct.txt` | `run_experiment.py --repeat 3` with judge | Gate closed. Unanswerable 83.3%; UNA-03 and UNA-06 failed. |

The gate has not passed with the judge yet. Runs 03 and 04 are kept to show the failures being worked on.
