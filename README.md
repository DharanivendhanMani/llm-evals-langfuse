# llm-evals-langfuse

A customer-support agent for a restaurant, plus an evaluation suite that measures how well it answers, and a pass/fail quality gate. Every conversation and every eval run is traced in [Langfuse](https://langfuse.com).

The assistant answers questions for **Arusuvai Kitchen**, a South Indian restaurant in Mississauga, Canada. It looks up hours, menu prices, delivery fees and contact details with tools, and it is tested on the cases that matter most for a support bot: does it stay correct, does it admit what it does not know, and does it resist manipulation.

This is a learning project for LLM evals and Langfuse.

## What it demonstrates

- **A tool-using agent** on Groq (OpenAI-compatible API) with three tools, traced end to end in Langfuse.
- **A golden dataset** of 25 cases in three groups, stored as JSON and synced to a Langfuse dataset.
- **Deterministic evaluators** (facts present, right tools called) and an **LLM judge** used only where a string match cannot express the requirement.
- **Judge calibration**, so the judge is checked against labelled samples before its scores are trusted.
- **Repeated runs and a quality gate** with per-category thresholds and CI-friendly exit codes.
- **Prompt versioning**, so two prompts can be compared on the same dataset in the Langfuse UI.

## How it works

```
customer message
      |
      v
app.py  --- agent loop (up to 8 steps) --->  tools
      |                                        get_opening_hours   (hours, open now, holidays)
      |                                        search_menu          (fuzzy dish / category search)
      v                                        get_restaurant_info (contact, services, catering, delivery)
reply + tools_called
      |
      v
evals/evaluators.py  --->  scores per test case  --->  scripts/run_experiment.py  --->  quality gate
```

One Langfuse trace is created per chat turn:

```
handle-support-message          (agent)
├── generate-response           (generation: messages, reasoning, tool calls, token usage)
├── get-opening-hours           (tool)
├── generate-response           (generation)
└── search-menu                 (tool)
```

Email addresses are redacted from all exported spans before they leave the process.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | The assistant: agent loop, tools, Langfuse tracing. `answer()` is the entry point the evals call. Run it directly for a terminal chat. |
| `restaurant_data.py` | Single source of truth for restaurant facts. Its header lists what is deliberately left out. |
| `prompts/` | Versioned system prompts. `support-v1.txt` is minimal. `support-v2.txt` adds grounding rules and injection resistance. |
| `dataset/arusuvai_v0.json` | The golden dataset: 14 answerable, 6 unanswerable, 5 adversarial cases. |
| `dataset/judge_calibration.json` | Labelled sample replies used to check the LLM judge. |
| `evals/evaluators.py` | The evaluators and the per-category pass-rate scorer. |
| `scripts/seed_dataset.py` | Validates the dataset and uploads it to Langfuse. |
| `scripts/calibrate_judge.py` | Measures judge agreement with the labelled samples. |
| `scripts/run_experiment.py` | Runs the dataset, repeats each case, scores it and applies the gate. |

## Test groups

| Group | Cases | What it tests |
|---|---|---|
| **answerable** | 14 (`ANS-01` to `ANS-14`) | Questions the data can answer: hours, menu prices, delivery fees, catering, contact. |
| **unanswerable** | 6 (`UNA-01` to `UNA-06`) | Questions the data cannot answer: allergens, halal status, reservations, payment methods, delivery minimum and time, delivery beyond 30 miles. The assistant must say it does not know and give the phone number, never invent an answer. |
| **adversarial** | 5 (`ADV-01` to `ADV-05`) | Manipulation: prompt extraction, a fake discount from "the owner", instructions hidden in pasted text, a wrong price stated as fact, a persona switch. |

Each dataset item stores the input, the expected behaviour, and machine-checkable rules (`must_include`, `must_not_include`, `tools_required`, `must_not_claim`). Some inputs pin the clock (`now`), so "are you open right now?" has one right answer.

## Metrics

Every score is boolean (pass or fail).

| Score | Type | What it checks |
|---|---|---|
| `facts_correct` | Deterministic | Required facts are present and forbidden ones are absent. Formatting is normalised, so "9 AM", "9:00 a.m." and "9am" all match. |
| `tool_use_correct` | Deterministic | The right tools were called and forbidden ones were not. |
| `answer_matches_reference` | LLM judge | The reply agrees with the reference answer and invents nothing. |
| `honest_handoff` | LLM judge | For unanswerable questions: says it does not know and points to the restaurant. Both a confident "yes" and a confident "no" fail. |
| `injection_resistance` | LLM judge | Does not obey the manipulation, makes no forbidden claims, and does not leak instructions or tool names. |
| `passed` | Composite | `true` only if every check that applied to the item passed. The gate reads this. |
| `judge_error` | Diagnostic | The judge itself failed. This is "no verdict", not a model failure. |
| `pass_rate/<group>` | Run-level | Pass rate per group, written onto each Langfuse dataset run. |

The judge model (default `qwen/qwen3.8-27b`) is a different family from the assistant (`openai/gpt-oss-120b`) to limit self-preference bias.

## Quality gate

Pass rates are computed over **trials** (test case x repeat), so one flaky answerable trial does not close the gate, but a single successful injection does.

| Group | Required pass rate |
|---|---|
| answerable | 95% |
| unanswerable | 100% |
| adversarial | 100% |

Thresholds are a product decision. Edit `THRESHOLDS` in `scripts/run_experiment.py`.

| Exit code | Meaning |
|---|---|
| 0 | Gate open |
| 1 | Gate closed (a group is below its threshold) |
| 2 | No verdict (judge or task errors), never mistaken for a model failure |

## Setup

Requires Python 3.13 (other recent versions should work), a [Langfuse](https://cloud.langfuse.com) project and a [Groq](https://console.groq.com) API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
```

`.env` settings:

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Langfuse project credentials (project Settings, API Keys) |
| `GROQ_API_KEY` | Model access for the assistant and the judge |
| `PROMPT_VERSION` | Which `prompts/support-<version>.txt` to use (default `v2`) |
| `JUDGE_MODEL` | Judge model (default `qwen/qwen3.8-27b`) |

## Usage

```bash
# 1. Validate the dataset (offline, writes nothing)
python scripts/seed_dataset.py

# 2. Chat with the assistant (traces go to Langfuse)
python app.py

# 3. Run the evals with deterministic checks only (no judge calls, fastest)
python scripts/run_experiment.py --local --no-judge --repeat 1

# 4. Upload the dataset to Langfuse
python scripts/seed_dataset.py --apply

# 5. Check the judge against labelled samples
python scripts/calibrate_judge.py

# 6. Full run: deterministic checks + LLM judge + gate
python scripts/run_experiment.py --repeat 3

# 7. Compare prompts on the same dataset
python scripts/run_experiment.py --prompt-version v1 --repeat 3
```

`--no-judge` is for learning and quick iteration. Cases that rely only on the judge are not scored in that mode, so it never prints "GATE OPEN".

In Langfuse, look at **Tracing** for individual conversations, **Datasets** for the test cases, and each dataset's **Runs** tab to compare experiments side by side.

## Limitations

- **Expected outputs are AI-authored** and flagged `needs_human_review`. Treat scores as a smoke test until you have reviewed the dataset and filled in `human_label` in the calibration file.
- **Single-turn only.** The evals do not cover multi-turn conversations.
- **Not measured:** latency and cost.
- **Menu data is a snapshot** (2026-09-25) taken from the restaurant's website and summarised by a model, so prices should be re-verified against the live site.
- **Delivery distance** is judged from what the customer says. The assistant has no way to compute a distance from an address.
- The restaurant is a real business. Check that publishing its details is acceptable before making this repository public.

## Acknowledgements

The evaluation design follows the public **llm-quality-gate** project by Jagadeesh Jayachandran (MIT License): a golden dataset split into answerable, unanswerable and adversarial cases, deterministic assertions before an LLM judge, a judge calibrated on hand-labelled samples, repeated runs, and per-group pass-rate thresholds (95% / 100% / 100%) that fail the build.

This project is an independent implementation with different tooling and domain: Python and Langfuse instead of Node and promptfoo, a tool-calling agent instead of retrieval, and a restaurant instead of an online store. No code was copied from it.
