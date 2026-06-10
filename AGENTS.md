# Codex Project Instructions

## Research Mission

- This repository is Search-R1, an RL training framework for language models
  that interleave reasoning with retrieval calls.
- The long-term goal is to develop the method described in `docs/paper/`:
  treating selected passages as stochastic policy actions and jointly
  optimizing token-action and passage-action likelihoods.
- The paper's core novelty is not merely retriever fine-tuning or better query
  generation. It is explicit trajectory-level credit assignment to selected
  passage actions from a fixed-index candidate set.
- The passage encoder and corpus index should remain fixed during the proposed
  RL method. Trainable retrieval components belong on the query side.
- Retrieved text inside `<information>...</information>` is an observation, not
  generated output. Its token log-probability must not be treated as an LLM
  action; the probability of selecting the passage is the retrieval action.

Read these files before making research-design or method-level changes:

- `docs/paper/paper_intent_design_memo.md`: source of truth for the intended
  contribution, comparison logic, and claims.
- `docs/paper/AAAI2027_Ohashi_Japanese.md`: current paper draft and planned
  experimental protocol.

The paper is a research plan and draft. Numerical results written there are
targets or provisional claims until reproduced by artifacts in this repository.
Never present a draft number as an observed result without a traceable run.

## Current Milestone

This branch currently focuses on establishing a reproducible Search-R1
fixed-retriever baseline before implementing Unified Token-Passage RL.

The immediate order of work is:

1. Make the Singularity/PJM environment reliable.
2. Verify retrieval, inference, and short smoke training end to end.
3. Reproduce Search-R1 token-only RL with a fixed dense retriever.
4. Add deterministic evaluation and experiment artifact recording.
5. Measure baseline answer, retrieval, and search-behavior metrics.
6. Only then implement query-side passage-policy learning and passage-action
   log-probabilities.

Do not mix proposed-method behavior into a baseline run. A Search-R1 baseline
optimizes token actions only and treats fixed-retriever results as observations.

## Repository Map

- The training stack is based on veRL. Core training code lives under `verl/`.
- Retrieval server code lives under `search_r1/search/`.
- Dataset preparation scripts live under `scripts/data_process/`.
- Cluster submission and integrated retrieval/training scripts live under
  `jobs/`.
- Research papers and design notes live under `docs/paper/`.
- Experiment outputs should be kept separate from source code and should not be
  committed unless they are small, curated result summaries.

## Execution Environment

- The primary environment is a PJM-managed GPU cluster using Singularity.
- The default project root is
  `/home/pj24003196/ku60000140/Search-R1`.
- The default container is `search_r1.sif`.
- Retrieval data is normally stored in `$HOME/searchr1_data` and includes:
  - `e5_Flat.index`
  - `wiki-18.jsonl`
- Inside the container, the repository is mounted at `/workspace`.
- Do not assume GPU libraries, `pjsub`, cluster modules, models, or retrieval
  data are available in the current shell.

## Current Workflows

- Smoke PPO with an integrated retrieval server:
  `pjsub jobs/train_ppo_smoke_with_retrieval.sh`
- Full PPO with an integrated retrieval server:
  `pjsub jobs/train_ppo_with_retrieval.sh`
- Interactive allocation:
  `bash jobs/pjsub_interact.sh`
- Smoke training entry point:
  `bash train_ppo_smoke.sh`
- Standalone retrieval server:
  `bash run_retriever.sh`
- Inference:
  `bash run_infer.sh`

The smoke workflow creates a small dataset with
`scripts/data_process/make_smoke_dataset.py`, starts
`search_r1/search/retrieval_server.py`, waits for `/health`, and then runs
`train_ppo_smoke.sh`.

Smoke success proves plumbing only. It is not evidence that the baseline has
been reproduced or that a paper metric has been achieved.

## Baseline Protocol

For baseline experiments, explicitly record and hold constant:

- Git commit and dirty-worktree status.
- Dataset name, version, split, preprocessing command, and sample counts.
- Retrieval corpus, index file, passage encoder, retriever model, FAISS mode,
  `topk`, and corpus/index checksums where practical.
- LLM initialization and exact model identifier or local checkpoint.
- Prompt template and special-token protocol.
- Training algorithm, reward definition, hyperparameters, and maximum search
  calls.
- Random seed.
- GPU type/count, wall-clock time, and relevant software/container versions.
- Evaluation checkpoint and deterministic inference settings.

Comparisons must use the same corpus, fixed passage index, prompt, LLM
initialization, search-call budget, dataset split, and evaluation code unless
the difference is the variable under study.

The paper's final protocol plans three seeds: `13`, `21`, and `42`. During
bring-up, one seed or a reduced dataset is acceptable, but label it clearly as
smoke, debug, or pilot rather than a final baseline.

Primary answer metrics:

- Exact Match (EM), using one shared normalization implementation.
- Token-level F1.

Planned retrieval metrics, when annotations or answer matching permit:

- Evidence Recall@k.
- Answer-containing passage rate.
- nDCG@k.
- Full evidence-chain completion for multi-hop datasets.

Planned agentic-behavior metrics:

- Average search calls.
- Useful search rate.
- Redundant search rate.
- Failed search rate.

Do not silently use training reward as evaluation accuracy. Persist generated
answers and retrieval trajectories so metrics can be recomputed offline.

The paper's main target datasets are HotpotQA, 2WikiMultiHopQA, and MuSiQue.
Natural Questions and TriviaQA are secondary open-domain QA evaluations. The
current NQ smoke dataset is for pipeline validation, not the final multi-hop
baseline.

## Experiment Artifacts

Each meaningful run should be identifiable without relying only on a scheduler
log filename. Prefer an output directory or manifest containing:

- Run ID and timestamp.
- Command/config snapshot and environment-variable overrides.
- Git revision and working-tree diff status.
- Seed and hardware information.
- Training and validation metrics.
- Generated answers.
- Per-search query, ranked passage IDs/scores, and search-step index.
- Final checkpoint path and failure status.

Preserve stable passage IDs. Future same-query replay and evidence-chain
analysis require search events and retrieved document identities, not just
aggregate rewards.

## Editing Guidelines

- Preserve existing user changes; the working tree may intentionally be dirty.
- Keep changes focused. Avoid broad refactors of vendored or upstream veRL code
  unless the task requires them.
- Maintain compatibility with Bash and the PJM directives at the top of files
  under `jobs/`.
- Quote shell paths and variables unless word splitting is intentional.
- Keep configurable paths, ports, GPU assignments, and experiment names
  overridable through environment variables.
- Retrieval and training may share a node. Check `CUDA_VISIBLE_DEVICES` and
  `RETRIEVER_CUDA_VISIBLE_DEVICES` carefully to avoid accidental GPU overlap.
- Background services must have cleanup traps and readiness/error checks.
- Never submit a PJM job, launch a long GPU training run, download a model, or
  rebuild a large retrieval index unless the user explicitly asks.
- Keep baseline, proposed-method, evaluation, and analysis code separable.
  Feature flags or distinct configs are preferable to silently changing
  baseline semantics.
- Do not change dataset splits, answer normalization, retrieval depth, search
  budget, or reward definitions merely to improve a reported score.
- Avoid committing personal absolute paths. Use environment variables with
  documented defaults.
- Do not commit generated caches, large checkpoints, raw run logs, datasets, or
  paper PDFs unless explicitly requested.

## Validation

Use the lightest validation that covers the change:

- Shell syntax:
  `bash -n path/to/script.sh`
- Python syntax:
  `python -m compileall path/to/changed/module.py`
- Focused tests, when present:
  `pytest path/to/test_file.py`
- For cluster scripts, inspect the generated command and environment-variable
  flow when cluster execution is unavailable.
- For metric code, add a tiny fixture with known EM/F1/retrieval outcomes.
- For experiment changes, run smoke validation before requesting a full job.

State clearly when GPU, Singularity, retrieval-data, or PJM-dependent behavior
could not be executed locally.

## Research Integrity

- Separate observed results from hypotheses, expected results, and paper-draft
  placeholders.
- Never invent missing metrics, seeds, error bars, significance tests, or
  baseline results.
- Report failed runs and changed conditions; do not compare incompatible runs
  as if they used the same protocol.
- Treat the paper's headline values, including the drafted Search-R1 average EM
  of `51.7`, as unverified until reproduced here.
- When an implementation choice departs from the paper design, document the
  reason and its likely effect on comparability.

## Communication

- User-facing explanations should normally be in Japanese.
- Keep commands and code identifiers in their original English form.
