# Codex Project Instructions

## Project Overview

- This repository is Search-R1, an RL training framework for language models
  that interleave reasoning with retrieval calls.
- The training stack is based on veRL. Core training code lives under `verl/`.
- Retrieval server code lives under `search_r1/search/`.
- Dataset preparation scripts live under `scripts/data_process/`.
- Cluster submission and integrated retrieval/training scripts live under
  `jobs/`.

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

## Important Workflows

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

State clearly when GPU, Singularity, retrieval-data, or PJM-dependent behavior
could not be executed locally.

## Communication

- User-facing explanations should normally be in Japanese.
- Keep commands and code identifiers in their original English form.
