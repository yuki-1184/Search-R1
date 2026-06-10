# Search-R1 NQ Baseline Runbook

This runbook establishes the repository's original Search-R1-style baseline:
PPO token-policy training with a fixed e5 retriever on Natural Questions.

## 1. Prepare the NQ dataset

The repository currently expects:

```text
data/nq_search/train.parquet
data/nq_search/test.parquet
```

Create them with:

```bash
pjsub jobs/prepare_nq_search.sh
```

Check the job and output:

```bash
pjstat
tail -f logs/prepare_nq_search_<JOB_ID>.log
ls -lh data/nq_search/
```

This step downloads `RUC-NLPIR/FlashRAG_datasets` from Hugging Face. The
compute node therefore needs outbound access or a populated Hugging Face cache.

## 2. Confirm model access

The default baseline model is:

```text
meta-llama/Llama-3.2-3B
```

This is the model selected by the upstream `train_ppo.sh`, but it is gated on
Hugging Face. Accept its license and provide `HF_TOKEN`, or override
`BASE_MODEL` with an accessible local path/model.

## 3. Submit the four-GPU pilot

```bash
pjsub jobs/train_ppo_baseline_pilot.sh
```

The pilot uses one B node and four GPUs with:

- 1,024 training examples
- 256 validation examples
- 20 PPO updates (`trainer.total_training_steps=21` in the upstream loop)
- validation before training and every 10 steps
- fixed CPU FAISS retrieval
- seed 13

Monitor it with:

```bash
pjstat
tail -f logs/train_ppo_baseline_pilot_<JOB_ID>.log
```

Run artifacts are written under:

```text
experiments/nq-search-r1-ppo-llama3.2-3b-pilot-seed13/
```

Important files:

- `manifest.txt`: commit, model, data, retriever, seed, and hardware.
- `train.log`: training and aggregate validation metrics.
- `validation/validation_step_*.jsonl`: per-example predictions, EM, F1, and
  basic search statistics.
- `checkpoints/`: saved actor and critic checkpoints.

## 4. Pilot acceptance criteria

The pilot is successful when:

- The retrieval server reaches `/health`.
- Initial validation completes.
- All 20 steps complete without CUDA OOM, Ray failure, or NaNs.
- `val/test_score/nq` and `val/token_f1/nq` are emitted.
- Validation JSONL files and the step-20 checkpoint are present.
- Runtime suggests that a 1,005-step run can fit within the selected PJM
  wall-time.

The pilot score is not a final baseline result.

## 5. Full baseline

After inspecting pilot memory and timing, use the same job with full settings:

```bash
TOTAL_TRAINING_STEPS=1005 \
TRAIN_DATA_NUM=null \
VAL_DATA_NUM=null \
TRAIN_BATCH_SIZE=512 \
VAL_BATCH_SIZE=256 \
SAVE_FREQ=100 \
TEST_FREQ=50 \
EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-full-seed13 \
pjsub jobs/train_ppo_baseline_pilot.sh
```

PJM environment-variable forwarding should be confirmed on Genkai before using
this form. If the submitted environment is not inherited, create a copied job
script with these defaults or use the site's supported `pjsub -x` option.

The current PPO trainer can save checkpoints but does not resume optimizer and
trainer state. Do not start the full run until the pilot gives a realistic
wall-time estimate.

## 6. Reporting

For an initial repository baseline, report:

- Model and checkpoint.
- Dataset and split sizes.
- Seed.
- GPU type/count and wall-clock time.
- Final EM and token F1.
- Average valid searches from the validation JSONL.
- Failed runs or changed settings.

Do not compare this NQ result directly with the paper draft's multi-hop average,
which targets HotpotQA, 2WikiMultiHopQA, and MuSiQue under a different protocol.
