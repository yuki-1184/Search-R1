# 九大スパコン「玄界」GPU利用チートシート

> 目的：サーバー運用を深掘りせずに、LLM/RAG系の実装・実験を玄界のGPUで回すための最低限メモ。  
> 想定：Mac/Linux から SSH、Python/PyTorch、GPUジョブ、バッチジョブ中心。

---

## 0. まず覚えること

玄界では、**ログインノードで重い計算をしない**。  
GPUを使う処理、長時間処理、学習・推論・大量前処理は、必ず **ジョブ** として計算ノードに投げる。

基本の流れはこれ。

```text
自分のPC
  ↓ ssh/scp
ログインノード
  ↓ pjsub
GPU計算ノード
  ↓ 結果を /home or /fast に保存
ログインノード/自分のPCで確認
```

よく使うコマンドだけ先にまとめる。

```bash
# ログイン
ssh <username>@genkai.hpc.kyushu-u.ac.jp

# ファイル転送: local -> 玄界
scp -r ./my_project <username>@genkai.hpc.kyushu-u.ac.jp:/home/<group>/<username>/

# ファイル転送: 玄界 -> local
scp -r <username>@genkai.hpc.kyushu-u.ac.jp:/home/<group>/<username>/results ./results

# モジュール確認
module avail
module list
show_module

# GPUインタラクティブジョブ
pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00

# バッチジョブ投入
pjsub train.sh

# ジョブ確認
pjstat
pjstat -l
pjstat2

# ジョブ削除
pjdel <JOB_ID>

# 混雑状況
pjshowrsc --rscgrp b-batch
pjshowrsc --rscgrp b-inter

# quota確認
show_quota
```

---

## 1. 玄界のGPUノードざっくり理解

LLM/RAG系で主に使うのは **ノードグループB/C**。

| ノードグループ | 用途 | GPU | メモリ | 備考 |
|---|---|---:|---:|---|
| B | 通常のGPU実験 | NVIDIA H100 x 4 / node | 約1TB RAM | まずはここで十分 |
| C | 大きめのGPU実験 | NVIDIA H100 x 8 / node | 約8TB RAM | 大規模・単一ノード大メモリ向け |

基本は **Bを使う**。Cはかなり大きい実験や、Bではメモリ・GPU枚数が足りないとき。

---

## 2. ログイン

```bash
ssh <username>@genkai.hpc.kyushu-u.ac.jp
```

初回は `Are you sure you want to continue connecting?` と出るので `yes`。

SSH鍵を使う場合、`~/.ssh/config` に書いておくと楽。

```sshconfig
Host genkai
    HostName genkai.hpc.kyushu-u.ac.jp
    User <username>
    IdentityFile ~/.ssh/id_rsa
```

以後はこれだけでログインできる。

```bash
ssh genkai
```

---

## 3. ディレクトリ構成

玄界では主に `/home` と `/fast` を使う。

| 場所 | 用途 | メモ |
|---|---|---|
| `/home/<group>/<username>` | 普段の作業、コード、軽めのデータ | 基本ここ |
| `/home/<group>/share` | グループ共有 | 共通データなど |
| `/fast/<group>` | 高速ストレージ | 使える場合のみ。大きいデータ・I/O多い実験向け |
| `$GENKAI_FAST_DIR` | fast領域の環境変数 | 使える場合 |
| `$GENKAI_CACHE_DIR` | Hot Nodes用キャッシュ | 大規模データを何度も読む場合 |

容量確認：

```bash
show_quota

# Lustreのグループquotaを直接確認
lfs quota -h -g <group> /home/<group>
```

`/home/<group>/share` はグループ共有領域で、グループquotaを消費する。
Singularityのビルドキャッシュや大きな共有データは、個人ディレクトリではなく
`/home/<group>/share` に置くと管理しやすい。

おすすめ構成：

```bash
/home/<group>/<username>/
├── projects/
│   └── agentic-rag/
├── datasets/
├── models/
├── logs/
└── results/
```

---

## 4. ファイル転送

### ローカルから玄界へ

```bash
scp -r ./agentic-rag genkai:/home/<group>/<username>/projects/
```

または host設定していない場合：

```bash
scp -r ./agentic-rag <username>@genkai.hpc.kyushu-u.ac.jp:/home/<group>/<username>/projects/
```

### 玄界からローカルへ

```bash
scp -r genkai:/home/<group>/<username>/projects/agentic-rag/results ./results
```

### 大きいファイルは `rsync` 推奨

```bash
rsync -avP ./agentic-rag/ genkai:/home/<group>/<username>/projects/agentic-rag/
```

`rsync` は途中で切れても再開しやすい。

---

## 5. module の使い方

玄界では CUDA, PyTorch などは `module` で読み込む。

```bash
module avail        # 使えるmodule一覧
module list         # 今読み込んでいるmodule
module load cuda    # CUDAを読み込む
module unload cuda  # CUDAを外す
module purge        # すべて外す
show_module         # 利用可能なアプリ/ライブラリ確認
```

PyTorchを使う例：

```bash
module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2
python3.11 -c "import torch; print(torch.cuda.is_available())"
```

注意：ログインノードで `module load` しても、**ジョブ内に自動では引き継がれないことがある**。  
基本的に、**ジョブスクリプトの中でも module load を書く**。

---

## 6. Python環境

### 6.1 公式moduleのPyTorchを使う場合

一番楽。

```bash
module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2
python3.11 train.py
```

### 6.2 venvを作る場合

インタラクティブGPUジョブに入ってから作るのが安全。

```bash
pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00

python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

以後：

```bash
source .venv/bin/activate
python train.py
```

### 6.3 conda/minicondaについて

使えるなら便利だが、HPCでは環境の衝突や容量問題が起きやすい。  
急ぎならまずは `module + venv` の方が無難。

---

## 7. インタラクティブGPUジョブ

デバッグ、環境構築、短い動作確認に使う。

### Singularityを使う場合

Singularityコンテナを使うジョブでは `jobenv=singularity` を必ず指定する。
指定しない場合、次のエラーになることがある。

```text
Failed to create mount namespace: mount namespace requires privileges
```

1ノードを確保して、そのノードのGPUを使う例：

```bash
pjsub \
  --interact \
  -L rscgrp=b-inter \
  -L node=1 \
  -L elapse=04:00:00 \
  -L jobenv=singularity \
  --sparam wait-time=3000
```

Search-R1リポジトリでは、同等のラッパーを使える。

```bash
bash jobs/pjsub_interact.sh -r b-inter -t 04:00:00 -n 1
```

### Bノード GPU 1枚

```bash
pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00
```

入れたら：

```bash
hostname
nvidia-smi
module load cuda/12.2.2
module load pytorch-cuda/2.3.1-12.2.2
python3.11 -c "import torch; print(torch.cuda.is_available())"
```

終了：

```bash
exit
```

### Cノード GPU 1枚

```bash
pjsub --interact -L rscgrp=c-inter,gpu=1,elapse=1:00:00
```

Cは大きいので、基本はBで足りないときだけ。

---

## 8. バッチジョブ基本

長い実験は `train.sh` を書いて `pjsub train.sh` する。

### 8.1 最小PyTorch GPUジョブ

`train.sh`:

```bash
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -j
#PJM -N agentic_rag_test

module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2

cd /home/<group>/<username>/projects/agentic-rag
python3.11 train.py
```

投入：

```bash
pjsub train.sh
```

確認：

```bash
pjstat
pjstat -l
```

ログは通常、ジョブ投入ディレクトリに出る。

```bash
ls -lh *.out *.err
cat agentic_rag_test.<JOB_ID>.out
```

---

## 9. よく使うジョブ設定

### Bノード GPU 1枚

```bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=1:00:00
```

### Bノード GPU 2枚

```bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=3:00:00
```

### Cノード GPU 1枚

```bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=1:00:00
```

### Cノード GPU 2枚

```bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=2
#PJM -L elapse=3:00:00
```

### エラー出力を標準出力にまとめる

```bash
#PJM -j
```

### ジョブ名をつける

```bash
#PJM -N my_experiment
```

---

## 10. 複数GPUを使う場合

PyTorch DDP / torchrun を使うなら、まずは単一ノード複数GPUから。

`train_multi_gpu.sh`:

```bash
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=2
#PJM -L elapse=3:00:00
#PJM -j
#PJM -N ddp_test

module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2

cd /home/<group>/<username>/projects/agentic-rag

nvidia-smi
python3.11 -m torch.distributed.run \
  --nproc_per_node=2 \
  train.py
```

まずはGPU 1枚で動作確認してから、2枚以上に増やす。

---

## 11. Hugging Face / wandb / cache設定

Hugging Faceやdatasetsのcacheがホーム直下に散らばると管理しづらいので、プロジェクトごとに指定する。

```bash
export HF_HOME=/home/<group>/<username>/models/huggingface
export TRANSFORMERS_CACHE=/home/<group>/<username>/models/huggingface/transformers
export HF_DATASETS_CACHE=/home/<group>/<username>/datasets/huggingface
export WANDB_DIR=/home/<group>/<username>/logs/wandb
```

ジョブスクリプトにも書く：

```bash
export HF_HOME=/home/<group>/<username>/models/huggingface
export HF_DATASETS_CACHE=/home/<group>/<username>/datasets/huggingface
export WANDB_DIR=/home/<group>/<username>/logs/wandb
```

wandbを使わない場合：

```bash
export WANDB_MODE=disabled
```

オンラインログが必要な場合：

```bash
wandb login
```

---

## 12. Git運用

ログインノード上で：

```bash
cd /home/<group>/<username>/projects

git clone <repo-url>
cd <repo>
```

private repoならSSH鍵設定が必要。面倒なら一時的にローカルから `rsync` でもよい。

```bash
rsync -avP --exclude .git --exclude .venv ./agentic-rag/ genkai:/home/<group>/<username>/projects/agentic-rag/
```

---

## 13. Jupyterを使いたい場合

玄界には Open OnDemand がある。ブラウザから使う場合は：

```text
genkai-ood.hpc.kyushu-u.ac.jp
```

ただし、最初はJupyterより **SSH + VS Code Remote SSH + バッチジョブ** の方が事故が少ない。

VS Code Remote SSHでログインノードに入って、実行だけジョブに投げるのが楽。

---

## 14. nvidia-smi が見えないとき

ログインノードではGPUが見えないことがある。これは正常。

GPU確認はインタラクティブジョブ内、またはバッチジョブ内で行う。

```bash
pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=30:00
nvidia-smi
```

バッチジョブ内に入れる：

```bash
nvidia-smi
python3.11 -c "import torch; print(torch.cuda.is_available())"
```

---

## 15. よくある事故と対処

### ログインノードで重い処理を走らせてしまう

ダメ。学習・推論・大量前処理はジョブで投げる。

### `torch.cuda.is_available()` が False

確認順：

```bash
nvidia-smi
module list
which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

ありがちな原因：

- GPUノードに入っていない
- CUDA/PyTorch moduleを読み込んでいない
- venv/condaのtorchとmodule CUDAが合っていない
- CPU版PyTorchを入れてしまった

### `No space left on device`

```bash
show_quota

# cache確認
du -sh ~/.cache

du -sh /home/<group>/<username>/*
```

Hugging Face cacheを明示する。

```bash
export HF_HOME=/home/<group>/<username>/models/huggingface
```

### Singularityビルドで `Disk quota exceeded`

SingularityはSIF本体以外に、大きなキャッシュと一時ファイルを作る。
作業領域をグループ共有領域へ明示する。

```bash
mkdir -p /home/<group>/share/singularity-cache
mkdir -p /home/<group>/share/singularity-tmp

export SINGULARITY_CACHEDIR=/home/<group>/share/singularity-cache
export SINGULARITY_TMPDIR=/home/<group>/share/singularity-tmp
export TMPDIR=/home/<group>/share/singularity-tmp
```

### SingularityでGPUまたはmount namespaceが使えない

```text
WARNING: Could not find any nv files on this host!
Failed to create mount namespace: mount namespace requires privileges
```

確認事項：

- `a-batch` / `a-inter` ではなく、GPU用の `b-batch` / `b-inter` を使う
- PJM指定に `-L jobenv=singularity` を追加する
- コンテナ実行時に `singularity exec --nv` を使う

Singularity用バッチジョブの最小ヘッダ：

```bash
#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j
```

### Rayで `AF_UNIX path length cannot exceed 107 bytes`

PJMが設定する一時ディレクトリのパスが長く、Rayのsocketパス上限を超えている。
Ray用の短いパスを指定する。

```bash
export RAY_TMPDIR=/tmp/ray_${PJM_JOBID}
export TMPDIR=$RAY_TMPDIR
mkdir -p "$RAY_TMPDIR"
```

### Search-R1のSIFパス

このリポジトリでは、SIFを次の場所に置く。

```text
/home/pj24003196/ku60000140/Search-R1/search_r1.sif
```

`/home/pj24003196/ku60000140/search_r1.sif` は旧パスであり、使用しない。
PJMジョブは `$PROJ/search_r1.sif` を参照する。

### ジョブがなかなか始まらない

混雑確認：

```bash
pjshowrsc --rscgrp b-batch
pjshowrsc --rscgrp b-inter
pjstat
```

対処：

- `elapse` を短めにする
- GPU枚数を減らす
- まずは `b-inter` で短時間デバッグ
- 長い本実験は `b-batch` に投げる

### ジョブを止めたい

```bash
pjdel <JOB_ID>
```

---

## 16. Agentic RAG実装向けおすすめ運用

### 最初の段階

まずはGPU 1枚で、Search-R1や既存repoを動かす。

```bash
pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00
```

確認するもの：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

### 小さい実験

- small model
- small dataset
- 10〜100 steps
- GPU 1枚
- `b-inter` or 短めの `b-batch`

### 本実験

- バッチジョブ化
- stdout/stderr保存
- config保存
- seed固定
- checkpoint保存
- wandb or jsonl log

おすすめディレクトリ：

```bash
projects/agentic-rag/
├── scripts/
│   ├── train_debug.sh
│   ├── train_b1g1.sh
│   └── train_b2g2.sh
├── configs/
├── outputs/
├── logs/
└── checkpoints/
```

---

## 17. そのまま使えるテンプレ

### 17.1 GPU 1枚デバッグ用

`debug_gpu.sh`:

```bash
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N debug_gpu

module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2

cd /home/<group>/<username>/projects/agentic-rag

nvidia-smi
python3.11 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY
```

投入：

```bash
pjsub debug_gpu.sh
```

### 17.2 学習ジョブ用

`train_b1g1.sh`:

```bash
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM -j
#PJM -N agentic_rag_train

module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2

export HF_HOME=/home/<group>/<username>/models/huggingface
export HF_DATASETS_CACHE=/home/<group>/<username>/datasets/huggingface
export WANDB_DIR=/home/<group>/<username>/logs/wandb

cd /home/<group>/<username>/projects/agentic-rag

mkdir -p logs outputs checkpoints

nvidia-smi
python3.11 train.py \
  --config configs/debug.yaml \
  2>&1 | tee logs/train_${PJM_JOBID}.log
```

---

## 18. 最低限の習慣

実験前：

```bash
pwd
show_quota
module list
pjshowrsc --rscgrp b-batch
```

ジョブスクリプトには必ず入れる：

```bash
#PJM -j
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

実験後：

```bash
pjstat
ls -lh *.out *.err logs/
tail -n 100 logs/train_<JOB_ID>.log
```

---

## 19. 判断基準

| やりたいこと | 使うもの |
|---|---|
| 環境構築・短い確認 | `b-inter` |
| 1時間以上の学習 | `b-batch` |
| GPU 1枚で十分 | `gpu=1` |
| DDP確認 | `gpu=2` |
| かなり大きいモデル/データ | `c-batch` 検討 |
| 大量ファイルを何度も読む | `/fast` or `$GENKAI_CACHE_DIR` 検討 |

---

## 20. 最短実行手順

初回はこれだけやればよい。

```bash
ssh genkai
cd /home/<group>/<username>/projects
# repoを置く
cd agentic-rag

cat > debug_gpu.sh <<'SH'
#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=0:30:00
#PJM -j
#PJM -N debug_gpu

module load cuda/12.2.2
module load cudnn/8.9.7
module load nccl/2.22.3
module load pytorch-cuda/2.3.1-12.2.2

nvidia-smi
python3.11 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')
PY
SH

pjsub debug_gpu.sh
pjstat
```

ログ確認：

```bash
ls -lh *.out *.err
cat debug_gpu.*.out
```
