# Search-R1 Singularity 環境構築手順

## 前提条件

- Singularity (SingularityCE 3.x 以降) がインストール済み
- NVIDIA GPU + CUDA ドライバが利用可能
- `--fakeroot` または `sudo` でコンテナビルドが可能

## 1. SIF ファイルのビルド

```bash
cd /path/to/Search-R1
singularity build --fakeroot search_r1.sif search_r1.def
```

ビルドには 30〜60 分程度かかる（flash-attn のソースビルドが大部分）。

ビルド済み SIF を共有する場合は、チームの共有ストレージにコピーして使ってもらう。

## 2. データの準備

検索に必要なファイルを任意のディレクトリに配置する。

```
/path/to/searchr1_data/
  |- e5_Flat.index        # FAISS インデックス
  |- wiki-18.jsonl         # コーパス
```

## 3. 検索サーバーの起動

ターミナル 1 で以下を実行。サーバーが `Waiting for application startup` → `Application startup complete.` と表示されるまで待つ。

```bash
singularity exec --nv \
    --bind /path/to/Search-R1:/workspace \
    --bind /path/to/searchr1_data:/path/to/searchr1_data \
    search_r1.sif \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path /path/to/searchr1_data/e5_Flat.index \
        --corpus_path /path/to/searchr1_data/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --faiss_gpu
```

ポート 8000 が他のプロセスに使われている場合は、環境変数でポートを変更できる。

```bash
RETRIEVAL_PORT=8001 singularity exec --nv \
    --bind /path/to/Search-R1:/workspace \
    --bind /path/to/searchr1_data:/path/to/searchr1_data \
    search_r1.sif \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path /path/to/searchr1_data/e5_Flat.index \
        --corpus_path /path/to/searchr1_data/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --faiss_gpu
```

## 4. 推論の実行

ターミナル 2 で以下を実行。検索サーバーと同じポートを指定する。

```bash
singularity exec --nv \
    --bind /path/to/Search-R1:/workspace \
    search_r1.sif \
    python /workspace/infer.py
```

ポートを変更した場合:

```bash
RETRIEVAL_PORT=8001 singularity exec --nv \
    --bind /path/to/Search-R1:/workspace \
    search_r1.sif \
    python /workspace/infer.py
```

## 5. インタラクティブシェル

デバッグ等でコンテナ内のシェルに入りたい場合:

```bash
singularity shell --nv \
    --bind /path/to/Search-R1:/workspace \
    search_r1.sif
```

## ヘルパースクリプト

リポジトリに同梱の `run_retriever.sh` / `run_infer.sh` を使うとパス指定の手間が省ける。
パスは各自の環境に合わせて編集すること。

```bash
# ターミナル 1
bash run_retriever.sh

# ターミナル 2
bash run_infer.sh
```

## トラブルシューティング

| 症状 | 原因と対処 |
|------|-----------|
| `FATAL: " ": executable file not found` | コマンドの `\` 改行後にスペースが入っている。1 行で実行するかスクリプトを使う |
| `address already in use` | ポート 8000 が使用中。`RETRIEVAL_PORT=8001` で別ポートを使う |
| `ImportError: flash_attn_2_cuda ... undefined symbol` | flash-attn と torch のバージョン不一致。SIF を再ビルドする |
| ホスト側の Python パッケージが混入する | `search_r1.def` の `PYTHONNOUSERSITE=1` が設定されているか確認 |
| `JSONDecodeError` (infer.py) | 検索サーバーが起動していない、またはポートが一致していない |

## コンテナの主要パッケージ

| パッケージ | バージョン | 備考 |
|-----------|-----------|------|
| Python | 3.10 | Miniforge (conda-forge) |
| CUDA | 12.1.1 | ベースイメージ |
| vllm | 0.6.3 | torch は vllm が自動で導入 |
| flash-attn | (ソースビルド) | torch バージョンに合わせてビルド |
| faiss-gpu | faiss-gpu-cu12 | pip wheel (ビルド不要) |
| transformers | < 4.48 | |
| numpy | < 2 | torch 互換性のため |
