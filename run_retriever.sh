#!/bin/bash
singularity exec --nv \
    --bind /work02/satoki/Search-R1:/workspace \
    --bind /work02/satoki/searchr1_data:/work02/satoki/searchr1_data \
    search_r1.sif \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path /work02/satoki/searchr1_data/e5_Flat.index \
        --corpus_path /work02/satoki/searchr1_data/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --faiss_gpu
