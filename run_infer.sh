#!/bin/bash
singularity exec --nv \
    --bind /work02/satoki/Search-R1:/workspace \
    search_r1.sif \
    python /workspace/infer.py
