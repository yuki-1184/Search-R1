#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash jobs/pjsub_interact.sh [-r rscgrp] [-t elapse] [-n node] [-g project] [-w wait_time]

Options:
  -r  Resource group (default: ${RESOURCE_TYPE:-b-inter})
  -t  Elapse time     (default: ${JOBTIME:-02:00:00})
  -n  Node count      (default: ${NODES:-1})
  -g  Project code    (default: ${PROJECT_CODE:-pj24003196})
  -w  Wait time sec   (default: ${WAIT_TIME:-3000})

Example:
  bash jobs/pjsub_interact.sh -r a-batch -t 04:00:00
EOF
}

RESOURCE_TYPE="${RESOURCE_TYPE:-b-inter}"
JOBTIME="${JOBTIME:-02:00:00}"
NODES="${NODES:-1}"
PROJECT_CODE="${PROJECT_CODE:-pj24003196}"
WAIT_TIME="${WAIT_TIME:-3000}"
JOBENV="${JOBENV:-singularity}"

while getopts ":r:t:n:g:w:h" opt; do
    case "$opt" in
        r) RESOURCE_TYPE="$OPTARG" ;;
        t) JOBTIME="$OPTARG" ;;
        n) NODES="$OPTARG" ;;
        g) PROJECT_CODE="$OPTARG" ;;
        w) WAIT_TIME="$OPTARG" ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            exit 2
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
            usage
            exit 2
            ;;
    esac
done

if ! command -v pjsub >/dev/null 2>&1; then
    echo "pjsub command not found in PATH." >&2
    exit 127
fi

echo "Submitting interactive job:"
echo "  rscgrp=${RESOURCE_TYPE}"
echo "  elapse=${JOBTIME}"
echo "  node=${NODES}"
echo "  project=${PROJECT_CODE}"
echo "  jobenv=${JOBENV}"
echo "  wait-time=${WAIT_TIME}"

exec pjsub \
    --interact \
    -L "rscgrp=${RESOURCE_TYPE}" \
    -L "node=${NODES}" \
    -L "elapse=${JOBTIME}" \
    -L "jobenv=${JOBENV}" \
    -g "${PROJECT_CODE}" \
    --sparam "wait-time=${WAIT_TIME}"
