#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${PIPELINE_DIR}/results/stub_contract"
LOG1="${PIPELINE_DIR}/tests/.stub_first.log"
LOG2="${PIPELINE_DIR}/tests/.stub_resume.log"
LOG3="${PIPELINE_DIR}/tests/.stub_invalidated.log"
LOG4="${PIPELINE_DIR}/tests/.stub_report_only.log"

rm -rf "${OUTDIR}"
"${PIPELINE_DIR}/run.sh" -profile test -stub-run \
  --outdir "${OUTDIR}" --cache_buster contract-a -ansi-log false | tee "${LOG1}"

test -f "${OUTDIR}/01_manifest/prepared/provenance.json"
test -f "${OUTDIR}/01_manifest/prepared/environment.txt"
test -f "${OUTDIR}/02_training/training_run/checkpoints/last.ckpt"
test -f "${OUTDIR}/03_evaluation/evaluation/metrics.json"
test -f "${OUTDIR}/03_evaluation/evaluation/report.md"

"${PIPELINE_DIR}/run.sh" -profile test -stub-run -resume \
  --outdir "${OUTDIR}" --cache_buster contract-a -ansi-log false | tee "${LOG2}"
grep -q 'Cached process' "${LOG2}"
grep -q 'Cached process > PREPARE_MANIFEST' "${LOG2}"
grep -q 'Cached process > TRAIN_MODEL' "${LOG2}"
grep -q 'Cached process > EVALUATE_AND_REPORT' "${LOG2}"

# A report-only cache-key change must reuse manifest preparation and training,
# while rerunning evaluation. This models an evaluation/report failure followed
# by `-resume` without paying the training cost again.
"${PIPELINE_DIR}/run.sh" -profile test -stub-run -resume \
  --outdir "${OUTDIR}" --cache_buster contract-a --report_revision v2 -ansi-log false | tee "${LOG4}"
grep -q 'Cached process > PREPARE_MANIFEST' "${LOG4}"
grep -q 'Cached process > TRAIN_MODEL' "${LOG4}"
if grep -q 'Cached process > EVALUATE_AND_REPORT' "${LOG4}"; then
  echo 'evaluation cache invalidation failed: report revision reused evaluation' >&2
  exit 1
fi

"${PIPELINE_DIR}/run.sh" -profile test -stub-run -resume \
  --outdir "${OUTDIR}" --cache_buster contract-b -ansi-log false | tee "${LOG3}"
if grep -q 'Cached process' "${LOG3}"; then
  echo 'cache invalidation failed: changed cache_buster reused a task' >&2
  exit 1
fi

echo 'NEXTFLOW_STUB_CONTRACT_PASS'
