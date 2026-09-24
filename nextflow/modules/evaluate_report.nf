process EVALUATE_AND_REPORT {
    tag 'best checkpoint FP32 evaluation'
    label 'process_gpu'
    publishDir "${params.outdir}/03_evaluation", mode: 'copy', overwrite: true

    input:
    path training_run
    path evaluation_entrypoint
    val project_root
    val val_limit
    val num_workers
    val report_revision

    output:
    path 'evaluation', emit: report

    script:
    def valLimitArg = val_limit as int > 0 ? "--val-limit ${val_limit}" : ''
    """
    CHECKPOINT='${training_run}/checkpoints/best.ckpt'
    if [[ ! -f \${CHECKPOINT} ]]; then
      CHECKPOINT='${training_run}/checkpoints/last.ckpt'
    fi
    test -f \${CHECKPOINT}
    mkdir -p evaluation
    PYTHONPATH='${project_root}/src' PYTHONUNBUFFERED=1 \
      python ${evaluation_entrypoint} \
        --config ${training_run}/effective_config.yaml \
        --checkpoint \${CHECKPOINT} \
        --output evaluation \
        --accelerator gpu \
        --devices 1 \
        --num-workers ${num_workers} \
        ${valLimitArg}
    cp ${training_run}/pipeline_provenance.json evaluation/pipeline_provenance.json
    printf '%s\n' '${report_revision}' > evaluation/report_revision.txt
    """

    stub:
    """
    mkdir -p evaluation
    printf '{"status":"STUB","metrics":{}}\n' > evaluation/metrics.json
    printf '# Multimodal V1 evaluation\n\nStub execution.\n' > evaluation/report.md
    cp ${training_run}/pipeline_provenance.json evaluation/pipeline_provenance.json
    printf '%s\n' '${report_revision}' > evaluation/report_revision.txt
    """
}
