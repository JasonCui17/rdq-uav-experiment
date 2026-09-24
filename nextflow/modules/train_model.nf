process TRAIN_MODEL {
    tag "E5 epochs=${epochs} train_limit=${train_limit}"
    label 'process_gpu'
    publishDir "${params.outdir}/02_training", mode: 'copy', overwrite: true

    input:
    path prepared
    path materialize_tool
    path training_entrypoint
    val project_root
    val epochs
    val batch_size
    val accumulate
    val num_workers
    val train_limit
    val val_limit
    val precision
    val source_fingerprint
    val durable_run_dir

    output:
    path 'training_run', emit: run

    script:
    def trainLimitArg = train_limit as int > 0 ? "--train-limit ${train_limit}" : ''
    def valLimitArg = val_limit as int > 0 ? "--val-limit ${val_limit}" : ''
    """
    WORK_DIR=\"\${PWD}\"
    python ${materialize_tool} \
      --base-config \"\${WORK_DIR}/${prepared}/base_config.yaml\" \
      --manifest \"\${WORK_DIR}/${prepared}/manifest.jsonl\" \
      --split \"\${WORK_DIR}/${prepared}/split.json\" \
      --output \"\${WORK_DIR}/training_config.yaml\" \
      --run-output \"\${WORK_DIR}/training_run\" \
      --epochs ${epochs} \
      --batch-size ${batch_size} \
      --accumulate ${accumulate} \
      --num-workers ${num_workers}

    DURABLE_RUN_DIR='${durable_run_dir}'
    mkdir -p \"\${DURABLE_RUN_DIR}\"
    CONFIG_SHA=\$(sha256sum \"\${WORK_DIR}/training_config.yaml\" | awk '{print \$1}')
    CONTRACT=\"${source_fingerprint}:\${CONFIG_SHA}\"
    if [[ -f \"\${DURABLE_RUN_DIR}/pipeline_contract.txt\" ]] && \
       [[ \"\$(cat \"\${DURABLE_RUN_DIR}/pipeline_contract.txt\")\" != \"\${CONTRACT}\" ]]; then
      echo 'Refusing to resume an incompatible durable training state; select a new --outdir.' >&2
      exit 65
    fi
    printf '%s\n' \"\${CONTRACT}\" > \"\${DURABLE_RUN_DIR}/pipeline_contract.txt\"

    RESUME_ARG=''
    if [[ -f \"\${DURABLE_RUN_DIR}/checkpoints/last.ckpt\" ]]; then
      RESUME_ARG='--resume auto'
    fi
    PYTHONPATH='${project_root}/src' PYTHONUNBUFFERED=1 \
      python ${training_entrypoint} \
        --config \"\${WORK_DIR}/training_config.yaml\" \
        --output \"\${DURABLE_RUN_DIR}\" \
        --accelerator gpu \
        --devices 1 \
        --precision ${precision} \
        --epochs ${epochs} \
        --num-workers ${num_workers} \
        ${trainLimitArg} ${valLimitArg} \${RESUME_ARG}
    cp ${prepared}/provenance.json \"\${DURABLE_RUN_DIR}/pipeline_provenance.json\"
    printf '%s\n' '${source_fingerprint}' > \"\${DURABLE_RUN_DIR}/source_fingerprint.txt\"
    cp -a \"\${DURABLE_RUN_DIR}\" training_run
    """

    stub:
    """
    mkdir -p training_run/checkpoints training_run/logs
    touch training_run/checkpoints/last.ckpt
    touch training_run/checkpoints/best.ckpt
    cp ${prepared}/base_config.yaml training_run/effective_config.yaml
    printf '{"status":"STUB"}\n' > training_run/effective_config.json
    cp ${prepared}/provenance.json training_run/pipeline_provenance.json
    printf '%s\n' '${source_fingerprint}' > training_run/source_fingerprint.txt
    """
}
