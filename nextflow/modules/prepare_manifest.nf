process PREPARE_MANIFEST {
    tag 'reviewed annotations + manifest'
    label 'process_low'
    publishDir "${params.outdir}/01_manifest", mode: 'copy', overwrite: true

    input:
    path base_config
    path source_manifest
    path source_split
    path prepare_tool
    val project_root
    val git_commit
    val source_fingerprint
    val data_version
    val engine_version
    val test_train_sequence
    val test_val_sequence
    val cache_buster

    output:
    path 'prepared', emit: prepared

    script:
    def testArgs = test_train_sequence && test_val_sequence \
        ? "--test-train-sequence ${test_train_sequence} --test-val-sequence ${test_val_sequence}" \
        : ''
    """
    python ${prepare_tool} \
      --base-config ${base_config} \
      --manifest ${source_manifest} \
      --split ${source_split} \
      --output prepared \
      --project-root '${project_root}' \
      --git-commit '${git_commit}' \
      --source-fingerprint '${source_fingerprint}' \
      --data-version '${data_version}' \
      --nextflow-version '${engine_version}' \
      ${testArgs}
    echo '${cache_buster}' > prepared/cache_buster.txt
    """

    stub:
    """
    mkdir -p prepared
    cp ${base_config} prepared/base_config.yaml
    cp ${source_manifest} prepared/manifest.jsonl
    cp ${source_split} prepared/split.json
    printf '{"status":"STUB"}\n' > prepared/manifest_summary.json
    printf '{"status":"STUB","git_commit":"${git_commit}"}\n' > prepared/provenance.json
    printf 'nextflow=${engine_version}\nstatus=STUB\n' > prepared/environment.txt
    echo '${cache_buster}' > prepared/cache_buster.txt
    """
}
