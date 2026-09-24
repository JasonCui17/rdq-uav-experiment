#!/usr/bin/env nextflow

include { PREPARE_MANIFEST }   from './modules/prepare_manifest'
include { TRAIN_MODEL }        from './modules/train_model'
include { EVALUATE_AND_REPORT } from './modules/evaluate_report'

workflow {
    def projectRoot = file(params.project_root, checkIfExists: true).toString()
    def outputRoot = file(params.outdir).toString()
    def baseConfig = file(params.config, checkIfExists: true)
    def manifest = file(params.manifest, checkIfExists: true)
    def split = file(params.split, checkIfExists: true)
    def prepareTool = file("${projectRoot}/nextflow/bin/prepare_manifest.py", checkIfExists: true)
    def materializeTool = file("${projectRoot}/nextflow/bin/materialize_config.py", checkIfExists: true)
    def trainingEntrypoint = file("${projectRoot}/tools/train_multimodal_v1_lightning.py", checkIfExists: true)
    def evaluationEntrypoint = file("${projectRoot}/tools/evaluate_multimodal_v1_lightning.py", checkIfExists: true)

    PREPARE_MANIFEST(
        baseConfig,
        manifest,
        split,
        prepareTool,
        projectRoot,
        params.git_commit,
        params.source_fingerprint,
        params.data_version,
        workflow.nextflow.version.toString(),
        params.test_train_sequence,
        params.test_val_sequence,
        params.cache_buster,
    )
    TRAIN_MODEL(
        PREPARE_MANIFEST.out.prepared,
        materializeTool,
        trainingEntrypoint,
        projectRoot,
        params.epochs,
        params.batch_size,
        params.accumulate,
        params.num_workers,
        params.train_limit,
        params.val_limit,
        params.precision,
        params.source_fingerprint,
        "${outputRoot}/.training_state",
    )
    EVALUATE_AND_REPORT(
        TRAIN_MODEL.out.run,
        evaluationEntrypoint,
        projectRoot,
        params.val_limit,
        params.num_workers,
        params.report_revision,
    )
}
