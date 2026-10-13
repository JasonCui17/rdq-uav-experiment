# V2 real forward trace and replay

This tool runs the existing V2 forward path in WSL with **trained** A2, B2, or B3 checkpoints. It does not train, alter inputs, or infer again from the browser. The default split is `validation_sub`. Use `heldout_test_sub` only for final evaluation and pass `--final-heldout` explicitly.

## Call chain and tensor contract

| Stage | Existing code | Tensor examples for `seq0026`, `1706254919.2855873` |
| --- | --- | --- |
| Query, same sequence 1 s history, selected image and YOLO label | `data.py`, `contracts.py` | points `[46720,3]`, time/sensor `[46720]` |
| L0/L1/L2, point and parent maps | `radar_geometry.py` | `[7652,3]`, `[3046,3]`, `[1191,3]` coordinates |
| A2 Point MLP, eight slots, VQSA, L0 embedding | `lidar_learned_sbe.py`, `radar_sbe.py` | `[46720,5]` → `[46720,32]`; slots `[7652,8,66]`; L0 `[7652,128]` |
| Window attention, merge, global L2, up and skip | `radar_model.py` | selected L0 attention `[4,64,64]`; features `[V,128]` |
| Raw prediction, stable TopK, 1 m NMS | `radar_model.py`, `radar_selector.py` | 7652 logits/XYZ; pre NMS 50; final 10 |
| Project, visual evidence, association, score | `geometry.py`, `interaction.py`, `scoring.py` | candidate indexed arrays and tables |

The trace recorder observes these modules only inside its context and removes all hooks on exit or exception. Spatial attention Q/K/V, bias, logits, weights, masks, residual and output are actual tensors from the chosen window. VQSA and cross modal PyTorch `MultiheadAttention` run with `need_weights=False`; their actual inputs and outputs are saved, but **their weights are unavailable**. The viewer labels this limit and never presents reconstructed weights as actual. `--focus-voxel` selects an L0 token; parent maps choose its L1/L2 focus. Full point and hierarchy arrays remain in `.npy`; browser arrays are marked display samples.

## WSL commands

Run from the repository root. `--device cpu` works on this WSL machine; use `cuda:0` when WSL GPU access is fixed. CPU is slower, especially for B2/B3. `--display-points` changes only browser drawing, and `--max-tensor-mb` omits individual tensors above the cap with their shape and omission reason recorded.

```bash
cd /home/jasoncui/projects/rdq-uav-experiment
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-v2-trace XDG_CACHE_HOME=/tmp/v2-trace-cache

python experiments/multimodal_v2/diagnostics/trace_model.py list --sequence seq0026 --limit 20
python experiments/multimodal_v2/diagnostics/trace_model.py trace \
  --models A2 --sequence seq0026 --query-time 1706254919.2855873 \
  --device cpu --output outputs/v2_trace_replay
python experiments/multimodal_v2/diagnostics/trace_model.py trace \
  --models A2 B2 B3 --sequence seq0026 --query-time 1706254919.2855873 \
  --device cpu --output outputs/v2_trace_replay
python experiments/multimodal_v2/diagnostics/trace_model.py batch \
  --models A2 B2 B3 --indices 0 1 --device cpu --output outputs/v2_trace_replay
python experiments/multimodal_v2/diagnostics/trace_model.py viewer \
  --output outputs/v2_trace_replay --host 127.0.0.1 --port 8767
```

Open `http://127.0.0.1:8767` in a browser. Select a query and model. In the 3D panel, left drag rotates; Shift + left drag, right drag, or the pan mode button pans; the mouse wheel zooms around the pointer; **适配视角** resets the position and scale. The crosshair stays visible over the radar canvas. In the image panel, use the mouse wheel or `−` / `＋` buttons to zoom the image together with its boxes and projections, drag to pan, and double click or use **重置图像** to reset. Use the stage slider/play control, click displayed points or voxels for their original IDs and parent relationships, and click a candidate to link its image box/projected UV. The comparison table uses the **same `sample_id`**, not array position. The viewer shows sensor/time colors, feature statistics and vectors, selected spatial attention, selector decisions, pair reasons, evidence validity, gate/delta, rank changes, GT distance, and missing/invalid flags. Channels of learned 128-dimensional features have no physical labels. There is no input intervention or rerun from the browser.

The image panel overlays every exported final visual candidate box in blue with its visual source ID (`V ID`). The table below the image lists each box's final candidate row, 2D score before/after, original-image XYXY pixels, and associated radar token. Click a row to highlight that box in pink and link it to the 3D candidate; use **全部视觉候选框** to hide or show the other boxes. Green is GT. These are the visual candidates after NMS, not all raw DINO predictions. Refresh the viewer page to use this display with existing exports; no new trace is needed.

The 3D panel's **层级** selector displays real L0/L1/L2 voxel centers with full voxel counts and per-voxel point-count statistics. Voxel marker size scales with the logarithm of its actual point count. Select a voxel or enter its original ID to inspect its parent, children, points, window neighbors and saved feature vectors. Display decimation is explicitly shown; original IDs still address the full saved arrays. Click a point or enter its original ID to select its Point MLP input in **张量阶段**; choose `frontend.point_mlp_output` there to read all 32 output values. The five input fields are `(point_xyz − L0_center)/0.5 m` in x/y/z order, the learned scalar sensor embedding, and `observation_time − query_time` in seconds. The 32 outputs are numbered learned activations of `Linear(5→32)+GELU`; they have no assigned physical meaning.

To locate a radar point in the left image, click a displayed raw point or enter its **原始点 ID** and select **查看点并投影**. You can also enter `x, y, z` in metres in the radar/GT reference frame and select **投影 XYZ**. The server reads the saved point XYZ when an ID is given, then uses the existing omni+radtan calibration to compute source-image UV on demand; no model checkpoint is loaded and no model forward is rerun. A cyan cross marks a valid position in the selected left image. The image panel displays the original XYZ and UV; missing images, out-of-frame points, and invalid projections are reported instead of drawing a false marker. This on-demand projection is a geometry query, not a tensor recorded during the original model forward.

In **候选筛选、关联和评分**, the final table shows each output candidate row, its radar L0 token ID, and its visual source ID. The TopK/NMS table shows the original ranking, L0 token ID, linked final candidate row, and suppressor token. Click a selected row to follow it into the final candidate; click a rejected row to inspect its raw prediction and rejection reason in **选中元素** and highlight its raw XYZ in the 3D panel. The radar token ID indexes an L0 voxel/raw prediction, not an original point; use the L0 voxel ID field to inspect the points belonging to that token.

In **张量阶段**, click a tensor and enter comma-separated leading indices: `frontend.point_mlp_input[point_id]` returns five values, `frontend.point_mlp_output[point_id]` returns 32, and `frontend.slot_features[voxel_id,slot_id]` returns 66. The server reads only the requested `.npy` slice; it rejects slices over 4096 values and reports unavailable tensors. L1/L2 layers and per-point MLP values also work with traces exported before this viewer update, because their complete `.npy` tensors were already saved.

`trace` accepts `--index N` (zero based within `--sequence`, otherwise whole split) instead of `--query-time`. Time matching prints the selected actual time and tolerance; the default tolerance is `1e-5` seconds. `list` finds query identities if no local per-query evaluation table exists. Use `--data-root PATH` for another WSL dataset root. Each model is built, strictly loaded from its **full** checkpoint, run once normally and once with observers, then released before the next model. B2/B3 do not require the old B1 initialization checkpoint for this strict full load.

## Output

Each query is at `<output>/<sample_id>/<A2|B2|B3>/`:

- `manifest.json`: schema, commit, source/effective config and checkpoint hashes, path overrides, calibration hashes, saved/omitted status.
- `summary.json`, `parity.json`: tensor shapes/dtypes/min/max/mean/std/nonfinite counts, final result and forward equality. Floating fields use `atol=rtol=1e-5`; indices and association use exact equality.
- `tensors/*.npy`, `tensor_index.json`: Python-readable complete arrays unless a named tensor exceeds `--max-tensor-mb`.
- `selector.json`, `candidates.json`, `events.json`: source token order, actual evaluated NMS boundary, reject/suppress reason, association reason, scores and rank changes.
- `browser.json`, `left_image.png`: local browser data and selected real left image. `browser.json` points/voxels are display samples only; original indices and full counts are included.
- `effective_config.yaml`: WSL effective configuration. Server absolute dataset paths are replaced only if unavailable locally, and override source is recorded.

Read tensors with `numpy.load(..., allow_pickle=False)`; the browser API serves individual vectors and small arrays without reading `.pt`. `manifest.complete=false` means one or more tensors exceeded the configured per tensor cap. A missing `manifest.json` means the run did not finish; inspect the traceback before using its partial arrays.

## Missing assets and environment

Expected trained checkpoints are listed in `trace_model.py` under `ASSETS`. The optional B1 initialization checkpoint is not required to strictly restore B2/B3. If any A2/B2/B3 config or full checkpoint is absent, copy only that missing file from the server. For example:

```bash
scp -P 21305 root@connect.bjb1.seetacloud.com:/root/autodl-tmp/rdq-uav-experiment-v2/outputs/b2_b3_a2_coco_seed42/b3/checkpoints/best.ckpt outputs/b2_b3_a2_coco_seed42/b3/checkpoints/
```

SSH may ask for interactive authentication; do not put passwords in a command. The server is only an asset source. Do not copy a server `detrex._C` binary across environments. Check local imports with `python -c 'import torch, torchvision, detrex._C; print(torch.__version__, torchvision.__version__)'` before tracing.
