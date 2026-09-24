# MMAUD GT-guided YOLO annotation web tool

For the **left** MMAUD camera (`1280 × 960`), in WSL. It uses the frozen
`calibration/official_left_p4_current_geometry.json` and
`configs/calibration/mmaud_v1_omni.yaml` from the repository. The web server
runs on Python's standard `http.server`, without requiring Flask/FastAPI.

## Install into existing repo

Place these paths at the **existing repository root**:

```
tools/gt_bbox_annotation_server.py
tools/annotation_web/index.html
tests/test_gt_bbox_annotation_server.py   # optional regression tests
```

If using the ZIP bundle, from the root:

```bash
unzip -o /path/to/gt_bbox_annotator.zip
conda activate rdq
python tools/gt_bbox_annotation_server.py --host 127.0.0.1 --port 8765
```

Open `http://localhost:8765` in the Windows browser. If the Windows browser
cannot access WSL loopback, use WSL networking / port-forward configuration;
avoid exposing this write-enabled tool to public networks.

## Supported on-disk structure

```
data/mmaud_official_train/
  seq0001/
    Image/                # numeric timestamp filenames (PNG/JPG), e.g. 1706255497.138053.png
    ground_truth/         # timestamp.npy, XYZ meters, shape [3]
    2d_detect/            # timestamp.txt, YOLO normalized relative to the LEFT 1280x960 image
```

If a PNG contains both fisheye cameras concatenated at `2560×960`, the tool
**only displays and annotates its left `1280×960` crop**. Existing 1280×960
left-crop images are supported. Other resolutions are rejected instead of
silently remapping old labels.

The GT matching convention is `gt_time = image_time + time_offset_s`, read from
the calibration JSON (current offset 0). Nearest GT time must be within 0.08s
by default (`--max-gt-gap-s` to override); the default projection-to-box
inconsistency threshold is 32px (`--discrepancy-px` to override). These are
**annotation-tool thresholds** and are not formal calibration acceptance criteria.

## Workflow

1. Choose a sequence. Existing YOLO `.txt` files are treated as confirmed and
   never modified on initial loading. If the label file does not exist, the
   tool generates a **non-persisted** candidate box using projected GT shift
   relative to the nearest preceding confirmed positive frame. Before the
   first eligible reference frame, a later confirmed frame may backfill.
2. Review each frame. Arrow keys move the active box by 1px; Ctrl+arrows move
   it by 5px; Shift+arrows change width/height by 1px. A/D switch frames;
   Enter confirms and goes next. Drag the rectangle to move, drag its corners
   to resize; drag empty space to draw a new box. Zoom window allows the same
   interactions and mouse wheel controls zoom.
3. Confirmed frames write YOLO labels immediately. A newly confirmed frame
   re-generates only later *unconfirmed* candidates; existing labels and
   earlier drafts stay unchanged. Use the explicit recompute button if needed.
4. Bulk delete saves blank YOLO labels for the specified range. This is an
   **explicit human action**, not automated detection of target absence. Blank
   labels stop propagation until a subsequent positive reference is available.
   A review-state JSON distinguishes confirmed negatives from uncertain
   frames. The browser also remembers the last visited sequence and frame. Uncertain frames are excluded from output and do not create blanks.
5. A confirmed bbox more than 32px away from its projected GT point (distance
   to rectangle, not center) is shown as inconsistent and cannot be an anchor
   unless you explicitly review and force-enable it.

YOLO output: `2d_detect/<image_stem>.txt` with one row per bbox,
`class cx cy w h` (normalized). Existing multi-box labels are preserved and
editable; only **one-box confirmed frames** can be used as automatic reference
anchors, because this version assumes one UAV track per sequence.

Metadata: `2d_detect/.gt_bbox_review_state.json`. The first time any existing
label is edited, the original is copied to `2d_detect/_annotation_backup/`.
Do not run two copies of the server against the same sequence or modify `.txt`
files externally while the application is open. Reload the sequence after any
out-of-band label edits.

**E5 integration:** `tools/train_multimodal_v1_full.py` currently consumes
`manifests/multimodal_v1/vision_manual161_train.jsonl`, not newly written
YOLO `.txt` files. Generating new labels does not automatically add them to
training; a separate manifest export step is required once you approve the
annotations. Do not mix validation/test labels into the training manifest.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_gt_bbox_annotation_server.py' -v
```

## Important limitations

- This is a **GT-guided box translation**, not an image recognizer or tracker.
  Generated proposals are unverified. Visibility, orientation, and scale may
  change; the reviewer must adjust box size and delete false proposals.
- For an unmapped or invalid GT, no proposal is generated. You can still draw
  and save a box manually; without a valid projection it is not an anchor.
- Browser server is local, unauthenticated, and grants write access to
  `2d_detect`; do not bind `--host 0.0.0.0` on shared networks.

## Multimodal training portability

Training and evaluation configs use repository-relative paths. Datasets and
large checkpoints can live anywhere on a new server through explicit
`RDQ_*` environment variables. See
[`docs/PORTABLE_PATHS.md`](docs/PORTABLE_PATHS.md) and run
`python tools/check_assets.py` before starting an experiment.
