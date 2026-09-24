# Runtime dataset

Dataset contents are intentionally excluded from Git. Before training, place
the MMAUD official training tree at `mmaud_official_train/`, or create a local
symbolic link with that name.

Run `python tools/check_assets.py` to verify it.

The symlink is only a convenience. On another server, leave it absent and set:

```bash
export RDQ_DATA_ROOT=/path/to/MMAUD/official/train
```
