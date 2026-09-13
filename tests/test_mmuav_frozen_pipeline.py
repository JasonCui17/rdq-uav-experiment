import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_mmuav_pose_pipeline as pipeline


class FrozenTests(unittest.TestCase):
    def test_cli_same_path(self):
        for split in ['validation_sub','heldout_test_sub']:
            args=pipeline.create_parser().parse_args(['--mode','full','--split',split,'--output-dir','unused',
                '--frozen-config','unused.json'])
            self.assertEqual(args.split,split)
            self.assertEqual(args.mode,'full')

    def test_heldout_requires_freeze_before_data_access(self):
        with patch.object(sys,'argv',['pipeline','--mode','full','--split','heldout_test_sub','--output-dir','unused']), \
             patch.object(pipeline.json,'loads',side_effect=AssertionError('No data/config should be loaded')):
            with self.assertRaises(SystemExit): pipeline.main()

    def fixture(self,root):
        mirror=root/'results/mmuav_reproduction';mirror.mkdir(parents=True)
        paths={name:root/name for name in ['splits','classifier_checkpoint','center_checkpoint','code.py']}
        for name,path in paths.items(): path.write_text(name)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        config=dict(status='FROZEN_AFTER_VALIDATION',data_root=str(root),split_definition_hash=sha(paths['splits']),
            M1_checkpoint_sha256=sha(paths['classifier_checkpoint']),M2_checkpoint_sha256=sha(paths['center_checkpoint']),
            pipeline_code_sha256={'code.py':sha(paths['code.py'])},trajectory_selection='longest_lived_track',
            AR_order=3,AR_grid_seconds=.1,max_gap_seconds=1.,spline_s=.5,nearest_evaluation_tolerance_seconds=.05,
            M3='BYPASSED',tracker=dict(process_noise=.15,measurement_noise=.001,missed_distance=3.,covar_trace_thresh=30.,min_points=1))
        file=root/'frozen.json';content=json.dumps(config)
        file.write_text(content);(mirror/'frozen_reproduction_config.json').write_text(content)
        return SimpleNamespace(frozen_config=file,data_root=root,smoke_no_ar_fit=False,sequence=None,
            **{k:v for k,v in paths.items() if k!='code.py'})

    def test_read_validate_and_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pipeline,'ROOT',Path(tmp)):
            args=self.fixture(Path(tmp))
            self.assertEqual(len(pipeline.validate_frozen(args)),64)
            args.center_checkpoint.write_text('wrong weights')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): pipeline.validate_frozen(args)

    def test_config_and_code_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pipeline,'ROOT',Path(tmp)):
            args=self.fixture(Path(tmp));original=args.frozen_config.read_text()
            args.frozen_config.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'config hash'): pipeline.validate_frozen(args)
            args.frozen_config.write_text(original)
            (Path(tmp)/'code.py').write_text('changed code')
            with self.assertRaisesRegex(ValueError,'code hash'): pipeline.validate_frozen(args)


if __name__=='__main__': unittest.main()
