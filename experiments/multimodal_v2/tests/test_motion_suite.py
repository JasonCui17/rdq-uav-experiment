"""Runner preparation invariants without requiring real MMAUD/CUDA."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT/'experiments/multimodal_v2/mtce/run_suite.py'


def test_four_ablation_configs_and_duplicate_guard(tmp_path):
    data=tmp_path/'data';data.mkdir()
    checkpoint=tmp_path/'a2.ckpt';checkpoint.write_bytes(b'only hashed during preparation')
    suite=tmp_path/'suite'
    cmd=[sys.executable,str(RUNNER),'--action','prepare','--output',str(suite),
         '--checkpoint',str(checkpoint),'--data-root',str(data),'--epochs','3']
    subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True)
    manifest=json.loads((suite/'manifest.json').read_text())
    assert [r['name'] for r in manifest['runs']] == ['baseline','gate','velocity','joint']
    for run in manifest['runs']:
        c=yaml.safe_load(Path(run['config']).read_text())
        r=yaml.safe_load(Path(c['initialization']['lidar_config']).read_text())
        m=r['model']['voxel']['motion_centroid']
        assert c['training']['radar_train_mode'] == 'lora' and c['training']['epochs'] == 3
        assert c['checkpoint']['monitor'] == 'val/success_1m'
        assert r['model']['lora']['enabled']
        assert m['enabled'] == (run['name'] != 'baseline')
        assert m['feature_gate'] == (run['name'] in ('gate','joint'))
        assert m['velocity_guidance'] == (run['name'] in ('velocity','joint'))
    result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode != 0 and 'already exists' in result.stderr
    # Reject configuration drift before entering a subprocess or loading data.
    p=Path(manifest['runs'][0]['config']);p.write_text(p.read_text()+'\n# edited\n')
    result=subprocess.run([sys.executable,str(RUNNER),'--action','gate','--output',str(suite),
                           '--device','cpu'],capture_output=True,text=True)
    assert result.returncode != 0 and 'configuration changed' in result.stderr
