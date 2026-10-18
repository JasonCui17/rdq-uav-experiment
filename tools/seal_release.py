"""Record the release contents. Run once after validation, before final handoff."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def seal():
    files={};links={}
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if any(part in ['runs','__pycache__','.pytest_cache','.git'] for part in relative.parts) or p.name=='MANIFEST.json':continue
        if p.is_symlink():
            if not p.resolve().is_relative_to(ROOT):raise ValueError('External symlink: '+str(relative))
            links[str(relative)]=str(p.readlink());continue
        if not p.is_file() or p.suffix in ['.pyc','.so']:continue
        files[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    payload={'release':'rdq-uav-final','schema':1,'runtime_external_project_dependencies':False,'external_dataset_allowed':True,'files_sha256':files,'relative_symlinks':links,'platform_note':'detrex compiled .so bundled for local environment, excluded from portable checksum because rebuilt on another CUDA environment','selected_models':{'radar':'weights/radar.ckpt','vision_full':'weights/vision_full.ckpt','vision_crop':'weights/vision_crop.ckpt'},'default_inference_schedule':'sensor','reference_schedule':'GT filenames benchmark','annotation_protocol':'MMAUD 2d_detect only; missing/empty labels unscored, not negative'}
    (ROOT/'MANIFEST.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print('Sealed',len(files),'files and',len(links),'internal symlinks')

if __name__=='__main__':seal()
