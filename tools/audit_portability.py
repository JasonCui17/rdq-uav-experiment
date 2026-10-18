"""Check that the final engineering source has no dependencies on older projects."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=['/home/jasoncui/projects/rdq-uav-experiment','/root/autodl-tmp/rdq-uav-experiment','/home/jasoncui/Downloads/','/Users/jasoncui/']

def audit():
    errors=[];count=0
    for directory in ['src','configs','calibration','reference','tools','docs']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_symlink() and not p.resolve().is_relative_to(ROOT):errors.append('External symlink: '+str(p))
            if not p.is_file() or p.suffix not in ['.py','.json','.yaml','.yml','.md','.sh']:continue
            if p.resolve()==Path(__file__).resolve():continue
            count+=1
            s=p.read_text(errors='replace')
            for prefix in FORBIDDEN:
                if prefix in s:errors.append(f'External project path: {p.relative_to(ROOT)}')
    for p in (ROOT/'third_party').rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(ROOT):errors.append('External vendored symlink: '+str(p))
    result={'checked_text_files':count,'external_project_references':errors,'passed':not errors}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    if not audit()['passed']:raise SystemExit(1)
