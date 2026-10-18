"""Render a completed pipeline metrics JSON as a short local report."""
import argparse,json
from pathlib import Path

def render(directory):
    data=json.loads((directory/'metrics.json').read_text())
    lines=['# RDQ-UAV 推理结果','',f"时间轴：{data['protocol']['schedule']}；sequence数：{len(data['sequences'])}；smoke：{data['protocol']['smoke']}。",'', '| 分支 | RMSE(m) | GT覆盖率 | AP50 | AP50–95 | Precision | Recall |','|---|---:|---:|---:|---:|---:|---:|']
    def number(v,percent=False):return '--' if v is None else f'{v*100:.2f}%' if percent else f'{v:.4f}'
    for mode in ['realtime','lag05']:
        radar=data['radar'][mode];image=data['image'][mode]
        lines.append('| '+mode+' | '+number(radar['rmse_3d_m'])+' | '+number(radar['coverage_at_GT'],True)+' | '+' | '.join(number(image.get(k),True) for k in ['AP50','AP','Precision','Recall'])+' |')
    lines+=['','| Sequence | 传感器查询 | 图像总数 | bbox输出图数 | 完整左图覆盖 |','|---|---:|---:|---:|---|']
    for seq,m in data['sequences'].items():
        c=m['coverage'];lines.append(f"| {seq} | {c['sensor_queries']} | {c['all_left_images']} | {c['predicted_images']} | {c['complete_image_coverage']} |")
    lines+=['','sensor与benchmark协议的查询密度、图像分母不同，不直接比较历史分数。标注只读取输入MMAUD的2d_detect；缺失/空标签的图片不作为负例。平滑输出含最多0.5秒观测前视，不是零延迟实时。','', '轨迹缺失以null保留，覆盖率单独报告。bbox为左鱼眼1280×960原图xyxy像素坐标。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();render(a.directory)
