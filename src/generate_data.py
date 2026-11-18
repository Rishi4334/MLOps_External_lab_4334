from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def make_records(n=300, version=1, seed=42):
    rng = np.random.default_rng(seed + version)
    dpi = rng.choice([200, 240, 300, 400, 600], n, p=[.12,.18,.40,.22,.08]).astype(int)
    rotation = rng.normal(0.5 if version == 1 else 2.2, 2.0 if version == 1 else 3.1, n).clip(-8, 14)
    contrast = rng.normal(78 if version == 1 else 65, 11, n).clip(25, 100)
    marker = rng.beta(8 if version == 1 else 5, 2 if version == 1 else 3, n).clip(.05, 1)
    multi = rng.poisson(0.7 if version == 1 else 1.4, n).clip(0, 12).astype(int)
    blanks = rng.beta(2.2, 9, n) * 100
    damage = rng.beta(1.6 if version == 1 else 2.3, 6, n).clip(0, 1)
    scanner = rng.choice(['modern_flatbed','legacy','high_speed'], n, p=[.55,.25,.20])
    scanner_penalty = np.select([scanner=='legacy', scanner=='high_speed'], [0.55,0.18], 0.0)
    risk = (-1.8 + 0.11*np.abs(rotation) + 0.055*(70-contrast).clip(0) + 2.2*(1-marker) + 0.30*multi + 0.012*blanks + 2.0*damage + 0.65*(dpi<240) + scanner_penalty + rng.normal(0,.35,n))
    rejected = (risk > 1.05).astype(int)
    return pd.DataFrame({'record_id':[f'v{version}_{i:04d}' for i in range(n)],'dataset_version':f'v{version}','dpi':dpi,'rotation_angle':np.round(rotation,3),'print_contrast_score':np.round(contrast,2),'alignment_marker_visibility':np.round(marker,3),'multiple_marked_questions':multi,'blank_answer_percentage':np.round(blanks,2),'sheet_damage_score':np.round(damage,3),'scanner_type':scanner,'scan_rejected':np.where(rejected==1,'Yes','No')})

def generate():
    DATA.mkdir(exist_ok=True)
    v1=make_records(300,1,42); v2=make_records(300,2,42)
    v1.to_csv(DATA/'omr_scans_v1.csv',index=False); v2.to_csv(DATA/'omr_scans_v2.csv',index=False)
    pd.concat([v1,v2],ignore_index=True).to_csv(DATA/'omr_scans_all.csv',index=False)
    return v1,v2

if __name__=='__main__':
    a,b=generate(); print(f'Generated V1={len(a)} V2={len(b)} records; rejection rates={a.scan_rejected.eq("Yes").mean():.1%}, {b.scan_rejected.eq("Yes").mean():.1%}')
