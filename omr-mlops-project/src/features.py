from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
NUMERIC=['dpi','rotation_angle','print_contrast_score','alignment_marker_visibility','multiple_marked_questions','blank_answer_percentage','sheet_damage_score','scan_quality_index','marking_error_ratio']
CATEGORICAL=['scanner_type']
TARGET='scan_rejected'

def engineer_features(df):
    out=df.copy()
    out['scan_quality_index']=(0.30*(out.dpi/600)+0.25*(out.print_contrast_score/100)+0.25*out.alignment_marker_visibility+0.20*(1-out.sheet_damage_score)).clip(0,1)
    out['marking_error_ratio']=((out.multiple_marked_questions + 0.35*out.blank_answer_percentage)/100).clip(0,1)
    if TARGET in out.columns:
        if not pd.api.types.is_numeric_dtype(out[TARGET]):
            out[TARGET]=out[TARGET].astype(str).str.strip().str.lower().eq('yes').astype(int)
        else:
            out[TARGET]=out[TARGET].astype(int)
    return out

def build_preprocessor():
    return ColumnTransformer([('num',StandardScaler(),NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])

def create_feature_store(input_csv, output_csv=None):
    df=engineer_features(pd.read_csv(input_csv))
    output_csv=output_csv or ROOT/'artifacts/feature_store.csv'; Path(output_csv).parent.mkdir(exist_ok=True)
    df.to_csv(output_csv,index=False); return df

if __name__=='__main__':
    p=ROOT/'data/omr_scans_all.csv'; print(create_feature_store(p).head())
