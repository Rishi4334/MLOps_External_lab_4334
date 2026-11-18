from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .features import NUMERIC,CATEGORICAL,engineer_features
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'artifacts/omr_rejection_model.joblib'
app=FastAPI(title='OMR Rejection Prediction API',version='1.0.0')
class ScanInput(BaseModel):
    dpi:int=Field(...,ge=150,le=800); rotation_angle:float=Field(...,ge=-15,le=15); print_contrast_score:float=Field(...,ge=0,le=100); alignment_marker_visibility:float=Field(...,ge=0,le=1); multiple_marked_questions:int=Field(...,ge=0,le=60); blank_answer_percentage:float=Field(...,ge=0,le=100); sheet_damage_score:float=Field(...,ge=0,le=1); scanner_type:str
@app.get('/health')
def health(): return {'status':'ok','model_loaded':MODEL_PATH.exists()}
@app.post('/predict')
def predict(item:ScanInput):
    if not MODEL_PATH.exists(): raise RuntimeError('Model artifact missing. Run python -m src.pipeline first.')
    df=engineer_features(pd.DataFrame([item.model_dump()]))
    pred=int(joblib.load(MODEL_PATH).predict(df[NUMERIC+CATEGORICAL])[0])
    return {'prediction':'Manual Verification Required' if pred else 'Scan Accepted','scan_rejected':bool(pred),'model':'registered OMR_Rejection_Model'}
