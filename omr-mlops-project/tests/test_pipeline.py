import pandas as pd
from fastapi.testclient import TestClient
from src.api import app
from src.features import engineer_features

def test_feature_engineering_bounds():
    df=pd.DataFrame([{'dpi':300,'rotation_angle':2,'print_contrast_score':80,'alignment_marker_visibility':.9,'multiple_marked_questions':1,'blank_answer_percentage':5,'sheet_damage_score':.1,'scanner_type':'modern_flatbed','scan_rejected':'No'}])
    out=engineer_features(df); assert 0<=out.scan_quality_index.iloc[0]<=1; assert 0<=out.marking_error_ratio.iloc[0]<=1

def test_api_parameter_validation():
    client=TestClient(app); r=client.post('/predict',json={'dpi':50,'rotation_angle':0,'print_contrast_score':80,'alignment_marker_visibility':.9,'multiple_marked_questions':0,'blank_answer_percentage':0,'sheet_damage_score':0,'scanner_type':'modern_flatbed'}); assert r.status_code==422

def test_api_prediction_contract():
    client=TestClient(app); r=client.post('/predict',json={'dpi':300,'rotation_angle':0,'print_contrast_score':90,'alignment_marker_visibility':.98,'multiple_marked_questions':0,'blank_answer_percentage':2,'sheet_damage_score':.02,'scanner_type':'modern_flatbed'}); assert r.status_code==200; assert r.json()['prediction'] in ['Scan Accepted','Manual Verification Required']
