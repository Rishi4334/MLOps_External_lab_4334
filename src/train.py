from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from .features import NUMERIC,CATEGORICAL,TARGET,build_preprocessor

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'

def train_models(feature_csv=ART/'feature_store.csv'):
    ART.mkdir(exist_ok=True); df=pd.read_csv(feature_csv)
    X=df[NUMERIC+CATEGORICAL]; y=df[TARGET]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    candidates={'Logistic Regression':LogisticRegression(max_iter=1000,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=180,max_depth=8,random_state=42,class_weight='balanced')}
    rows=[]; fitted={}
    for name,est in candidates.items():
        pipe=Pipeline([('preprocessor',build_preprocessor()),('model',est)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        row={'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)}
        rows.append(row); fitted[name]=pipe
    metrics=pd.DataFrame(rows).sort_values(['f1','recall'],ascending=False).reset_index(drop=True)
    metrics.to_csv(ART/'model_comparison.csv',index=False)
    best_name=metrics.iloc[0]['model']; best=fitted[best_name]; joblib.dump(best,ART/'omr_rejection_model.joblib')
    (ART/'model_metadata.json').write_text(json.dumps({'selected_model':best_name,'positive_label':'Manual Verification Required','features':NUMERIC+CATEGORICAL},indent=2))
    try:
        import mlflow, mlflow.sklearn
        mlflow.set_tracking_uri(f'sqlite:///{ROOT}/mlflow.db'); mlflow.set_experiment('omr-rejection-prediction')
        for row in rows:
            with mlflow.start_run(run_name=row['model']):
                mlflow.log_params({'model':row['model'],'dataset':'omr_scans_v1+v2','test_size':.25})
                mlflow.log_metrics({k:row[k] for k in ['accuracy','precision','recall','f1']})
                mlflow.sklearn.log_model(fitted[row['model']], 'model', serialization_format='pickle')
        with mlflow.start_run(run_name='selected_model_registry') as run:
            mlflow.log_metrics({k:float(metrics.iloc[0][k]) for k in ['accuracy','precision','recall','f1']})
            mlflow.sklearn.log_model(best,'model',registered_model_name='OMR_Rejection_Model',serialization_format='pickle')
            (ART/'mlflow_registry_evidence.txt').write_text(f'Registered model: OMR_Rejection_Model\nSelected run: {run.info.run_id}\nSelected candidate: {best_name}\n')
    except Exception as e:
        (ART/'mlflow_registry_evidence.txt').write_text('MLflow registry unavailable during this run: '+repr(e)+'\nInstall requirements and rerun pipeline to enable registry.')
    return metrics

if __name__=='__main__': print(train_models().to_string(index=False))
