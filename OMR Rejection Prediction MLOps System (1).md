# OMR Rejection Prediction MLOps System

This repository implements an end-to-end MLOps system for predicting whether an automated OMR scanner will reject an answer sheet and require manual verification. It includes synthetic dataset generation (Version 1 and Version 2), preprocessing and feature reuse, Logistic Regression and Random Forest comparison, MLflow experiment/registry evidence, a FastAPI service, pytest validation, GitHub Actions CI, an architecture diagram, and a Colab-ready notebook.

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data
python -m src.pipeline
pytest -q
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs` for the interactive API.

## Example request
```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"dpi":300,"rotation_angle":4.2,"print_contrast_score":52,"alignment_marker_visibility":0.55,"multiple_marked_questions":4,"blank_answer_percentage":18,"sheet_damage_score":0.35,"scanner_type":"legacy"}'
```
Expected response: `Manual Verification Required` for a poor-quality scan.

## Project map
- `omr_mlops_colab.ipynb`: complete notebook for Google Colab/local execution.
- `src/generate_data.py`: deterministic V1/V2 synthetic data generator.
- `src/features.py`: preprocessing and reusable feature-store generation.
- `src/train.py`: model training, metrics, artifacts and MLflow logging/registration.
- `src/pipeline.py`: one-command orchestration.
- `src/api.py`: FastAPI prediction endpoint.
- `tests/`: input-range, feature, and prediction tests.
- `docs/implementation.pdf`: implementation documentation with results and evidence.
- `docs/architecture.mmd`: source architecture diagram.

## Data and label logic
The target is generated from meaningful risk drivers: rotation, low contrast, poor marker visibility, multiple marks, blanks, sheet damage, lower DPI, and scanner type. Version 2 deliberately shifts the population toward larger misalignment and lower contrast to represent scanner/process degradation. A deterministic seed makes the experiment reproducible.

## MLflow
The pipeline writes a local MLflow tracking directory to `mlruns/`, logs both candidates, and registers the selected model as `OMR_Rejection_Model`. If MLflow is not available, the pipeline still writes metrics and a joblib model; the requirements file enables the full registry path.
