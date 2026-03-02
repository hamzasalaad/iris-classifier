# Iris Classifier Project

This project demonstrates a simple machine learning workflow using the Iris dataset from scikit-learn.

It includes:
- A Jupyter notebook walkthrough
- A reproducible training script (train.py)
- Saved outputs (model + confusion matrix)
- Basic unit testing

---

## Setup (Windows)

1. Create virtual environment:
   python -m venv venv

2. Activate environment:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

---

## Run Training

python src/train.py

This generates:
- outputs/confusion_matrix.png
- outputs/model.joblib

---

## Run Tests

pytest tests/test_train.py