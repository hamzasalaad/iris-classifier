import os

def test_outputs_exist():
    assert os.path.exists("outputs/model.joblib")
    assert os.path.exists("outputs/confusion_matrix.png")