from unittest.mock import patch, MagicMock
import torch
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medical_assistant_package.medical_functions import (
    disease_predict,
    match_symptoms,
    doctor_recommend,
    disease_advice,
    get_all_cosine_similarities,
    get_id_for_most_similar_item
)
dummy_text = ["fever", "cough"]
dummy_disease_name_df = pd.DataFrame({"Disease": ["Flu", "Cold"], "label": [0, 1]})
dummy_disease_df = pd.DataFrame({"label": [0, 1]})

def test_disease_predict(monkeypatch):
    import numpy as np
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    monkeypatch.setattr(
        "medical_assistant_package.medical_functions.lm.load_setfit_model",
        lambda: mock_model
    )

    predicted = disease_predict(dummy_text, dummy_disease_name_df, dummy_disease_df)

    if isinstance(predicted, np.ndarray):
        predicted = predicted[0]

    assert isinstance(predicted, str), "The return value should be a string"
    assert predicted == "Flu"

def test_match_symptoms():
    feature_names = ["fever", "headache", "cough"]
    user_inputs = ["fever", "dizziness"]
    matches = match_symptoms(user_inputs, feature_names, threshold=0.1)
    assert "fever" in matches, "'fever' should be matched"
    assert "dizziness" not in matches, "'dizziness' should NOT be matched"

@patch("medical_assistant_package.medical_functions.load_doctor_model")
def test_doctor_recommend(mock_load_model):
    class DummyModel:
        def __call__(self, x):
            return torch.tensor([[0.1, 0.9, 0.0]])
    mock_load_model.return_value = DummyModel()
    prompts = ["fever"]
    doctors = doctor_recommend(prompts)
    assert len(doctors) == 3, "Should return top 3 doctors"

def test_disease_advice():
    medical_advice_df = pd.DataFrame({
        "Disease": ["Flu"],
        "Advice1": ["Rest"],
        "Advice2": ["Hydrate"]
    })
    advice = disease_advice("Flu", medical_advice_df)
    assert "Rest" in advice, "Advice should contain 'Rest'"

def test_get_all_cosine_similarities():
    a = np.array([[1, 0], [0, 1]])
    b = np.array([1, 0])
    sims = get_all_cosine_similarities(a, b)
    np.testing.assert_almost_equal(sims, np.array([1.0, 0.0]), err_msg="Cosine similarities mismatch")

def test_get_id_for_most_similar_item():
    sims = np.array([0.1, 0.9, 0.5])
    ids = ["id1", "id2", "id3"]
    best_id = get_id_for_most_similar_item(sims, ids)
    assert best_id == "id2", f"Expected 'id2', got {best_id}"