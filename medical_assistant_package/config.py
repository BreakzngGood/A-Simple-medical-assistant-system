import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------
# Models
# -------------------
WHISPER_MODEL_NAME = "turbo"
SETFIT_MODEL_PATH = "ckpt_new/"
OPENCLIP_MODEL_TYPE = "ViT-B-32"
DOCTOR_MODEL_PATH = "Specalist.pkl"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
IMAGE_CLASSIFICATION_MODEL = "best_resnet18_model.pt"
SKIN_DISEASE_MODEL = "best_skin_disease_model.pt"

# -------------------
# Embeddings
# -------------------
MET_EMBEDDINGS_PATH = BASE_DIR / "demo" / "embeddings" / "met_embeddings.npy"
MET_EMBEDDING_IDS_PATH = BASE_DIR / "demo" / "embeddings" / "met_embedding_ids.npy"

# -------------------
# Datasets
# -------------------
SPECIALIST_EXCEL_PATH = BASE_DIR / "Datasets" / "Specialist.xlsx"
PROMPT_CSV = BASE_DIR / "Datasets" / "Symptom" / "Symptom-severity.csv"
DISEASE_CSV = BASE_DIR / "Datasets" / "Cleaned_Disease_with_label.csv"
MEDICAL_ADVICE_CSV = BASE_DIR / "Datasets" / "Symptom" / "symptom_precaution.csv"
MEDICINE_CSV = BASE_DIR / "Datasets" / "Medicine_Details.csv"
SKIN_DISEASE_TRAIN = BASE_DIR / "Datasets" / "SkinDisease" / "train"
SKIN_DISEASE_TEST = BASE_DIR / "Datasets" / "SkinDisease" / "train"
DISEASE_PREDICTION_TRAIN = BASE_DIR / "Datasets" /"train.csv"
DISEASE_PREDICTION_TEST = BASE_DIR / "Datasets" / "test.csv" 

# -------------------
# Encoder
# -------------------
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"