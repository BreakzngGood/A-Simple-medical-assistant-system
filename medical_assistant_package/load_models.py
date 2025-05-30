import streamlit as st
import numpy as np
import pandas as pd
import open_clip
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from setfit import SetFitModel
import whisper
import pickle
import medical_assistant_package.config as cfg

# -------------------------------------------------
# Load models with st.cache_resource which can Speed ​​up operation
# Reference from https://docs.streamlit.io/develop/concepts/architecture/caching
# -------------------------------------------------

# @st.cache_data
def load_whisper():
    return whisper.load_model(cfg.WHISPER_MODEL_NAME)

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer(cfg.SENTENCE_TRANSFORMER_MODEL)

@st.cache_resource
def load_setfit_model():
    return SetFitModel.from_pretrained(cfg.SETFIT_MODEL_PATH, local_files_only=True)

@st.cache_resource
def load_openclip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(cfg.OPENCLIP_MODEL_TYPE, pretrained="openai")
    return model, preprocess

@st.cache_resource
def load_doctor_model():
    return pickle.load(open(cfg.DOCTOR_MODEL_PATH, "rb"))

# opencilp_model.eval()
def load_embeddings():
    met_embeddings = np.load(cfg.MET_EMBEDDINGS_PATH)
    met_ids = np.load(cfg.MET_EMBEDDING_IDS_PATH)
    id_list = met_ids.tolist()
    return met_embeddings, id_list

@st.cache_data
def load_symptom_feature():
    df = pd.read_excel(cfg.SPECIALIST_EXCEL_PATH)
    x = df.drop(['Disease', 'Unnamed: 0'], axis = 1)
    feature_names = x.columns.tolist()
    return feature_names

@st.cache_data
def load_label_encoder():
    with open(cfg.LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder   
