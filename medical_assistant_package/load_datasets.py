import streamlit as st
import pandas as pd
import medical_assistant_package.config as cfg

# -------------------------------------------------
# Load datasets with st.cache_resource which can Speed ​​up operation
# Reference from https://docs.streamlit.io/develop/concepts/architecture/caching
# -------------------------------------------------


@st.cache_data
def load_data():
    try:
        prompt_df = pd.read_csv(cfg.PROMPT_CSV)
        prompt_df["Symptom"] = prompt_df["Symptom"].str.replace("_", " ", regex=False)
        prompt_list = prompt_df["Symptom"].dropna().tolist()

        disease_df = pd.read_csv(cfg.DISEASE_CSV)
        disease_name_df = disease_df["Disease"]

        medical_advice_df = pd.read_csv(cfg.MEDICAL_ADVICE_CSV)

        medicine_df = pd.read_csv(cfg.MEDICINE_CSV)

        Doctor_df = pd.read_excel(cfg.SPECIALIST_EXCEL_PATH)
        all_symptoms = Doctor_df.drop(["Disease", "Unnamed: 0"], axis=1).columns.tolist()

        return {
        "prompt_df": prompt_df,
        "prompt_list": prompt_list,
        "disease_df": disease_df,
        "disease_name_df": disease_name_df,
        "medical_advice_df": medical_advice_df,
        "medicine_df": medicine_df,
        "all_symptoms": all_symptoms
    }
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error while reading data: {e}")
        return None, None, None, None, None, None, None
    
