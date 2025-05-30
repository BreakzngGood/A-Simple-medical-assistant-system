# Switch the pages
import streamlit as st

def init_step_state(default_step=1):
    """Initialise the step state in st.session_state"""
    if "step" not in st.session_state:
        st.session_state["step"] = default_step


def goto_step(step_num: int):
    st.session_state["step"] = step_num


def form_review_field(prompt_question, original_value, field_key, doctor_diagnosis_info, input_label):
    user_feedback = st.radio(prompt_question, ("Yes", "No"), key=field_key + "_radio")
    if user_feedback == "Yes":
        doctor_diagnosis_info[field_key] = original_value
    else:
        doctor_diagnosis_info[field_key] = st.text_input(label=input_label, key=field_key + "_input")
