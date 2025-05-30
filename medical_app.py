import streamlit as st
import json
import glob

import torch
import torch.nn.functional as F

import os
from PIL import Image
import open_clip
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# from Image_classifier import ClassificationNetwork
from model_train.image_classifier_from_pretrained import ClassificationNetwork
from model_train.doctor_recommendation_architecture import SpecialistNN
from datetime import datetime
from pandas import json_normalize

import medical_assistant_package.load_models as lm
import medical_assistant_package.load_datasets as ld
import medical_assistant_package.streamlit_navigation as sn
import medical_assistant_package.audio_utils as au
import medical_assistant_package.file_save as fs
import medical_assistant_package.medical_functions as mf
import medical_assistant_package.text_utils as tu
import medical_assistant_package.config as cfg
 
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

data = ld.load_data()

# Create dictionaries to save patient's medical record and doctor's diagnosis
file_list = []
medical_record_info = {
        "name": None,
        "biological_sex": None,
        "age_info": None,
        "ethnicity": None,
        "family_history": None,
        "email_info": None,
        "symptom_list": None,
        "predict_disease": None,
        "doctor_recommendation": None,
        "disease_advice": None,
        "disease_severity": None,
        "medicine_name" : None
    }

doctor_diagnosis_info = {
        "name": None,
        "symptom_list": None,
        "predict_disease": None,
        "doctor_recommendation": None,
        "disease_advice": None,
        "disease_severity": None,
        "medicine_name" : None,
        "rating" : None,
        "opinion" : None
}

# -------------------------------------------------
# Main page (Page 1)
# -------------------------------------------------
def page_main_page():
    st.title("🏥 Medical Assistant System")
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>➡️📝 Medical Record</h2>", unsafe_allow_html=True) 
    
    # Collect the patient's Medical record by form
    with st.form(key="form1"):
        medical_record_info["name"] = st.text_input(label="Name")
        
        medical_record_info["biological_sex"] = st.selectbox(
            label="Biological Sex:",
            options=["Male", "Female", "Intersex"]
        )
        medical_record_info["age_info"] = st.slider(label="age", min_value=0, max_value=100)
        
        medical_record_info["ethnicity"] = st.selectbox(
            label="Ethnicity:",
            options=["White / Caucasian", "Black / African American", "Asian","Hispanic / Latino","Native American / Alaska Native","Middle Eastern / North African","Native Hawaiian / Pacific Islander","Mixed / Multiracial","Other","Prefer not to say"]
        )
        
        family_history = st.radio("Do you have a family medical history?", ("Yes", "No"))
        
        if family_history == "Yes":
            medical_record_info["family_history"] = st.multiselect(
                label="Please select ( Muti_selection ):",
                options=["Cardiovascular Diseases", "Neurological Disorders", "Oncological Disorders","Respiratory Diseases","Psychiatric Disorders","Autoimmune Diseases","Genetic & Hereditary Disorders"]
        )


        medical_record_info["share_info"] = st.radio(label="share my information:", options=["yes", "no"])
        medical_record_info["email_info"] = st.checkbox(label="Receive Email notification")
        # submit button
        form_submitted = st.form_submit_button(label="Submit!")

        def check_form_valid():
            required_fields = ["name", "biological_sex", "age_info"]
            return all(medical_record_info[field] not in [None, ""] for field in required_fields)  
            # vals = medical_record_info.values()
            # return all([True if val not in [None, ""] else False for val in vals ])
        
        # check format
        if form_submitted:
            if not check_form_valid():
                st.warning("Please fill in all required values : 'name', 'biological_sex', 'age' !")
            else:
                # st.balloons()
                st.write("form submitted!")

    st.divider()
    
    # st.subheader("Skin disease diagnosis")
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>🧑🏽‍🦱 Skin Disease Diagnosis</h2>", unsafe_allow_html=True)
    st.write("If you think you have a skin problem, please click the 'Go to skin disease'")
    st.button(label="Go to skin disease", on_click=sn.goto_step, args=(2,))
    
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>🩹 Symptom Diagnosis</h2>", unsafe_allow_html=True)
        
    symptom_input = st.text_input(
        label="Please input your symptoms or Click the button to record your symptoms : ",
        )

    col1, col2, col3 = st.columns([1, 1, 1],gap="small")
    run = False

    # split the page space
    with col1:
        if st.button("Record and Chat"): 
            audio, samplerate = au.record_audio()
            audio = au.preprocess_audio(audio)
            whisper_model = lm.load_whisper()
            result = whisper_model.transcribe(audio)
            user_text = result['text'] 
            st.write("You said:", user_text)
    

    # Predict patient's symptoms, disease, and recommend the proper doctor, medicine, offer advice for the disease and its severity
    with col2:
        if st.button("Submit"): 
            if not symptom_input.strip():  
                st.warning("Input cannot be empty! Please enter your symptoms.")
            else:
                run = True
                st.success(f"Submitted Symptoms: {symptom_input}")
                
                similar_prompts , prob_prompt_list = tu.extract_prompts_from_dataset(symptom_input)
                medical_record_info["symptom_list"] = prob_prompt_list
    if run:
        st.subheader("🔍 Symptom prediction")
        if medical_record_info["symptom_list"]:
            prob_prompt_str = ", ".join(medical_record_info["symptom_list"])
            
            n = 0 
            for i in similar_prompts:
                n += 1
                symptom, score = i
                st.write(f"Your possible main symptom {n} is: {symptom}, {score *100:.2f}%.")

            st.subheader("⚕️ Disease Prediction")
            predict_disease = mf.disease_predict(prob_prompt_str)
            st.write(f"Your possible disease are: {predict_disease}")
            medical_record_info["predict_disease"] = predict_disease
            
            st.subheader("👩‍⚕️ Doctor Recommendation")
            # st.write(medical_record_info["symptom_list"])
            doctor = mf.doctor_recommend(medical_record_info["symptom_list"])
            
            st.write(f"The best Recommend doctor is : {doctor[0]}")
            st.write(f"You may also choose to see : {doctor[1]}, {doctor[2]}")
            medical_record_info["doctor_recommendation"] = doctor
            
            st.subheader("📋 Disease Advice")
            advice = mf.disease_advice(predict_disease)
            st.write(f"{advice}")
            medical_record_info["disease_advice"] = advice

            st.subheader("🤕 Disease Severity")
            severity = mf.disease_severity(medical_record_info["symptom_list"])
            st.write(f"Medical severity: {severity}.")
            if severity > 5:
                st.write("Maybe you should See a doctor as soon as possible.")
            else :
                st.write("Pay attention to rest, take medicine under doctor's orders, and observe the situation.")

            medical_record_info["disease_severity"] = severity
            
            st.subheader("💉 Medicine Use")
            meidcine_col1, meidcine_col2 = st.columns([1, 1],gap="small")
            medicine_idx = mf.medicine_use(predict_disease)
            match_medicine_name = []
            # st.write("Top 3 matching Medicine:")
            with meidcine_col1:
                for idx in medicine_idx:
                    best_match_medicine = data["medicine_df"].iloc[idx]
                    st.write("-" * 50)
                    st.write(f"Name: {best_match_medicine['Medicine Name']}")
                    match_medicine_name.append(best_match_medicine['Medicine Name'])
                    st.write(f"Use: {best_match_medicine['Uses']}")
                    st.write(f"Composition: {best_match_medicine['Composition']}")
                    st.write(f"Side_effects: {best_match_medicine['Side_effects']}")
                    st.write("-" * 50)
                medical_record_info["medicine_name"] = match_medicine_name
                filename = fs.save_to_json(medical_record_info)
                file_list.append(filename)

                st.success(f"Medical record formed and saved as {filename}!")    
            with meidcine_col2:  
                for name in medical_record_info["medicine_name"]:
                    img_path = mf.medicine_picture_by_name(name)
                    st.image(img_path, caption=name, use_container_width=True)
        # st.divider()
    
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>➡️🧑‍⚕️ Doctor Diagnosis</h2>", unsafe_allow_html=True)    
    st.write("This button is for the doctor to check users' Medical record, and recheck the diagnosis.")
    st.button(label="Go to doctor diagnosis", on_click=sn.goto_step, args=(4,))
    
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>➡️💊 Medicine Search</h2>", unsafe_allow_html=True)  
    st.write("If you know your disease and forget which kind of medicine you need to take in:")
    st.button(label="Go to medicine search", on_click=sn.goto_step, args=(3,))            

# Page 2 : Skin disease prediction by Image 
def page_skin_disease():
    device = "cpu"
    num_classes = 22  
    model = ClassificationNetwork(num_classes=num_classes)
    model.load_state_dict(torch.load(cfg.IMAGE_CLASSIFICATION_MODEL, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ImageFolder(cfg.SKIN_DISEASE_TRAIN)  
    class_names = dataset.classes  

    def predict_image(image_path):
        image = Image.open(image_path).convert("RGB")  
        image = transform(image).unsqueeze(0).to(device) 

        with torch.no_grad():  
            output = model(image)  
            probabilities = F.softmax(output, dim=1)  
            confidence, predicted = torch.topk(probabilities, k=3, dim=1) 

        top3_classes = [class_names[idx] for idx in predicted[0].cpu().numpy()]
        top3_confidences = [round(float(conf), 2) for conf in confidence[0].cpu().numpy()]

        return top3_classes,top3_confidences

    st.title("🩺 Skin Disease Diagnosis")
    user_name = st.text_input(label= "Name")
    if st.button("Submit"): 
        if not user_name.strip():  
            st.warning("Input cannot be empty! Please enter your NAME.")
        else:
            st.success(f"Submitted NAME: {user_name}")

    SAVE_DIR = os.path.join("skin_disease_images", user_name)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    st.title("📸 Streamlit Camera App")
    st.write("Press 'Take Photo' to capture an image.")

    # camera input
    captured_image = st.camera_input("Take a picture")

    if captured_image:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captured_image_path = os.path.join(SAVE_DIR, f"image_{timestamp}.jpg")

        image = Image.open(captured_image)
        image.save(captured_image_path)
        
        st.success(f"✅ Image saved at: `{captured_image_path}`")
        st.image(image, caption="Captured Image", use_container_width=True)

        top3_classes,top3_confidences = predict_image(captured_image_path)
        st.write("Top 3 Predictions:\n")
        for i,j in zip(top3_classes,top3_confidences):
            st.write(f"possible disease is {i}, the possibility is {j*100:.2f} %")


    st.title("🖼️ Upload and Display Image")

    # Upload img
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        uploaded_img_path = os.path.join(SAVE_DIR, uploaded_file.name)
        image.save(uploaded_img_path)
        st.image(image, caption="Uploaded Image", use_container_width=True)  
        st.success("✅ Image uploaded successfully!")

        top3_classes,top3_confidences = predict_image(uploaded_img_path)
        st.write("Top 3 Predictions:\n")
        for i,j in zip(top3_classes,top3_confidences):
            st.write(f"possible disease is {i}, the possibility is {j*100:.2f} %")

    st.button(label="Go to main step", on_click=sn.goto_step, args=(1,))

# Page 3 : Medicine Search by Text
def page_medicine_search():
    st.title("Medicine Search")
    meidicine_description = st.text_input(
        label="Please input the description of your medicine (e.g. Color of package, pills or bottle,etc) : ",
        )
    
    if st.button("Submit"): 
        if not meidicine_description.strip():  
            st.warning("Input cannot be empty! Please enter your symptoms.")
        else:
            st.success(f"Submitted description: {meidicine_description}")
            text_tensor = open_clip.tokenizer.tokenize(meidicine_description)
            with torch.no_grad():
                opencilp_model, preprocess = lm.load_openclip_model()
                text_embedding = opencilp_model.encode_text(text_tensor).squeeze().numpy()
            
            met_embeddings, id_list = lm.load_embeddings()
            cosine_sim_array = mf.get_all_cosine_similarities(met_embeddings, text_embedding)
            closest_match_id = mf.get_id_for_most_similar_item(cosine_sim_array, id_list)

            image_path = f'Datasets/Medicine_Picture/{closest_match_id}.jpg'
            closest_match = Image.open(image_path).convert("RGB")
            st.write(f"closest match for text string: '{meidicine_description}' is the image:")
            st.image(image_path, caption=f"{closest_match_id}", use_container_width=True)
            st.button(label="Go to main page", on_click=sn.goto_step, args=(1,))

# Page 4 : Doctor diagnosis
def page_doctor_diagnosis():
    def get_latest_json(folder="Medical_record"):
        try:
            json_files = glob.glob(os.path.join(folder, "*.json"))
            if not json_files:
                return None
            latest_file = max(json_files, key=os.path.getctime)
            return latest_file
        except Exception as e:
            st.error(f"Error while retrieving the latest JSON file: {e}")
            return None

    latest_json = get_latest_json()
    st.title("🩺 Doctor Diagnosis")

    if latest_json is None:
        st.warning("No medical record JSON file found. Please save a record first.")
        return

    try:
        with open(latest_json, "r", encoding="utf-8") as f:
            diagnostic_data = json.load(f)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return
    except json.JSONDecodeError:
        st.error("JSON format error. Unable to parse the file.")
        return
    except Exception as e:
        st.error(f"Unexpected error while reading the file: {e}")
        return

    st.json(diagnostic_data)

    try:
        medical_info_df = json_normalize(diagnostic_data)
        st.dataframe(medical_info_df)
    except Exception as e:
        st.error(f"Error converting data to table format: {e}")
        return

    try:
        doctor_diagnosis_info["name"] = diagnostic_data["name"]
    except KeyError:
        st.warning("The diagnosis data is missing the 'name' field.")
    except Exception as e:
        st.error(f"Error processing diagnosis information: {e}")
        
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>Doctor Diagnosis</h1>", unsafe_allow_html=True)
    st.divider()
    with st.form(key="form2"):

        sn.form_review_field(
            prompt_question="Do you think this is an accurate symptom list?",
            original_value=diagnostic_data["symptom_list"],
            field_key="symptom_list",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the symptom:"
        )

        sn.form_review_field(
            prompt_question="Do you think this is an accurate disease prediction?",
            original_value=diagnostic_data["predict_disease"],
            field_key="predict_disease",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the disease:"
        )

        sn.form_review_field(
            prompt_question="Do you think this is an accurate doctor recommendation?",
            original_value=diagnostic_data["doctor_recommendation"],
            field_key="doctor_recommendation",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the doctor:"
        )

        sn.form_review_field(
            prompt_question="Do you think these are accurate disease advices?",
            original_value=diagnostic_data["disease_advice"],
            field_key="disease_advice",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the disease advices:"
        )

        sn.form_review_field(
            prompt_question="Do you think this is an accurate severity prediction?",
            original_value=diagnostic_data["disease_severity"],
            field_key="disease_severity",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the severity:"
        )

        sn.form_review_field(
            prompt_question="Do you think this is a correct medicine recommendation?",
            original_value=diagnostic_data["medicine_name"],
            field_key="medicine_name",
            doctor_diagnosis_info=doctor_diagnosis_info,
            input_label="Please input the medicine name:"
        )
        doctor_diagnosis_info["rating"] = st.slider(label="rating to the AI diagnosis", min_value=0, max_value=10)
        
        doctor_diagnosis_info["opinion"] = st.text_input(label="Please write your opinions:")

        # submit button
        form_submitted = st.form_submit_button(label="Submit!")

        def check_form_valid():
            return all(val and str(val).strip() for val in doctor_diagnosis_info.values())
        
        # check format
        if form_submitted:
            if not check_form_valid():
                st.warning("Please fill in all values.")
            else:
                # st.balloons()
                st.write("form submitted!")
                filename = fs.save_to_json(doctor_diagnosis_info)
                st.success(f"Doctor diagnosis formed and saved as {filename}!")

    st.button(label="Go to main page", on_click=sn.goto_step, args=(1,))

def main():
    sn.init_step_state()
    step = st.session_state.step
    if step == 1:
        page_main_page()
    elif step == 2:
        page_skin_disease()
    elif step == 3:
        page_medicine_search()
    elif step == 4:
        page_doctor_diagnosis()
    else:
        st.error("Unknown step")

if __name__ == "__main__":
    main()