import streamlit as st
import json
import glob

import torch
import torch.nn.functional as F
import whisper


import sounddevice as sd
import numpy as np
import pandas as pd
import librosa
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from setfit import SetFitModel

import string
import pickle
import os
import numpy as np
from PIL import Image
import open_clip
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
# from Image_classifier import ClassificationNetwork
from Image_classifier_from_pretrained import ClassificationNetwork
from Doctor_recommendation_architecture import SpecialistNN
from datetime import datetime
from pandas import json_normalize
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# -------------------------------------------------
# Load datasets and models with st.cache_resource which can Speed ​​up operation
# Reference from https://docs.streamlit.io/develop/concepts/architecture/caching
# -------------------------------------------------

# @st.cache_data
def load_whisper():
    return whisper.load_model("turbo")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_setfit_model():
    return SetFitModel.from_pretrained("ckpt_new/", local_files_only=True)

@st.cache_resource
def load_openclip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model, preprocess

@st.cache_resource
def load_doctor_model():
    return pickle.load(open("Specalist.pkl", "rb"))

# opencilp_model.eval()
def load_embeddings():
    met_embeddings = np.load("demo/embeddings/met_embeddings.npy")
    met_ids = np.load("demo/embeddings/met_embedding_ids.npy")
    id_list = met_ids.tolist()
    return met_embeddings, id_list

@st.cache_data
def load_data():

    prompt_df = pd.read_csv(r"Datasets\Symptom\Symptom-severity.csv")
    prompt_df["Symptom"] = prompt_df["Symptom"].str.replace("_", " ", regex=False)
    prompt_list = prompt_df["Symptom"].dropna().tolist()

    disease_df = pd.read_csv(r"Datasets\Cleaned_Disease_with_label.csv")
    disease_name_df = disease_df["Disease"]

    medical_advice_df = pd.read_csv(r"Datasets\Symptom\symptom_precaution.csv")

    medicine_df = pd.read_csv(r"Datasets\Medicine_Details.csv")

    Doctor_df = pd.read_excel(r"Datasets\Specialist.xlsx")
    all_symptoms = Doctor_df.drop(["Disease", "Unnamed: 0"], axis=1).columns.tolist()

    return prompt_df,prompt_list,disease_df,disease_name_df, medical_advice_df, medicine_df, all_symptoms

prompt_df,prompt_list, disease_df,disease_name_df, medical_advice_df, medicine_df, all_symptoms = load_data()

@st.cache_data
def load_symptom_feature():
    df = pd.read_excel("Datasets\Specialist.xlsx")
    x = df.drop(['Disease', 'Unnamed: 0'], axis = 1)
    feature_names = x.columns.tolist()
    return feature_names

feature_names = load_symptom_feature()
# Create dictionaries to save patient's medical record and doctor's diagnosis

@st.cache_data
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder   

label_encoder = load_label_encoder()

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

# st.session_state.symptom_input = ""
# symptom_input = ''

if "step" not in st.session_state:
    st.session_state["step"] = 1

# Switch the pages
def goto_step(step_num: int):
    st.session_state["step"] = step_num
    # st.write(text)

def record_audio(duration= 8, samplerate=44100):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    st.write('recording')
    sd.wait()
    st.write('recording complete')
    return audio, samplerate

def preprocess_audio(audio_input, original_sr = 44100, target_sr = 16000):
    audio_input = audio_input.squeeze()
    audio_input = audio_input.astype(np.float32) / 32768.0
    audio_input = librosa.resample(audio_input, orig_sr=original_sr, target_sr=target_sr)
    audio_input = whisper.pad_or_trim(audio_input)
    return audio_input

# Nltk to process the user's input
def preprocess_text_nltk(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)
    
    # Remove stop words (such as "the", "is", "and")
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Get the prompts of symptoms by user's input
def extract_prompts_from_dataset(text):
    
    # preprocess user'input text
    text = preprocess_text_nltk(text)

    # Calculate the similarity between text and the prompt dataset
    Sentence_model = load_sentence_transformer()
    text_embedding = Sentence_model.encode(text, convert_to_tensor=True)
    prompt_embeddings = Sentence_model.encode(prompt_list, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(text_embedding, prompt_embeddings)[0].cpu().numpy() 
    
    # Filter out prompts with similarity > 0.5.
    similar_prompts = sorted(
        [(prompt, score) for prompt, score in zip(prompt_list, cosine_scores) if score > 0.5],
        key=lambda x: x[1], reverse=True
    )
    
    if not similar_prompts:
        st.write("No symptom found")
        return None, None
    
    prob_prompt_list = [prompt for prompt, score in similar_prompts]

    return similar_prompts , prob_prompt_list

# predict the disease by symptoms
def disease_predict(text):
    setfit_model = load_setfit_model()
    output_probs = setfit_model.predict_proba(text)
    output_probs = output_probs.tolist()
    max_conf = max(output_probs)

    max_class = output_probs.index(max_conf)
    predict_disease = disease_name_df[disease_df["label"] == max_class].values[0]
    print(predict_disease)
    return predict_disease

# predict the proper doctor
def load_doctor_model():
    input_size = len(feature_names)  
    output_size = len(label_encoder.classes_) 
    model = SpecialistNN(input_size, output_size)
    model.load_state_dict(torch.load("specialist_nn.pth"))
    model.eval()
    return model

def match_symptoms(user_inputs, feature_names, threshold=0.2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(feature_names + user_inputs)  # Caculate the TF-IDF matrix

    feature_vectors = tfidf_matrix[:len(feature_names)]  
    user_vectors = tfidf_matrix[len(feature_names):]  

    # Caculate cosine similarity
    cos_sim = cosine_similarity(user_vectors, feature_vectors)

    matched_symptoms = []
    for i, user_symptom in enumerate(user_inputs):
        best_match_idx = np.argmax(cos_sim[i])  # Get the most similar symptom index
        best_match = feature_names[best_match_idx] if cos_sim[i, best_match_idx] > threshold else None  
        if best_match:
            matched_symptoms.append(best_match)

    return matched_symptoms

# Netural network recommendation
def doctor_recommend(prompts):
    matched_symptoms = match_symptoms(prompts, feature_names)  
    # if not matched_symptoms:
    #     return ["No matching symptoms found"]

    input_symptoms = pd.DataFrame(0, index=[0], columns=feature_names)
    input_symptoms.loc[0, matched_symptoms] = 1  

    input_tensor = torch.tensor(input_symptoms.values, dtype=torch.float32)

    model = load_doctor_model()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]  

    top_3_indices = np.argsort(probabilities)[-3:][::-1]  
    top_3_doctors = label_encoder.inverse_transform(top_3_indices)  

    return top_3_doctors

# Tradtional NLP doctor recommendation
# def doctor_recommend(prompts):

#     matched_symptoms = match_symptoms(prompts, feature_names)  # match userinput
#     # st.write(f"{matched_symptoms}")
#     # Create One-Hot encoded input data
#     input_symptoms = pd.DataFrame(0, index=[0], columns=feature_names)
#     input_symptoms.loc[0, matched_symptoms] = 1  # Only fill in the matched symptoms

#     # predict
#     Doctor_model = load_doctor_model()
#     probabilities = Doctor_model.predict_proba(input_symptoms)[0]  
#     top_3_indices = probabilities.argsort()[-3:][::-1]  
#     top_3_doctors = Doctor_model.classes_[top_3_indices]  

#     return top_3_doctors

# Return disease advice from the dataset, according to predict_disease
def disease_advice(predict_disease):
    disease_row = medical_advice_df[medical_advice_df["Disease"] == predict_disease]

    if not disease_row.empty:
        
        precautions = disease_row.iloc[0, 1:].dropna().tolist()  
        
        precautions_str = ", ".join(precautions)

        advice = (f"Medical advice: {precautions_str}")
        return advice
    else:
        advice = ("Did not find the advices to this disease")
        return advice

# Caculate the disease severity
def disease_severity(prompt_list):
    len_prompt_list = len(prompt_list)
    severity_degree = 0
    for prompt in prompt_list:
        weight = prompt_df.loc[prompt_df["Symptom"] == prompt, "weight"]
        weight = weight.values[0]
        severity_degree += weight
    severity_degree = round(severity_degree/len_prompt_list,2)
    print(severity_degree)
    return severity_degree

# medicine recommendation by the usage
def medicine_use(text):

    medicine_uses_texts = medicine_df["Uses"].fillna("")

    Sentence_model = load_sentence_transformer()
    uses_embeddings = Sentence_model.encode(medicine_uses_texts.tolist(), convert_to_tensor=True)
    input_embedding = Sentence_model.encode(text, convert_to_tensor=True)

    # Calculate similarity
    cosine_sim = util.pytorch_cos_sim(input_embedding, uses_embeddings)

    top_matches_idx = cosine_sim.argsort(descending=True)[0][:3].tolist()

    # Output the most matching medicine information
    # print("Top 3 matching Medicine:")
    # for idx in top_matches_idx:
    #     best_match_medicine = medicine_df.iloc[idx]
    #     print("-" * 50)
    #     print(f"Name: {best_match_medicine['Medicine Name']}")
    #     print(f"Use: {best_match_medicine['Uses']}")
    #     print(f"Composition: {best_match_medicine['Composition']}")
    #     print(f"Side_effects: {best_match_medicine['Side_effects']}")
    #     print("-" * 50)
    return top_matches_idx

def medicine_picture_by_name(name):
    image_path = f'Datasets/Medicine_Picture/{name}.jpg'
    print (image_path)
    return image_path

# search medicine by text 
def get_clip_embedding_from_PIL_image(image):
    opencilp_model, preprocess = load_openclip_model()
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = opencilp_model.encode_image(image_tensor).squeeze().numpy()
    return embedding

def get_all_cosine_similarities(embeddings_matrix, embedding_vector):
        dot_product = embeddings_matrix @ embedding_vector
        product_of_magnitudes = np.linalg.norm(embeddings_matrix, axis = 1) * np.linalg.norm(embedding_vector)
        return dot_product / product_of_magnitudes

def get_id_for_most_similar_item(similarity_array, id_list):
    highest_score_index = np.argmax(similarity_array)
    item_id = id_list[highest_score_index]
    return item_id


def save_to_json(record, folder="Medical_record"):
    
        if not os.path.exists(folder):
            os.makedirs(folder)

   
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/{record['name']}_{timestamp}.json"
        
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
    
        with open(filename, "w") as f:
            json.dump(record, f, indent=4)

        return filename 

# -------------------------------------------------
# Main page (Page 1)
# -------------------------------------------------
if st.session_state.step == 1:
    
    st.title("🏥 Medical Assistant System")
    # st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>Medical Assistant System</h1>", unsafe_allow_html=True)
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

    # print(medical_record_info)
    # medical_record_info
    st.divider()
    
    # st.subheader("Skin disease diagnosis")
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>🧑🏽‍🦱 Skin Disease Diagnosis</h2>", unsafe_allow_html=True)
    st.write("If you think you have a skin problem, please click the 'Go to skin disease'")
    st.button(label="Go to skin disease", on_click=goto_step, args=(2,))
    
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
            audio, samplerate = record_audio()
            audio = preprocess_audio(audio)
            whisper_model = load_whisper()
            result = whisper_model.transcribe(audio)
            user_text = result['text']
            
            # st.session_state.symptom_input = user_text
            # symptom_input = st.session_state.symptom_input
            # st.rerun()  
            st.write("You said:", user_text)
    

    # Predict patient's symptoms, disease, and recommend the proper doctor, medicine, offer advice for the disease and its severity
    with col2:
        if st.button("Submit"): 
            if not symptom_input.strip():  
                st.warning("Input cannot be empty! Please enter your symptoms.")
            else:
                run = True
                st.success(f"Submitted Symptoms: {symptom_input}")
                
                similar_prompts , prob_prompt_list = extract_prompts_from_dataset(symptom_input)
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
            predict_disease = disease_predict(prob_prompt_str)
            st.write(f"Your possible disease are: {predict_disease}")
            medical_record_info["predict_disease"] = predict_disease
            
            st.subheader("👩‍⚕️ Doctor Recommendation")
            # st.write(medical_record_info["symptom_list"])
            doctor = doctor_recommend(medical_record_info["symptom_list"])
            
            st.write(f"The best Recommend doctor is : {doctor[0]}")
            st.write(f"You may also choose to see : {doctor[1]}, {doctor[2]}")
            medical_record_info["doctor_recommendation"] = doctor
            
            st.subheader("📋 Disease Advice")
            advice = disease_advice(predict_disease)
            st.write(f"{advice}")
            medical_record_info["disease_advice"] = advice

            st.subheader("🤕 Disease Severity")
            severity = disease_severity(medical_record_info["symptom_list"])
            st.write(f"Medical severity: {severity}.")
            if severity > 5:
                st.write("Maybe you should See a doctor as soon as possible.")
            else :
                st.write("Pay attention to rest, take medicine under doctor's orders, and observe the situation.")

            medical_record_info["disease_severity"] = severity
            
            st.subheader("💉 Medicine Use")
            meidcine_col1, meidcine_col2 = st.columns([1, 1],gap="small")
            medicine_idx = medicine_use(predict_disease)
            match_medicine_name = []
            # st.write("Top 3 matching Medicine:")
            with meidcine_col1:
                for idx in medicine_idx:
                    best_match_medicine = medicine_df.iloc[idx]
                    st.write("-" * 50)
                    st.write(f"Name: {best_match_medicine['Medicine Name']}")
                    match_medicine_name.append(best_match_medicine['Medicine Name'])
                    st.write(f"Use: {best_match_medicine['Uses']}")
                    st.write(f"Composition: {best_match_medicine['Composition']}")
                    st.write(f"Side_effects: {best_match_medicine['Side_effects']}")
                    st.write("-" * 50)
                medical_record_info["medicine_name"] = match_medicine_name
                filename = save_to_json(medical_record_info)
                file_list.append(filename)

                st.success(f"Medical record formed and saved as {filename}!")    
            with meidcine_col2:  
                for name in medical_record_info["medicine_name"]:
                    img_path = medicine_picture_by_name(name)
                    st.image(img_path, caption=name, use_container_width=True)
        # st.divider()
    
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>➡️🧑‍⚕️ Doctor Diagnosis</h2>", unsafe_allow_html=True)    
    st.write("This button is for the doctor to check users' Medical record, and recheck the diagnosis.")
    st.button(label="Go to doctor diagnosis", on_click=goto_step, args=(4,))
    
    st.divider()
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>➡️💊 Medicine Search</h2>", unsafe_allow_html=True)  
    st.write("If you know your disease and forget which kind of medicine you need to take in:")
    st.button(label="Go to medicine search", on_click=goto_step, args=(3,))            

# Page 2 : Skin disease prediction by Image 
if st.session_state.step == 2:
    device = "cpu"
    num_classes = 22  
    model = ClassificationNetwork(num_classes=num_classes)
    model.load_state_dict(torch.load("best_resnet18_model.pt", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ImageFolder(r"Datasets\SkinDisease\train")  
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

    st.button(label="Go to main step", on_click=goto_step, args=(1,))

# Page 3 : Medicine Search by Text
if st.session_state.step == 3:
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
                opencilp_model, preprocess = load_openclip_model()
                text_embedding = opencilp_model.encode_text(text_tensor).squeeze().numpy()
            
            met_embeddings, id_list = load_embeddings()
            cosine_sim_array = get_all_cosine_similarities(met_embeddings, text_embedding)
            closest_match_id = get_id_for_most_similar_item(cosine_sim_array, id_list)

            image_path = f'Datasets/Medicine_Picture/{closest_match_id}.jpg'
            closest_match = Image.open(image_path).convert("RGB")
            st.write(f"closest match for text string: '{meidicine_description}' is the image:")
            st.image(image_path, caption=f"{closest_match_id}", use_container_width=True)
            st.button(label="Go to main page", on_click=goto_step, args=(1,))

# Page 4 : Doctor diagnosis
if st.session_state.step == 4:
    
    # For the st.cache will clean the medical record in page 1
    # So, I select the latest json saved in local 
    def get_latest_json(folder="Medical_record"):
        json_files = glob.glob(os.path.join(folder, "*.json"))  # Get all JSON file paths
        if not json_files:
            return None  # JSON file not found

        latest_file = max(json_files, key=os.path.getctime)  # Sort by creation time and get the latest files
        return latest_file

    # Calling the function
    latest_json = get_latest_json()
    # st.success(f"Medical record formed and saved as {filename}!"
    st.title("🩺 Doctor Diagnosis")
  
    
    
    # st.write(f"{file_list[0]}")
    with open(latest_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    st.json(data)
    file_list = []
    
    Medical_info_df = json_normalize(data)
    
    st.dataframe(Medical_info_df)
    doctor_diagnosis_info["name"] = data["name"]
    st.markdown("<h1 style='text-align: Left; font-size: 30px; margin-bottom: 5px'>Doctor Diagnosis</h1>", unsafe_allow_html=True)
    st.divider()
    with st.form(key="form2"):
        
        symptom_predict = st.radio("Do you think this is an accurate symptom list?", ("Yes", "No"))  
        if symptom_predict == "Yes":
            doctor_diagnosis_info["symptom_list"] = data["symptom_list"]
        elif symptom_predict == "No":
            doctor_diagnosis_info["symptom_list"] = st.text_input(label="please input the symptom:")
        
        disease_prediction = st.radio("Do you think this is an accurate disease prediction?", ("Yes", "No"))
        if disease_prediction == "Yes":
            doctor_diagnosis_info["predict_disease"]= data["predict_disease"]
        elif disease_prediction == "No":
            doctor_diagnosis_info["predict_disease"] = st.text_input(label="please input the disease:")

        doctor_recommendation = st.radio("Do you think this is an accurate doctor recommendation?", ("Yes", "No"))
        if doctor_recommendation == "Yes":
            doctor_diagnosis_info["doctor_recommendation"]= data["doctor_recommendation"]
        elif doctor_recommendation == "No":
            doctor_diagnosis_info["doctor_recommendation"] = st.text_input(label="please input the doctor:")

        disease_advices = st.radio("Do you think these are accurate disease advices?", ("Yes", "No"))
        if disease_advices == "Yes":
            doctor_diagnosis_info["disease_advice"]= data["disease_advice"]
        elif disease_advices == "No":
            doctor_diagnosis_info["disease_advice"] = st.text_input(label="please input the disease advices:")
        
        severity_predict = st.radio("Do you think this is an accurate severity prediction?", ("Yes", "No"))
        if severity_predict == "Yes":
            doctor_diagnosis_info["disease_severity"]= data["disease_severity"]
        elif severity_predict == "No":
            doctor_diagnosis_info["disease_severity"] = st.text_input(label="please input the severity:")        

        medicine_recommendation = st.radio("Do you think this is an medicine recommendation ?", ("Yes", "No"))
        if medicine_recommendation == "Yes":
            doctor_diagnosis_info["medicine_name"]= data["medicine_name"]
        elif medicine_recommendation == "No":
            doctor_diagnosis_info["medicine_name"] = st.text_input(label="please input the severity:")
        
        doctor_diagnosis_info["rating"] = st.slider(label="rating to the AI diagnosis", min_value=0, max_value=10)
        
        doctor_diagnosis_info["opinion"] = st.text_input(label="Please write your opinions:")

        # submit button
        form_submitted = st.form_submit_button(label="Submit!")

        def check_form_valid():
            vals = doctor_diagnosis_info.values()
            return all([True if val not in [None, ""] else False for val in vals ])
        
        # check format
        if form_submitted:
            if not check_form_valid():
                st.warning("Please fill in all values.")
            else:
                # st.balloons()
                st.write("form submitted!")
                filename = save_to_json(doctor_diagnosis_info)
                st.success(f"Doctor diagnosis formed and saved as {filename}!")

    st.button(label="Go to main page", on_click=goto_step, args=(1,))

