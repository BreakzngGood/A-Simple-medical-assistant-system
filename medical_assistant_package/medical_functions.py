import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import util
import numpy as np
from model_train.doctor_recommendation_architecture import SpecialistNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import medical_assistant_package.load_models as lm
import medical_assistant_package.load_datasets as ld
from PIL import Image

data = ld.load_data()
feature_names = lm.load_symptom_feature()
label_encoder = lm.load_label_encoder()

# predict the disease by symptoms
def disease_predict(text,disease_name_df = data["disease_name_df"], disease_df = data["disease_df"]):
    setfit_model = lm.load_setfit_model()
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
def NLP_doctor_recommend(prompts):

    matched_symptoms = match_symptoms(prompts, feature_names)  # match userinput
    # st.write(f"{matched_symptoms}")
    # Create One-Hot encoded input data
    input_symptoms = pd.DataFrame(0, index=[0], columns=feature_names)
    input_symptoms.loc[0, matched_symptoms] = 1  # Only fill in the matched symptoms

    # predict
    Doctor_model = load_doctor_model()
    probabilities = Doctor_model.predict_proba(input_symptoms)[0]  
    top_3_indices = probabilities.argsort()[-3:][::-1]  
    top_3_doctors = Doctor_model.classes_[top_3_indices]  

    return top_3_doctors

# Return disease advice from the dataset, according to predict_disease
def disease_advice(predict_disease, medical_advice_df = data["medical_advice_df"] ):
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
def disease_severity(prompt_list, prompt_df = data["prompt_df"]):
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
def medicine_use(text,medicine_df = data["medicine_df"]):

    medicine_uses_texts = medicine_df["Uses"].fillna("")

    Sentence_model = lm.load_sentence_transformer()
    uses_embeddings = Sentence_model.encode(medicine_uses_texts.tolist(), convert_to_tensor=True)
    input_embedding = Sentence_model.encode(text, convert_to_tensor=True)

    # Calculate similarity
    cosine_sim = util.pytorch_cos_sim(input_embedding, uses_embeddings)

    top_matches_idx = cosine_sim.argsort(descending=True)[0][:3].tolist()

    return top_matches_idx

def medicine_picture_by_name(name):
    image_path = f'Datasets/Medicine_Picture/{name}.jpg'
    # print (image_path)
    return image_path

# search medicine by text 
def get_clip_embedding_from_PIL_image(image):
    opencilp_model, preprocess = lm.load_openclip_model()
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = opencilp_model.encode_image(image_tensor).squeeze().numpy()
    return embedding

def generate_embeddings(image_directory):
    embedding_list = []
    id_list = []

    for image_name in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_name)
        item_id = os.path.splitext(image_name)[0]
        id_list.append((item_id))
        try:
            image = Image.open(image_path).convert("RGB")
            embedding = get_clip_embedding_from_PIL_image(image)
            embedding_list.append(embedding)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
        
    return embedding_list, id_list

def get_all_cosine_similarities(embeddings_matrix, embedding_vector):
        dot_product = embeddings_matrix @ embedding_vector
        product_of_magnitudes = np.linalg.norm(embeddings_matrix, axis = 1) * np.linalg.norm(embedding_vector)
        return dot_product / product_of_magnitudes

def get_id_for_most_similar_item(similarity_array, id_list):
    highest_score_index = np.argmax(similarity_array)
    item_id = id_list[highest_score_index]
    return item_id

