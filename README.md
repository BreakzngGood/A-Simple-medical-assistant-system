> Notice: This medical system is in a very early stage of development and poses significant risks if used in real-world settings. Please do not rely on or reference it for medical advice or diagnosis.

# 🧬 Siro
Siro is a simple medical assistant system deployed on Streamlit. It consists of five main functional modules:

•	📋 Medical record collection 

•	🧴 Skin disease diagnosis based on picture prediction

•	🤒 Symptom diagnosis based on text input (Also includes disease prediction, severity Assessment, Medical Advice, and doctor and medicine recommendation.)

•	💊 Medicine Search by text

•	🧑‍⚕️ Doctor Diagnosis based on health records and system's predicted results.

Siro is a developer’s attempt to apply AI in the field of medical consultation. The developer aims to build an AI-supported medical consultation system to save medical resources and improve the efficiency of the healthcare process.

> 🔗[💻 Setup instructions](#-Setup-instructions)[🐍 How to use](#-How-to-use)[📌 Example](#-Example)
[🚀 Updates](#-Updates)[💾 Directory Structure](#-Directory-Structure)

# 💻 Setup instructions
```bash
pip install -r requirements.txt 
```
If you encounter any nltk-related errors during runtime, please download the required resources manually.
```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
```
### 🌎 Project download
Since the project includes some large datasets and models, you can click [here]( https://artslondon-my.sharepoint.com/:u:/g/personal/h_shi1220231_arts_ac_uk/EbucxCL38lZAv89_uEF4nqMBEgVGQAdXjatTz0fevaz3Sw?e=yKOfkh) to download them if needed and experience the full functionality.

# 🐍 How to use
### ▶️ Guided Video
You can learn how to use Siro through a [guided video]( https://artslondon-my.sharepoint.com/:v:/g/personal/h_shi1220231_arts_ac_uk/EZRiLR7vTodHjLo_POplW84BcP0sMflfkgqoXM__XElizw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=FAD2Q7), but please note that the video contains some incorrect statements regarding the model’s accuracy and loss—please disregard those parts.

### 🧭 Usage Process 
First, please enter the following command in the terminal:
```bash
streamlit run medical_app.py
```
- It will open the Streamlit webpage in your default browser.

- You can interact with various features on the webpage and switch between pages using the switch page button.

- Suggested usage flow: please try the medical record feature first, and finish by trying the doctor diagnosis feature.

# 📌 Example

### 📋 Medical record collection 
Record the basic information of the patient, including biological sex, age, medical history, and so on.

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/244500dda35806000e10f066bbf1fdcb793df00e/project_image/Medical%20record%20collection.png)

### 🧴 Skin disease diagnosis
Perform diagnosis of skin diseases based on images uploaded by the user or captured via the camera.

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Skin%20disease%20diagnosis%201.png)

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Skin%20disease%20diagnosis%202.png)

### 🤒 Symptom diagnosis

Users can input their symptoms via voice or text. The system will provide a series of medical diagnoses and recommendations, including Symptom Prediction, Disease Prediction, Doctor Recommendation, Disease Advice, Disease Severity assessment, and Medicine Use guidance.

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Symptom%20Diagnosis.png)

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Symptom%20Diagnosis%202.png)

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Symptom%20Diagnosis%203.png)


### 💊 Medicine Search by text

Users can find corresponding medicines by describing their symptoms or needs in text form.

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Medicine%20search.png)

### 🧑‍⚕️ Doctor Diagnosis

All user inputs and the information obtained will be organized and provided to doctors for secondary diagnosis and confirmation.

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/project_image/Doctor%20diagnosis.png)

# 🚀 Updates
**2025.05**
The project underwent considerable adjustments in both code implementation and structure in order to better align with industry standards.

- ©️ “clean code”: The file and variable names have been standardized and modified to avoid irregular naming conventions and inconsistent use of letter cases.

- ©️ [Functional modularization](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/tree/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/medical_assistant_package): Repetitive code, especially in Streamlit form construction, was refactored into modular functions to enhance clarity and reusability.At the same time, the developer categorized the numerous functions in the code and encapsulated them within a single package.

- ©️ Code structure iteration: Mainly applied adjustments to the program’s main entry control structure to facilitate subsequent development and maintenance.

- ©️ Exception handling: The code includes exception capturing and appropriate error messages across multiple functions, especially in importing and saving JSON files, to prevent program crashes.

- ©️ [Unit test](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/tree/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/tests)：The main medical functionalities were tested using pytest/unittest.

- ©️ [Configuration File Setup](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/2bbbeb6af9fbbb9ce71a26c1be9123460728199e/medical_assistant_package/config.py): A configuration file is used to centrally manage project-level variables such as datasets and models.

# 💾 Directory Structure
```
|- medical_assistant_package/       (The package includes configuration files and functions)
    |- audio_utils.py             (Audio processing functionality)
    |- config.py                (Configuration management)
    |- file_save.py             (File saving functionality)
    |- load_datasets.py             (Optimized loading using @st.cache_data)
    |- load_models.py               (Optimized loading using @st.cache_data)           
    |- medical_functions.py             (Core Medical Functionalities)
    |- streamlit_navigation.py              (Streamlit Feature Design)
    |- text_utils.py                (Text Processing Functionality)      
|- model_train/           (models trained for the project)
    |- disease_prediciton_model.py             (Setfit model)
    |- doctor_recommendation_architecture.py            (model architecture)
    |- doctor_recommendation_model_classical_ML.py (Classical ML model)
    |- doctor_recommendation_model.py        (MLP model)
    |- image_classification_model_pretrained.py
    |- image_classification_model.py
    |- image_classifier_from_pretrained.py
    |- image_classifier.py
    |- medicine_image_dowload.py                (Download images from the links from the dataset)
```
