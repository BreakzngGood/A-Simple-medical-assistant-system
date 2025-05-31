> Notice: This medical system is in a very early stage of development and poses significant risks if used in real-world settings. Please do not rely on or reference it for medical advice or diagnosis.

## 🧬 Siro
Siro is a simple medical assistant system deployed on Streamlit. It consists of five main functional modules:

•	📋 Medical record collection 

•	🧴 Skin disease diagnosis based on picture prediction

•	🤒 Symptom diagnosis based on text input (Also includes disease prediction, severity Assessment, Medical Advice, and doctor and medicine recommendation.)

•	💊 Medicine Search by text

•	🧑‍⚕️ Doctor Diagnosis based on health records and system's predicted results.
Siro 是开发者将AI应用到医疗问诊领域的一次尝试，开发者希望能够构建一个ai支持的医疗问诊系统，以达到节约医疗资源，提升看病流程的效率。

> 🔗[Examples]

## 💻 Setup instructions
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

## 🐍 How to use
### ▶️ Guided Video
你可以通过一个引导视频来了解Siro的使用方法，但请注意，视频中存在一些关于模型准确率，损失方面的描述错误，请忽略这一部分的表述。

### 🧭 Usage Process 
首先请在终端输入以下命令
```bash
streamlit run medical_app.py
```
它会在你默认的浏览器上打开streamlit网页
您可以与网页上的各种功能进行交互，并通过switch page button 切换页面。
流程体验的建议：请先体验medical record功能，并最后体验doctor diagnosis

## 📌 Example
### 📋 Medical record collection 
记录问询者的基本情况，包括生理性别，年龄，病史等

![generation_img](https://github.com/BreakzngGood/A-Simple-medical-assistant-system/blob/244500dda35806000e10f066bbf1fdcb793df00e/project_image/Medical%20record%20collection.png)

### 🧴 Skin disease diagnosis
根据用户上传或者摄像头捕获的图片，进行皮肤疾病上的诊断

### 🤒 Symptom diagnosis
用户可以通过语音或者文字输入症状，系统会给出一系列的医学诊断和建议，包括Symptom prediction， Disease Prediction,Doctor Recommendation， Disease Advice，Disease Severity，Medicine Use

### 💊 Medicine Search by text
用户可以通过文字描述来找到对应的药物

### 🧑‍⚕️ Doctor Diagnosis
所有用户输入和得到的信息会被整理，交与医生进行二次的诊断和确认

## 🚀 Updates

**2025.05**
The project underwent considerable adjustments in both code implementation and structure in order to better align with industry standards.

- ©️ “clean code”: 对于文件，变量的命名进行统一和修改，规避了一些不规范的命名和大小写运用。
- ©️ Functional modularization: Repetitive code, especially in Streamlit form construction, was refactored into modular functions to enhance clarity and reusability.同时，对于代码中的众多功能，开发者进行了不同的分类，并将它们用封装在一个package中。
- ©️ 代码结构迭代：主要应用了程序主入口控制结构调整代码，方便后续的开发和维护
- ©️ 异常处理：代码在多处功能间设置了异常捕获以及合理的错误提示，特别是在json文件的导入和存储方面，避免程序崩溃
- ©️ Unit test：借助pytest/unittest对主要的医疗功能进行了测试
- ©️ 设置配置文件：配置文件统一管理项目中使用的数据集，模型等项目级别的变量

## Directory Structure
```
|- medical_assistant_package/       (package include 配置文件，streamlit 功能，医疗功能等)
    |- audio_utils.py             (音频处理功能)
    |- config.py                (配置管理)
    |- file_save.py             (文件保存功能)
    |- load_datasets.py             (结合@st.cache_data优化的加载功能)
    |- load_models.py               (结合@st.cache_data优化的加载功能)           
    |- medical_functions.py             (核心医疗功能)
    |- streamlit_navigation.py              (streamlit功能设计)
    |- text_utils.py                (文字处理功能)      
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


## Link to project video recording: https://artslondon-my.sharepoint.com/:v:/g/personal/h_shi1220231_arts_ac_uk/EZRiLR7vTodHjLo_POplW84BcP0sMflfkgqoXM__XElizw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=FAD2Q7


3. For that, there are many big data in this project, you can download the complete project through this link: https://artslondon-my.sharepoint.com/:u:/g/personal/h_shi1220231_arts_ac_uk/EbucxCL38lZAv89_uEF4nqMBEgVGQAdXjatTz0fevaz3Sw?e=yKOfkh
