> Notice: This medical system is in a very early stage of development and poses significant risks if used in real-world settings. Please do not rely on or reference it for medical advice or diagnosis.

# 🧬 Siro
Siro is a simple medical assistant system deployed on Streamlit. It consists of five main functional modules:

•	Medical record collection 
•	Skin disease diagnosis based on picture prediction
•	Symptom diagnosis based on text input (Also includes disease prediction, severity Assessment, Medical Advice, and doctor and medicine recommendation.)
•	Medicine Search by text
•	Doctor Diagnosis based on health records and system's predicted results.

> 🔗[Examples]

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
Since the project includes some large datasets and models, you can click [here]( https://artslondon-my.sharepoint.com/:u:/g/personal/h_shi1220231_arts_ac_uk/EbucxCL38lZAv89_uEF4nqMBEgVGQAdXjatTz0fevaz3Sw?e=yKOfkh) to download them if needed and experience the full functionality.


## Link to project video recording: https://artslondon-my.sharepoint.com/:v:/g/personal/h_shi1220231_arts_ac_uk/EZRiLR7vTodHjLo_POplW84BcP0sMflfkgqoXM__XElizw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=FAD2Q7

# Setup instructions:

1. You need to install another library called `nltk` to run this project. Maybe the `sentence_transformers` and the `setfit` are also needed to install.

```
pip install nltk
pip install sentence_transformers
pip install setfit

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

```

2. To Run this project, you can choose the 'Medical_System_streamlit.py', and input the 'streamlit run Medical_System_streamlit.py` in the Terminal to run.

3. For that, there are many big data in this project, you can download the complete project through this link: https://artslondon-my.sharepoint.com/:u:/g/personal/h_shi1220231_arts_ac_uk/EbucxCL38lZAv89_uEF4nqMBEgVGQAdXjatTz0fevaz3Sw?e=yKOfkh
