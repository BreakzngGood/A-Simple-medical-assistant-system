import os
import pandas as pd
from datasets import load_dataset, Dataset
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer,SetFitHead
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Clean and preprocess the original dataset
# -------------------------------------------------

# df = pd.read_csv("Symptom\dataset.csv")

# # Merge all columns except the first one and remove null values
# df["Symptom"] = df.iloc[:, 1:].astype(str).apply(lambda x: " ".join(x.dropna().replace("", None)), axis=1)
# df["Symptom"] = df["Symptom"].str.replace(r"\bnan\b", "", regex=True)  # Remove 'NaN'
# df["Symptom"] = df["Symptom"].str.replace(r"\s+", " ", regex=True).str.strip()  # Remove extra spaces
# df["Symptom"] = df["Symptom"].str.replace(" ", ",")  # Replace spaces with commas
# df["Symptom"] = df["Symptom"].str.replace("_", " ")  # Replace spaces with commas

# df = df[["Disease", "Symptom"]]
# df.to_csv(r"New_Disease_with_symptom.csv")

# df = pd.read_csv(r"New_Disease_with_symptom.csv")
# df["label"] = pd.factorize(df["Disease"])[0]
# df.to_csv("New_Disease_with_label.csv", index=False)

# df = pd.read_csv("New_Disease_with_label.csv")
# df = df.drop(df.columns[0], axis=1)
# df.to_csv("Cleaned_Disease_with_label.csv", index=False)

# -------------------------------------------------
# Split the dataset into train and test
# -------------------------------------------------

# df = pd.read_csv("Cleaned_Disease_with_label.csv")
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# train_df.to_csv("train.csv", index=False)
# test_df.to_csv("test.csv", index=False)

# -------------------------------------------------
# Download the BiomedNLP-PubMedBERT model from online and save to local
# -------------------------------------------------

# model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# # Download for the first time
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Save the model to Local
# tokenizer.save_pretrained("./pubmedbert_model")
# model.save_pretrained("./pubmedbert_model")

# print("PubMedBERT Saved successfully")


# Load dataset
# dataset = load_dataset('csv', data_files={
#     "train": r'train.csv',
#     "test": r'test.csv'
# })

# dataset["train"] = dataset["train"].rename_column("Symptom", "text")
# dataset["test"] = dataset["test"].rename_column("Symptom", "text")

# # Encode labels
# le = LabelEncoder()
# intent_dataset_train = le.fit_transform(dataset["train"]['label'])
# dataset["train"] = dataset["train"].remove_columns("label").add_column("label", intent_dataset_train).cast(dataset["train"].features)

# intent_dataset_test = le.fit_transform(dataset["test"]['label'])
# dataset["test"] = dataset["test"].remove_columns("label").add_column("label", intent_dataset_test).cast(dataset["test"].features)

# # Initialize model and trainer
# # Switch the model from 
# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# model = SetFitModel.from_pretrained(model_id)

# trainer = SetFitTrainer(
#     model=model,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     loss_class=CosineSimilarityLoss,
#     metric="accuracy",
#     batch_size=64,
#     num_iterations=20,
#     num_epochs=2,
#     column_mapping={"text": "text", "label": "label"}
# )

# # Train the model
# trainer.train()

# # Evaluate the model
# evaluation_results = trainer.evaluate()
# print("Evaluation Results:", evaluation_results)

# os.makedirs('ckpt/', exist_ok=True)

# trainer.model._save_pretrained(save_directory="ckpt/")

# Load PubMedBERT from local
# The model from https://www.kaggle.com/datasets/jpmiller/layoutlm.

local_model_path = "./pubmedbert_model"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
sentence_model = SentenceTransformer(local_model_path)

# Load the dataset
dataset = load_dataset('csv', data_files={"train": r"Datasets\train.csv", "test": r"Datasets\test.csv"})

# Unified data format
dataset["train"] = dataset["train"].rename_column("Symptom", "text")
dataset["test"] = dataset["test"].rename_column("Symptom", "text")

# Unified label encoding
le = LabelEncoder()
all_labels = dataset["train"]['label'] + dataset["test"]['label']  
le.fit(all_labels)

dataset["train"] = dataset["train"].map(lambda x: {"label": le.transform([x["label"]])[0]})
dataset["test"] = dataset["test"].map(lambda x: {"label": le.transform([x["label"]])[0]})

# Correctly load SetFitModel via `from_pretrained()`
setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Replace SetFit's SentenceTransformer encoder
setfit_model.model_body = sentence_model  # **Manual replacement with PubMedBERT**

trainer = SetFitTrainer(
    model=setfit_model,  
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size = 32,
    num_iterations= 10,
    num_epochs = 2,
    column_mapping={"text": "text", "label": "label"}
)

trainer.train()

evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)

os.makedirs('ckpt_new/', exist_ok=True)
trainer.model._save_pretrained(save_directory="ckpt_new/")