import os
from datasets import load_dataset, Dataset
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer,SetFitHead
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import medical_assistant_package.config as cfg

# Load PubMedBERT from local
# The model from https://www.kaggle.com/datasets/jpmiller/layoutlm.

def load_and_prepare_data(train_path, test_path):
    # Load datasets
    dataset = load_dataset('csv', data_files={"train": train_path, "test": test_path})

    # Harmonized Listings
    dataset["train"] = dataset["train"].rename_column("Symptom", "text")
    dataset["test"] = dataset["test"].rename_column("Symptom", "text")

    # HLC
    le = LabelEncoder()
    all_labels = dataset["train"]['label'] + dataset["test"]['label']  
    le.fit(all_labels)

    dataset["train"] = dataset["train"].map(lambda x: {"label": le.transform([x["label"]])[0]})
    dataset["test"] = dataset["test"].map(lambda x: {"label": le.transform([x["label"]])[0]})

    return dataset, le

def load_models(local_model_path):

    # Loading the tokenizer and sentence encoding model
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    sentence_model = SentenceTransformer(local_model_path)
    return tokenizer, sentence_model

def create_setfit_model(sentence_model):
    
    # Load the base SetFit model
    setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Replace the coding model with PubMedBERT
    setfit_model.model_body = sentence_model
    return setfit_model

def train_and_evaluate(setfit_model, train_dataset, test_dataset, save_dir):
    trainer = SetFitTrainer(
        model=setfit_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=32,
        num_iterations=10,
        num_epochs=2,
        column_mapping={"text": "text", "label": "label"}
    )

    trainer.train()
    evaluation_results = trainer.evaluate()
    print("Evaluation Results:", evaluation_results)

    os.makedirs(save_dir, exist_ok=True)
    trainer.model._save_pretrained(save_directory=save_dir)

def main():
    local_model_path = "./pubmedbert_model"
    train_path = cfg.DISEASE_PREDICTION_TRAIN
    test_path = cfg.DISEASE_PREDICTION_TEST
    save_dir = "ckpt_12w/"

    # Loading and processing data
    dataset, label_encoder = load_and_prepare_data(train_path, test_path)

    # Loading Models
    tokenizer, sentence_model = load_models(local_model_path)

    # Build SetFit Model
    setfit_model = create_setfit_model(sentence_model)

    train_and_evaluate(setfit_model, dataset["train"], dataset["test"], save_dir)

if __name__ == "__main__":
    main()