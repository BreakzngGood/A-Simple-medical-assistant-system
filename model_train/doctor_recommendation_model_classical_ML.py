import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import medical_assistant_package.config as cfg
import pickle

# The features in the original dataset have already been encoded (One-Hot Encoding), so no further feature engineering is needed.
# Based on these features, I trained and compared two models: Random Forest and Logistic Regression.
# The final result shows that the model can predict the most suitable doctor based on disease symptoms.

def load_data(path):
    df = pd.read_excel(path)
    X = df.drop(['Disease', 'Unnamed: 0'], axis=1)
    y = df['Disease']
    return X, y

def train_models(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=200, random_state=50)
    rf_model.fit(X_train, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=50)
    lr_model.fit(X_train, y_train)

    return rf_model, lr_model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_symptom_feature(path):
    df = pd.read_excel(path)
    X = df.drop(['Disease', 'Unnamed: 0'], axis=1)
    return X.columns.tolist()

if __name__ == "__main__":
    X, y = load_data(cfg.SPECIALIST_EXCEL_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

    rf_model, lr_model = train_models(X_train, y_train)

    rf_acc = evaluate_model(rf_model, X_test, y_test)
    lr_acc = evaluate_model(lr_model, X_test, y_test)
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print(f"Logistic Regression accuracy: {lr_acc:.4f}")

    save_model(lr_model, 'Specialist.pkl')

    feature_names = load_symptom_feature(cfg.SPECIALIST_EXCEL_PATH)
    print("Feature names:", feature_names)