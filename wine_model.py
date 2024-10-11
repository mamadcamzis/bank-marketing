import joblib

from pathlib import Path
from loguru import logger
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

def load_data():
    logger.info("Load wine data ...")
    X, y = load_wine(return_X_y=True)
 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test


def train(dftrain, ytrain, model=RandomForestClassifier(n_estimators=10,
                                                         random_state=42)):
    logger.info("Train model ...")
    model.fit(dftrain, ytrain)
    return model

def evaluate(model, dftest, ytest):
    logger.info("Evaluate model ...")
    ypred = model.predict(dftest)
    #score = f1_score(ytest, ypred, average='micro')
    #logger.info(f"Model score: {score}")
    return classification_report(ytest, ypred, output_dict=True)

def save_model(model, path):
    logger.info("Save model ...")
    with open(path, 'wb') as file:
        joblib.dump(model, file)


def load_model(path):
    logger.info("Load model ...")
    with open(path, 'rb') as file:
        model = joblib.load(file)
    return model

def clf_pipeline(model_path):
    logger.info("Start pipeline ...")
    X_train, X_test, y_train, y_test = load_data()
    if Path(model_path).exists():
        model = load_model(model_path)
    else:
        
        model = train(X_train, y_train)
        save_model(model, model_path)
    score = evaluate(model, X_test, y_test)
    logger.info(f"Model score: {score}")
   


def main():
    logger.info("Start main ...")
    clf_pipeline("rfc_wine.joblib")

if __name__ == "__main__":
    main()