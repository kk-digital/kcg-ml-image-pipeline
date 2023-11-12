from datetime import datetime
from io import BytesIO
import os
import sys
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd

class MulticlassPromptMutator:
    def __init__(self, minio_client, model=None):
        self.model = model
        self.minio_client= minio_client

    def train(self, input, output, 
              max_depth=7, 
              min_child_weight=1, 
              gamma=0.0, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.05):
        
        # Label encode the target variable
        label_encoder = LabelEncoder()
        output = label_encoder.fit_transform(output)

        X_train, X_val, y_train, y_val = train_test_split(input, output, test_size=0.2, shuffle=True)

        params = {
            'objective': 'multi:softmax',  # Use softmax for multi-class classification
            'num_class': 5,  # Number of classes
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
            'eval_metric': 'mlogloss'  # Use multi-logloss as the evaluation metric
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train the XGBoost model
        model = xgb.train(params, dtrain, num_boost_round=2000)

        # Assuming you have a separate test set X_test, y_test
        dval = xgb.DMatrix(X_val)

        # Make predictions on the test set
        y_pred = model.predict(dval)

        # Convert the predicted labels back to original class labels
        y_pred_original = LabelEncoder.inverse_transform(y_pred.astype(int))

        # Calculate accuracy
        accuracy = sum(y_pred_original == y_val) / len(y_val)
        print(accuracy)

        self.model = model
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def save_model(self, local_path ,minio_path):

        self.model.save_model(local_path)
        
        # Read the contents of the saved model file
        with open(local_path, "rb") as model_file:
            model_bytes = model_file.read()

        # Upload the model to MinIO
        cmd.upload_data(self.minio_client, 'datasets', minio_path, BytesIO(model_bytes))
