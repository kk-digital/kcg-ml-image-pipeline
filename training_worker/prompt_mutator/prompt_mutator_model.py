from datetime import datetime
from io import BytesIO
import os
import sys
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd

class PromptMutator:
    def __init__(self, minio_client, model=None):
        self.model = model
        self.minio_client= minio_client

    def train(self, 
              X_train, 
              y_train, 
              max_depth=10, 
              min_child_weight=1, 
              gamma=0, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.1,
              early_stopping=100):
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
        }

        evals_result = {}
        self.model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, "validation")], 
                               early_stopping_rounds=early_stopping, evals_result=evals_result)

 
        # Extract RMSE values and residuals
        residuals=[]
        rmse = evals_result['validation']['rmse']
        evals = [(dval, "validation")]

        for i in range(len(evals)):
            preds = self.model.predict(evals[i][0])
            residuals.append(y_val - preds)
        
        self.save_graph_report(rmse, residuals, len(X_train), len(X_val))
        
    
    def save_graph_report(self, rmse_per_round, residuals, training_size, validation_size):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.1, 0.55, "Date = {}\n"
                            "Dataset = {}\n"
                            "Model type = {}\n"
                            "Input type = {}\n"
                            "Input shape = {}\n"
                            "Output type= {}\n\n"
                            ""
                            "Training size = {}\n"
                            "Validation size = {}\n".format(datetime.now().strftime("%Y-%m-%d"),
                                                            'environmental',
                                                            'XGBoost',
                                                            'clip_embedding',
                                                            '1536',
                                                            'delta_score',
                                                            training_size,
                                                            validation_size,
                                                            ))

        # Plot Validation Loss vs. Epochs
        axs[0].plot(range(1, len(rmse_per_round) + 1), rmse_per_round, label='RMSE')
        axs[0].set_title('RMSE per Round')
        axs[0].set_ylabel('Round')
        axs[0].set_xlabel('RMSE')
        axs[0].legend(['RMSE'])

        # plot histogram of residuals
        axs[1].hist(residuals, bins=30, color='blue', alpha=0.7)
        axs[1].set_xlabel('Residuals')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Validation Residual Histogram')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.7, left=0.5)

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        graph_output = os.path.join('environmental', "output/prompt_mutator/xgboost_model.png")
        cmd.upload_data(self.minio_client, 'datasets', graph_output, buf)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def save_model(self, model_path):
        self.model.save_model(model_path)
        model_bytes =self.model.save_raw()
        buffer = BytesIO(model_bytes)
        buffer.seek(0)
        # upload the model
        model_path = os.path.join('environmental', model_path)
        cmd.upload_data(self.minio_client, 'datasets', model_path, buffer)


