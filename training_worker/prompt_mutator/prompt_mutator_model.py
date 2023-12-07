from datetime import datetime
from io import BytesIO
import os
import sys
import tempfile
import time
from matplotlib import pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd

class PromptMutator:
    def __init__(self, minio_client, model=None, output_type="sigma_score", prompt_type="positive",
                 ranking_model="elm", operation="substitution"):
        self.model = model
        self.minio_client= minio_client
        self.output_type= output_type
        self.prompt_type= prompt_type
        self.ranking_model=ranking_model
        self.operation=operation
        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path=self.get_model_path()

    def get_model_path(self):
        
        local_path=f"output/{self.output_type}_prompt_mutator.json"
        minio_path=f"environmental/models/prompt-generator/{self.operation}/{self.prompt_type}_prompts_only/{self.date}_{self.output_type}_{self.ranking_model}_model.json"

        return local_path, minio_path

    def train(self, 
              X_train,
              y_train, 
              max_depth=7, 
              min_child_weight=1,
              gamma=0.01, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.1,
              early_stopping=50):
        
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        
        params = {
            'objective': 'reg:absoluteerror',
            "device": "cuda",
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
            'eval_metric': 'mae'
        }

        evals_result = {}
        
        start = time.time()
        self.model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval,'eval'), (dtrain,'train')], 
                                early_stopping_rounds=early_stopping, evals_result=evals_result)

        end = time.time()

        training_time= end - start
 
        #Extract MAE values and residuals
        val_mae = evals_result['eval']['mae']
        train_mae = evals_result['train']['mae']


        start = time.time()
        # Get predictions for validation set
        val_preds = self.model.predict(dval)

        # Get predictions for training set
        train_preds = self.model.predict(dtrain)
        end = time.time()
        inference_speed=(len(X_train) + len(X_val))/(end - start)
        print(f'Time taken for inference of {len(X_train) + len(X_val)} data points is: {end - start:.2f} seconds')

        # Now you can calculate residuals using the predicted values
        val_residuals = y_val - val_preds
        train_residuals = y_train - train_preds
        
        self.save_graph_report(train_mae, val_mae, 
                               val_residuals, train_residuals, 
                               val_preds, y_val,
                               len(X_train), len(X_val))
        
        self.save_model_report(num_training=len(X_train),
                              num_validation=len(X_val),
                              training_time=training_time, 
                              train_loss=train_mae, 
                              val_loss=val_mae, 
                              inference_speed= inference_speed,
                              model_params=params)
        
        return val_mae[-1]
    
    def save_model_report(self,num_training,
                              num_validation,
                              training_time, 
                              train_loss, 
                              val_loss, 
                              inference_speed,
                              model_params):
        report_text = (
            "================ Model Report ==================\n"
            f"Number of training datapoints: {num_training} \n"
            f"Number of validation datapoints: {num_validation} \n"
            f"Total training Time: {training_time:.2f} seconds\n"
            "Loss Function: L1 \n"
            f"Training Loss: {train_loss[-1]} \n"
            f"Validation Loss: {val_loss[-1]} \n"
            f"Inference Speed: {inference_speed:.2f} predictions per second\n\n"
            "================ Parameters ==================\n"
        )

        # Add model parameters to the report
        for param, value in model_params.items():
            report_text += f"{param}: {value}\n"

        # Define the local file path for the report
        local_report_path = 'output/model_report.txt'

        # Save the report to a local file
        with open(local_report_path, 'w') as report_file:
            report_file.write(report_text)

        # Read the contents of the local file
        with open(local_report_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = BytesIO(content)
        buffer.seek(0)

        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.json', '.txt'), buffer)

        # Remove the temporary file
        os.remove(local_report_path)

    def save_graph_report(self, train_mae_per_round, val_mae_per_round, 
                          val_residuals, train_residuals, 
                          predicted_values, actual_values,
                          training_size, validation_size):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.02, 0.7, "Date = {}\n"
                            "Dataset = {}\n"
                            "Model = {}\n"
                            "Model type = {}\n"
                            "Input type = {}\n"
                            "Input shape = {}\n"
                            "Output type= {}\n\n"
                            ""
                            "Training size = {}\n"
                            "Validation size = {}\n"
                            "Training loss = {:.4f}\n"
                            "Validation loss = {:.4f}\n".format(self.date,
                                                            'environmental',
                                                            f'Prompt {self.operation}',
                                                            'XGBoost',
                                                            f'{self.prompt_type}_clip_text_embedding',
                                                            '2306',
                                                            self.output_type,
                                                            training_size,
                                                            validation_size,
                                                            train_mae_per_round[-1],
                                                            val_mae_per_round[-1],
                                                            ))

        # Plot validation and training Rmse vs. Rounds
        axs[0][0].plot(range(1, len(train_mae_per_round) + 1), train_mae_per_round,'b', label='Training mae')
        axs[0][0].plot(range(1, len(val_mae_per_round) + 1), val_mae_per_round,'r', label='Validation mae')
        axs[0][0].set_title('MAE per Round')
        axs[0][0].set_ylabel('MAE')
        axs[0][0].set_xlabel('Rounds')
        axs[0][0].legend(['Training mae', 'Validation mae'])
        
        # Scatter Plot of actual values vs predicted values
        axs[0][1].scatter(predicted_values, actual_values, color='green', alpha=0.5)
        axs[0][1].set_title('Predicted values vs actual values')
        axs[0][1].set_ylabel('True')
        axs[0][1].set_xlabel('Predicted')

        # plot histogram of training residuals
        axs[1][0].hist(train_residuals, bins=30, color='blue', alpha=0.7)
        axs[1][0].set_xlabel('Residuals')
        axs[1][0].set_ylabel('Frequency')
        axs[1][0].set_title('Training Residual Histogram')

        # plot histogram of validation residuals
        axs[1][1].hist(val_residuals, bins=30, color='blue', alpha=0.7)
        axs[1][1].set_xlabel('Residuals')
        axs[1][1].set_ylabel('Frequency')
        axs[1][1].set_title('Validation Residual Histogram')
        
        # plot histogram of predicted values
        axs[2][0].hist(predicted_values, bins=30, color='blue', alpha=0.7)
        axs[2][0].set_xlabel('Predicted Values')
        axs[2][0].set_ylabel('Frequency')
        axs[2][0].set_title('Validation Predicted Values Histogram')
        
        # plot histogram of true values
        axs[2][1].hist(actual_values, bins=30, color='blue', alpha=0.7)
        axs[2][1].set_xlabel('Actual values')
        axs[2][1].set_ylabel('Frequency')
        axs[2][1].set_title('Validation True Values Histogram')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.7, wspace=0.3, left=0.3)

        plt.savefig(self.local_path.replace('.json', '.png'))

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.json', '.png'), buf)  

        # Clear the current figure
        plt.clf()           
    
    def grid_search(self, X_train, y_train, param_grid, cv=5, scoring='neg_mean_squared_error'):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        xgb_model = xgb.XGBRegressor()

        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            scoring=scoring,
            cv=kfold,
            n_jobs=-1,
            verbose=1,
        )

        grid_result = grid_search.fit(X_train, y_train)

        best_params = grid_result.best_params_
        best_score = grid_result.best_score_

        return best_params, best_score

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self):
        minio_path=f"environmental/models/prompt-generator/{self.operation}/{self.prompt_type}_prompts_only/"
        file_name=f"_{self.output_type}_{self.ranking_model}_model.json"
        # get model file data from MinIO
        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', minio_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .pth files found in the list.")
            return
        
        print(most_recent_model)

        # Create a temporary file and write the downloaded content into it
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for data in model_file_data.stream(amt=8192):
                temp_file.write(data)

        # Load the model from the downloaded bytes
        self.model = xgb.Booster(model_file=temp_file.name)
        self.model.set_param({"device": "cuda"})
        
        # Remove the temporary file
        os.remove(temp_file.name)

    def save_model(self):
        self.model.save_model(self.local_path)
        
        #Read the contents of the saved model file
        with open(self.local_path, "rb") as model_file:
            model_bytes = model_file.read()

        # Upload the model to MinIO
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path, BytesIO(model_bytes))


