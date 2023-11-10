from datetime import datetime
from io import BytesIO
import os
import sys
import time
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
              alpha=0.8, 
              gamma=0.0, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.1,
              early_stopping=100):
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:linear',
            'alpha':alpha,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
        }

        evals_result = {}
        self.model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval,'eval'), (dtrain,'train')], 
                               early_stopping_rounds=early_stopping, evals_result=evals_result, feval='rmse')

 
        #Extract RMSE values and residuals
        val_rmse = evals_result['eval']['rmse']
        train_rmse = evals_result['train']['rmse']


        start = time.time()
        # Get predictions for validation set
        val_preds = self.model.predict(dval)

        # Get predictions for training set
        train_preds = self.model.predict(dtrain)
        end = time.time()
        print(f'Time taken for inference of {len(X_train) + len(X_val)} data points is: {end - start:.2f} seconds')

        # Now you can calculate residuals using the predicted values
        val_residuals = y_val - val_preds
        train_residuals = y_train - train_preds

        accuracy=0
        for val, pred_val in zip(y_val,val_preds):
            if((val<0 and pred_val<0) or (val>0 and pred_val>0) or (val==0 and pred_val==0)):
                accuracy+=1
        
        print(f'accuracy:{accuracy/len(y_val)}')
        
        self.save_graph_report(train_rmse, val_rmse, val_residuals, train_residuals, val_preds, y_val, len(X_train), len(X_val))
    
    def save_graph_report(self, train_rmse_per_round, val_rmse_per_round, 
                          val_residuals, train_residuals, 
                          predicted_values, actual_values,
                          training_size, validation_size):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.02, 0.7, "Date = {}\n"
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
                                                            'clip_text_embedding',
                                                            '1537',
                                                            'delta_score',
                                                            training_size,
                                                            validation_size,
                                                            ))

        # Plot Training Rmse vs. Rounds
        axs[0][0].plot(range(1, len(train_rmse_per_round) + 1), train_rmse_per_round, label='RMSE')
        axs[0][0].set_title('RMSE per Round for training')
        axs[0][0].set_ylabel('RMSE')
        axs[0][0].set_xlabel('Rounds')
        axs[0][0].legend(['RMSE'])
        
        # Plot Validation Rmse vs. Rounds
        axs[0][1].plot(range(1, len(val_rmse_per_round) + 1), val_rmse_per_round, label='RMSE')
        axs[0][1].set_title('RMSE per Round for validation')
        axs[0][1].set_ylabel('RMSE')
        axs[0][1].set_xlabel('Rounds')
        axs[0][1].legend(['RMSE'])

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

        plt.savefig('output/xgboost_model.png')

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


