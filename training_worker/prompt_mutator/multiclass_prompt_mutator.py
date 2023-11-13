from datetime import datetime
from io import BytesIO
import os
import sys
import time
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sb

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd

class MulticlassPromptMutator:
    def __init__(self, minio_client, model=None, output_type="binary"):
        self.model = model
        self.minio_client= minio_client
        self.output_type=output_type
        self.accuracy=0

    def train(self, input, output, 
              max_depth=7, 
              min_child_weight=1, 
              gamma=0.01, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.05,
              early_stopping=50,
              num_class=2):
        
        # Label encode the target variable
        label_encoder = LabelEncoder()
        output = label_encoder.fit_transform(output)

        X_train, X_val, y_train, y_val = train_test_split(input, output, test_size=0.2, shuffle=True)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
        }

        if(num_class>2):
            params['objective']='multi:softmax'
            params['num_class']= num_class
        else:
            params['objective']='binary:logistic'
            params['eval_metric']='logloss'
        
        # Train the XGBoost model
        evals_result={}
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval,'eval'), (dtrain,'train')], 
                               early_stopping_rounds=early_stopping, evals_result=evals_result)
        
        #Extract logloss values
        val_logloss = evals_result['eval']['logloss']
        train_logloss = evals_result['train']['logloss']

        # Make predictions on the test set
        y_pred = model.predict(dval)

        y_pred=label_encoder.inverse_transform(y_pred.astype(int))
        y_val=label_encoder.inverse_transform(y_val)

        # Calculate accuracy
        self.accuracy = sum(y_pred == y_val) / len(y_val)
        print(f"accuracy:{self.accuracy}")

        self.model = model

        self.save_graph_report(train_logloss, val_logloss, y_val, y_pred, len(X_train), len(X_val))
    
    def save_graph_report(self, train_logloss_per_round, val_logloss_per_round,
                          y_true, y_pred,  
                          training_size, validation_size):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.03, 0.7, "Date = {}\n"
                            "Dataset = {}\n"
                            "Model type = {}\n"
                            "Input type = {}\n"
                            "Input shape = {}\n"
                            "Output type= {}\n\n"
                            ""
                            "Training size = {}\n"
                            "Validation size = {}\n"
                            "Accuracy ={}".format(datetime.now().strftime("%Y-%m-%d"),
                                                            'environmental',
                                                            'XGBoost',
                                                            'clip_text_embedding',
                                                            '1537',
                                                            self.output_type,
                                                            training_size,
                                                            validation_size,
                                                            self.accuracy
                                                            ))

        # Plot validation and training logloss vs. Rounds
        axs[0].plot(range(1, len(train_logloss_per_round) + 1), train_logloss_per_round,'b', label='Training loss')
        axs[0].plot(range(1, len(val_logloss_per_round) + 1), val_logloss_per_round,'r', label='Validation loss')
        axs[0].set_title('Loss per Round')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Rounds')
        axs[0].legend(['Training loss', 'Validation loss'])

        #confusion matrix
        # Generate a custom colormap representing brightness
        colors = [(1, 1, 1), (1, 0, 0)]  # White to Red
        custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

        cm = confusion_matrix(y_true, y_pred)
        sb.heatmap(cm ,cbar=True, annot=True, cmap=custom_cmap, ax=axs[1])
        axs[1].set_title('Confusion Matrix')
        axs[1].set_xlabel('Predicted Labels')
        axs[1].set_ylabel('True Labels')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.7, wspace=0, left=0.4)

        plt.savefig(f'output/{self.output_type}_model.png')

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        graph_output = os.path.join('environmental', f"output/prompt_mutator/{self.output_type}_model.png")
        cmd.upload_data(self.minio_client, 'datasets', graph_output, buf)
        
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
