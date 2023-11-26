from datetime import datetime
from io import BytesIO
import os
import sys
import tempfile
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

class BinaryPromptMutator:
    def __init__(self, minio_client, prompt_type="positive", ranking_model="elm"):
        self.model = None
        self.minio_client= minio_client
        self.prompt_type= prompt_type
        self.ranking_model=ranking_model
        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path=self.get_model_path()
        self.accuracy=0

    def get_model_path(self):    
        local_path=f"output/binary_prompt_mutator.json"
        minio_path=f"environmental/models/prompt-generator/substitution/{self.prompt_type}_prompts_only/{self.date}_binary_{self.ranking_model}_model.json"

        return local_path, minio_path

    def train(self, input, output, 
              max_depth=7, 
              min_child_weight=1, 
              gamma=0.01, 
              subsample=1, 
              colsample_bytree=1, 
              eta=0.1,
              early_stopping=100):
        
        # Label encode the target variable
        label_encoder = LabelEncoder()
        output = label_encoder.fit_transform(output)

        X_train, X_val, y_train, y_val = train_test_split(input, output, test_size=0.2, shuffle=True)

        params = {
            'objective':'binary:logistic',
            "device": "cuda",
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'eta': eta,
            'eval_metric':'logloss',
            'n_estimators': 1000,
            'early_stopping_rounds':early_stopping
        }

        self.model=xgb.XGBClassifier(**params)

        # Train the XGBoost model
        evals_result={}
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)])
        # Extract the log loss values for each round
        evals_result = self.model.evals_result()

        # Extract log loss values for training and validation sets
        train_logloss = evals_result["validation_0"]["logloss"]
        val_logloss = evals_result["validation_1"]["logloss"]

        # Make predictions on the test set
        y_pred = self.model.predict(X_val)

        y_pred=label_encoder.inverse_transform(y_pred.astype(int))
        y_val=label_encoder.inverse_transform(y_val)

        # Calculate accuracy
        self.accuracy = sum(y_pred == y_val) / len(y_val)

        self.save_graph_report(train_logloss, val_logloss, y_val, y_pred, len(X_train), len(X_val))
    
    def save_graph_report(self, train_logloss_per_round, val_logloss_per_round,
                          y_true, y_pred,  
                          training_size, validation_size):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.03, 0.7, "Date = {}\n"
                            "Dataset = {}\n"
                            "Model = {}\n"
                            "Model type = {}\n"
                            "Input type = {}\n"
                            "Input shape = {}\n"
                            "Output type= {}\n\n"
                            ""
                            "Training size = {}\n"
                            "Validation size = {}\n"
                            "Accuracy ={:.4f}".format(self.date,
                                                            'environmental',
                                                            'Prompt Substitution',
                                                            'XGBoost',
                                                            f'{self.prompt_type}_clip_text_embedding',
                                                            '2306',
                                                            "binary",
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
 
        class_labels=['decrease', 'increase']
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        sb.heatmap(cm ,cbar=True, annot=True, cmap=custom_cmap, ax=axs[1], 
                   yticklabels=class_labels, xticklabels=class_labels, fmt='g')
        axs[1].set_title('Confusion Matrix')
        axs[1].set_xlabel('Predicted Labels')
        axs[1].set_ylabel('True Labels')
        axs[1].invert_yaxis()

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.7, wspace=0, left=0.4)

        plt.savefig(self.local_path.replace('.json', '.png'))

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.json', '.png'), buf)
        
    def predict_probs(self, X):
        dtest = xgb.DMatrix(X)
        class_labels=['decrease', 'increase']
        y_pred = self.model.predict_proba(dtest)
        # Create a list of dictionaries, where each dictionary represents the class probabilities for a single prediction
        predictions_with_probabilities = [
            {class_labels[i]: prob for i, prob in enumerate(row)} for row in y_pred
        ]

        return predictions_with_probabilities
        

    def load_model(self):
        minio_path=f"environmental/models/prompt-generator/substitution/{self.prompt_type}_prompts_only/"
        file_name=f"_binary_{self.ranking_model}_model.json"
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

        # Load the model from the temporary file into XGBClassifier
        self.model = xgb.XGBClassifier(device="cuda", tree_method='gpu_hist')
        self.model.load_model(temp_file.name)

        # Remove the temporary file
        os.remove(temp_file.name)

    def save_model(self):

        self.model.save_model(self.local_path)
        
        # Read the contents of the saved model file
        with open(self.local_path, "rb") as model_file:
            model_bytes = model_file.read()

        # Upload the model to MinIO
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path, BytesIO(model_bytes))
