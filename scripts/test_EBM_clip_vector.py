## Standard libraries
import os
import json
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from torch.utils.data import random_split, DataLoader

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    #!pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



# ADDED BY ME
from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from io import BytesIO
import io
import os
import sys
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from utility.minio import cmd
from utility.clip.clip_text_embedder import tensor_attention_pooling

import urllib.request
from urllib.error import HTTPError
from torchvision.datasets import SVHN
from torch.utils.data import random_split, DataLoader
from PIL import Image
import requests
import msgpack 

matplotlib.rcParams['lines.linewidth'] = 2.0

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../savedmodels"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False







########################################### Initialize the cuda device 
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


########################################### Initialize minio 

date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
print(date_now)


minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
minio_path="environmental/output/my_test"




###########################################

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object

API_URL = "http://192.168.3.1:8111"

batchsize_x = 16
########################################### Get tags

# def get_tag_jobs(tag_id):

#         response = requests.get(f'{API_URL}/tags/get-images-by-tag-id/?tag_id={tag_id}')
        
#         jobs = json.loads(response.content)

#         file_paths=[job['file_path'] for job in jobs]

#         return file_paths



########### Get embedings

# from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder

# image_embedder= KandinskyCLIPImageEncoder(device="cuda")
#     image_embedder.load_submodels()

#     image_paths= get_image_paths("input/characters")
#     images=[]

#     for index, path in enumerate(image_paths):
#         image= Image.open(path).convert("RGB")
#         image_embedding=image_embedder.get_image_features(image)





def get_tag_jobs(tag_id):
    response = requests.get(f'{API_URL}/tags/get-images-by-tag-id/?tag_id={tag_id}')
    
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        try:
            # Parse the JSON response
            response_data = json.loads(response.content)

            # Check if 'images' key is present in the JSON response
            if 'images' in response_data.get('response', {}):
                # Extract file paths from the 'images' key
                file_paths = [job['file_path'] for job in response_data['response']['images']]
                return file_paths
            else:
                print("Error: 'images' key not found in the JSON response.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Error: HTTP request failed with status code {response.status_code}")

    # Return an empty list or appropriate value to indicate an error
    return []

########################################### Get clip vectors

def get_clip_vectors(file_paths):
        clip_vectors=[]
        for path in file_paths:
            clip_path= path.replace(".jpg","_clip_kandinsky.msgpack")
            bucket, features_vector_path= separate_bucket_and_file_path(clip_path) 
            features_data = get_object(minio_client, features_vector_path)
            features = msgpack.unpackb(features_data)["clip-feature-vector"]
            
            features = torch.tensor(features)
            
            clip_vectors.append(features)
            
        return clip_vectors    



########################################### get images

def get_image(file_path: str):
    # get image from minio server
    bucket_name, file_path = separate_bucket_and_file_path(file_path)
    try:
        response = minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.data)
        img = Image.open(image_data)
        img = img.convert("RGB")
    except Exception as e:
        raise e
    finally:
        response.close()
        response.release_conn()

    return img


########################################### DATA Augmentation


def data_augmentation(images_tensor, num_of_passes):
    # Define probabilities for each transformation
    prob_mirror = 0.9
    prob_zoom = 0.5
    prob_rotation = 0.2
    prob_contrast = 0.5
    prob_brightness = 0.5

    # Apply data augmentation to each image in the array
    augmented_images = []

    for img in images_tensor:
        for _ in range(num_of_passes):
            transformed_img = img.clone()

            # Apply mirror transformation
            random_mirror = random.random()
            if random_mirror < prob_mirror:
                transformed_img = transforms.RandomHorizontalFlip()(transformed_img)

            # Apply zoom transformation
            random_zoom = random.random()
            if random_zoom < prob_zoom:
                transformed_img = transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.1), ratio=(0.9, 1.1))(transformed_img)

            # Apply rotation transformation
            random_rotation = random.random()
            if random_rotation < prob_rotation:
                transformed_img = transforms.RandomRotation(degrees=(-20, 20))(transformed_img)


            # New Augments
                
            # Apply contrast transformation
            random_contrast = random.random()
            if random_contrast < prob_contrast:
                transformed_img = transforms.ColorJitter(contrast=(0.5, 1.5))(transformed_img)

            # Apply brightness transformation
            random_brightness = random.random()
            if random_brightness < prob_brightness:
                transformed_img = transforms.ColorJitter(brightness=(0.2, 2))(transformed_img)



            augmented_images.append(transformed_img)

    # Convert the list of augmented images to a PyTorch tensor
    augmented_images_tensor = torch.stack(augmented_images)

    # Concatenate original and augmented images
    combined_images = images_tensor + list(augmented_images_tensor)

    return combined_images



########################################### Model Architectures

class CNNModel(nn.Module):
    def __init__(self, input_channels=3, input_size=512):  # Adjust input_channels and input_size based on your images
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the fully-connected layer input based on the architecture and input size
        fc_input_size = 64 * (input_size // 4) * (input_size // 4)

        self.fc = nn.Linear(fc_input_size, 1)  # Adjust the input size based on your needs

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing it to the fully-connected layer
        output = self.fc(x)
        return output
    






##################### Larger CNN 

class ABRankingFCNetwork(nn.Module):
    def __init__(self, minio_client, input_size=1280, hidden_sizes=[512, 256], input_type="input_clip" , output_size=1, 
                 output_type="sigma_score", dataset="cybernetics", learning_rate=0.001, validation_split=0.2):
        
        super(ABRankingFCNetwork, self).__init__()
        # set device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        # Define the multi-layered model architecture
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Adding the final layer (without an activation function for linear output)
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, data, batch_size=64):
        # Convert the features array into a PyTorch Tensor
        features_tensor = torch.Tensor(np.array(data)).to(self._device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # List to hold all predictions
        predictions = []

        # Perform prediction in batches
        with torch.no_grad():
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i + batch_size]  # Extract a batch
                outputs = self.model(batch)  # Get predictions for this batch
                predictions.append(outputs)

        # Concatenate all predictions and convert to a NumPy array
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

        return predictions 




class Clip_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Clip_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x





def get_dataset_from_id(id_class,data_augment_passes,label_value):

    images_paths = get_tag_jobs(id_class)
    ocult_images = []


    for path in images_paths:
        ocult_images.append(get_image(path))


    # Transforme into tansors
    ocult_images = [transform(img) for img in ocult_images]


    # Call your data_augmentation function
    ocult_images = data_augmentation(ocult_images, data_augment_passes)


    print("Occult lenght : ",len(ocult_images))


    # Create labels
    label_value = label_value
    labels_occult = [label_value] * len(ocult_images)

    data_occcult = []
    for image in ocult_images:
        data_occcult.append((image, label_value))

    ocult_images = data_occcult
    num_samples_ocult = len(ocult_images)
    print("the number of samples in ocult ", num_samples_ocult)
    train_size_ocult = int(0.8 * num_samples_ocult)
    val_size_ocult = num_samples_ocult - train_size_ocult
    train_set_ocult, val_set_ocult = random_split(ocult_images, [train_size_ocult, val_size_ocult])
    return train_set_ocult,val_set_ocult




def get_combined_adv_dataset_from_id_array(id_classes,data_augment_passes,label_value):
    i = 0
    for class_id in id_classes:
        images_paths[i] = get_tag_jobs(class_id)
        i += 1



    ocult_images = []

    for j in range(i):
        for path in images_paths[j]:
            ocult_images.append(get_image(path))


    # Transforme into tansors
    ocult_images = [transform(img) for img in ocult_images]


    # Call your data_augmentation function
    ocult_images = data_augmentation(ocult_images, data_augment_passes)


    print("Occult lenght : ",len(ocult_images))


    # Create labels
    label_value = label_value
    labels_occult = [label_value] * len(ocult_images)

    data_occcult = []
    for image in ocult_images:
        data_occcult.append((image, label_value))

    ocult_images = data_occcult
    num_samples_ocult = len(ocult_images)
    print("the number of samples in ocult ", num_samples_ocult)
    train_size_ocult = int(0.8 * num_samples_ocult)
    val_size_ocult = num_samples_ocult - train_size_ocult
    train_set_ocult, val_set_ocult = random_split(ocult_images, [train_size_ocult, val_size_ocult])
    return train_set_ocult,val_set_ocult












##################################### Get a sample of ID images
def get_real_images_and_labels(n_real,train_dataset):
  # Shuffle the dataset randomly
  # Get all indices
  indices = list(range(len(train_dataset)))

  # Shuffle the indices
  random.shuffle(indices)

  # Sample n_real images and labels using the shuffled indices
  real_images, real_labels = zip(*[train_dataset[i] for i in indices[:n_real]])

  # Convert to tensors
  real_images = torch.stack(real_images)  # Assuming images are already tensors
  real_labels = torch.tensor(real_labels)

  real_images = real_images.to(device)
  real_labels = real_labels.to(device)
  return real_images, real_labels




###################################
total_losses = []
class_losses = []
cdiv_losses = []
reg_losses = []

real_scores_s = []
fake_scores_s = []


class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size = batchsize_x, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = Clip_NN(input_size = 1280, hidden_size = 512, output_size =1) #todo
        #self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx): #maybe add the adv loader
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        #print("the _ is ",_)
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples #Give more steps later
        fake_imgs, fake_labels = next(iter(adv_loader))  #train_loader_noncats #train_loader_noncats  #val_loader_dog        # self.sampler.sample_new_exmps(steps=256, step_size=5)
        fake_imgs = fake_imgs.to(device)
        fake_labels = fake_labels.to(device)

        _.to(device)
        #print("The shapes are ", real_imgs.shape)
        #print("The shapes are ", fake_imgs)
        # Pass all images through the model
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        all_scores = self.cnn(all_imgs)

        # Separate real and fake scores and probabilities
        real_scores, fake_scores = all_scores.chunk(2, dim=0)
        #real_probs, fake_probs = class_probs.chunk(2, dim=0)

        # Calculate CD loss
        cdiv_loss = fake_scores.mean() - real_scores.mean()

        # Calculate classification loss (assuming softmax output)
        #class_loss = nn.CrossEntropyLoss()(real_probs, _)

        # regression loss

        reg_loss =(real_scores ** 2 + fake_scores ** 2).mean()

        # Combine losses and backpropagate
        alphaW = 1  # Adjust weight for cdiv_loss
        alphaY = 0.2  # Adjust weight for reg_loss
        total_loss =  ((alphaW) * cdiv_loss) + (alphaY * reg_loss)
        #total_loss = cdiv_loss + class_loss

        # Logging
        self.log('total loss', total_loss)
        #self.log('loss_regularization', class_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', 0)
        self.log('metrics_avg_fake', 0)

        #print(('total loss', total_loss.item()))
        #print(('cls loss', class_loss.item()))
        #print(('cdiv loss', cdiv_loss.item()))
        total_losses.append(total_loss.item())
        #class_losses.append(class_loss.item())
        cdiv_losses.append(cdiv_loss.item())
        reg_losses.append(reg_loss.item())

        real_scores_s.append(real_scores.mean().item())
        fake_scores_s.append(fake_scores.mean().item())
        return total_loss


        # ##""

        # # Predict energy score for all images
        # inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        # real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # # Calculate losses
        # reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        # cdiv_loss = fake_out.mean() - real_out.mean()
        # loss = reg_loss + cdiv_loss

        # # Logging
        # self.log('loss', loss)
        # self.log('loss_regularization', reg_loss)
        # self.log('loss_contrastive_divergence', cdiv_loss)
        # self.log('metrics_avg_real', real_out.mean())
        # self.log('metrics_avg_fake', fake_out.mean())
        # return loss



    # def validation_step(self, batch, batch_idx):
    #     # For validating, we calculate the contrastive divergence between purely random images and unseen examples
    #     # Note that the validation/test step of energy-based models depends on what we are interested in the model
    #     real_imgs, _ = batch
    #     fake_imgs = torch.rand_like(real_imgs) * 2 - 1

    #     inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
    #     real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

    #     cdiv = fake_out.mean() - real_out.mean()
    #     self.log('val_contrastive_divergence', cdiv)
    #     self.log('val_fake_out', fake_out.mean())
    #     self.log('val_real_out', real_out.mean())


    def validation_step(self, batch, batch_idx):

      # Validate with real images only (no noise/fakes)
      real_imgs, labels = batch

      # Pass through model to get scores and probabilities
      all_scores = self.cnn(real_imgs)

      # Calculate CD loss (optional, adjust if needed)
      cdiv = all_scores.mean()  # Modify based on scores or probabilities

      # Calculate classification metrics
      #predicted_labels = torch.argmax(class_probs, dim=1)
      #accuracy = (predicted_labels == labels).float().mean()
      #precision, recall, f1, _ = precision_recall_fscore(predicted_labels, labels, average='weighted')

      # Log metrics
      #print('val_accuracy', accuracy)
      self.log('val_contrastive_divergence', cdiv)
      #self.log('val_accuracy', accuracy)
      #self.log('val_precision', precision)
      #self.log('val_recall', recall)
      #self.log('val_f1', f1)





################### Call Backs

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):

        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
                trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=5, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    


class SamplerCallback(pl.Callback):

    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs             # Number of images to plot
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)


class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)











# Train model
def train_model(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=15,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("epoch")
                                   ])
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")

    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(model, train_loader, val_loader)

    #model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # No testing as we are more interested in other properties

    return model


################# Compare image A with image B energy wise
@torch.no_grad()
def compare_images_value_purevalue(img1, img2):
    # Pass the first image through the CNN model and get its score
    score1 = model.cnn(img1.unsqueeze(0).to(model.device)).cpu()

    # Pass the second image through the CNN model and get its score
    score2 = model.cnn(img2.unsqueeze(0).to(model.device)).cpu()

    # grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, pad_value=0.5, padding=2)
    # grid = grid.permute(1, 2, 0)
    # print(f"Score original image: {score1.item():4.2f}")
    # print(f"Score transformed image: {score2.item():4.2f}")
    return score1.item(), score2.item()

@torch.no_grad()
def compare_images_show(img1, img2):
    imgs = torch.stack([img1, img2], dim=0).to(model.device)
    score1, score2 = model.cnn(imgs).cpu().chunk(2, dim=0) # model.cnn(imgs)[0].cpu().chunk(2, dim=0)
    #class1, class2 = model.cnn(imgs)[1].cpu().chunk(2, dim=0)
    grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, pad_value=0.5, padding=2)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,4))
    plt.imshow(grid)
    plt.xticks([(img1.shape[2]+2)*(0.5+j) for j in range(2)],
               labels=[f"ID: {score1.item():4.2f}", f"OOD: {score2.item():4.2f}"])
    plt.yticks([])
    plt.savefig("output/comparaison_1.png")

    # Save the figure to a file
    bufx = io.BytesIO()
    plt.savefig(bufx, format='png')
    bufx.seek(0)

    # upload the photo
    
    minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
                "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
                None)
    minio_path="environmental/output/my_test"
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
    minio_path= minio_path + "/compare_id_vs_ood" +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, bufx)
    # Remove the temporary file
    os.remove("output/comparaison_1.png")
    # Clear the current figure
    plt.clf()
    return score1.item(), score2.item()



# Evaluation OOD capabalities


def energy_evaluation(training_loader,adv_loader):
    
    some_a = 0
    some_b = 0
    # load training set images
    test_imgs, _ = next(iter(training_loader))
    

    # load adv set images
    fake_imgs, _ = next(iter(adv_loader)) # val_loader_dog  val_ood_loader val val_loader_noncats val_loader
    

    rangeX = len(test_imgs)
    for i in range(rangeX):
        a,b =  compare_images_value_purevalue(test_imgs[i].to(model.device),fake_imgs[i].to(model.device))
        some_a += a
        some_b += b

    some_a = some_a / rangeX
    some_b = some_b / rangeX

    print(f"Score in distribution : {some_a:4.2f}")
    print(f"Score OOD : {some_b:4.2f}")




def energy_evaluation_with_pictures(training_loader,adv_loader):
    
    some_a = 0
    some_b = 0
    # load training set images
    test_imgs, _ = next(iter(training_loader))
    

    # load adv set images
    fake_imgs, _ = next(iter(adv_loader)) # val_loader_dog  val_ood_loader val val_loader_noncats val_loader
    
    # print("tes_imgs shape : ", test_imgs.shape)
    # print("fake_imgs shape : ", fake_imgs.shape)


    rangeX = 16
    for i in range(rangeX):
        a,b =  compare_images_show(test_imgs[i].to(model.device),fake_imgs[i].to(model.device))
        some_a += a
        some_b += b

    some_a = some_a / rangeX
    some_b = some_b / rangeX










#######################################################################################################################################################
################################################################    Run code here      ################################################################
#######################################################################################################################################################




##################### Load images

# Transformations: # don't use greyscale
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize after grayscale conversion
])


##################### Load images



#cybernetic: 35, occult: 39


# ################################################################### Load occults images
# images_paths_ClassA = get_tag_jobs(39)


# #Get clips vectors
# ocult_clips = []
# ocult_clips = get_clip_vectors(images_paths_ClassA )



# # ADD Labels
# data_occcult_clips = []
# for clip in ocult_clips:
#     data_occcult_clips.append((clip, 1))

# # Dataset Size
# print("Occult lenght : ",len(data_occcult_clips))


# ocult_clips = data_occcult_clips
# num_samples_ocult = len(ocult_clips)
# print("the number of samples in ocult ", num_samples_ocult)

# train_size_ocult = int(0.8 * num_samples_ocult)
# val_size_ocult = num_samples_ocult - train_size_ocult

# train_set_ocult, val_set_ocult = random_split(ocult_clips, [train_size_ocult, val_size_ocult])

# train_loader_clip_occult = data.DataLoader(train_set_ocult, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
# val_loader_clip_ocult= data.DataLoader(val_set_ocult, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)




# # ################################################################################ Load cybernetics images
# images_paths_ClassB = get_tag_jobs(35)

# cyber_clips = []
# cyber_clips = get_clip_vectors(images_paths_ClassB )
# data_cyber_clips = []
# for clip in cyber_clips:
#     data_cyber_clips.append((clip, 0))




# print("Occult lenght : ",len(data_cyber_clips))


# cyber_clips = data_cyber_clips
# num_samples_cyber = len(cyber_clips)
# print("the number of samples in ocult ", num_samples_cyber)
# train_size_cyber = int(0.8 * num_samples_cyber)
# val_size_cyber = num_samples_cyber - train_size_cyber



# train_set_cyber, val_set_cyber = random_split(cyber_clips, [train_size_cyber, val_size_cyber])
# train_loader_clip_cyber = data.DataLoader(train_set_cyber, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
# val_loader_clip_cyber= data.DataLoader(val_set_cyber, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)


# # Set loaders
# train_loader = train_loader_clip_occult
# val_loader = val_loader_clip_ocult
# adv_loader = train_loader_clip_cyber



# Load occult images
images_paths_ClassA = get_tag_jobs(39)
ocult_clips = get_clip_vectors(images_paths_ClassA)
#data_occcult_clips = ocult_clips
data_occcult_clips = [(clip, 1) for clip in ocult_clips]
print("Occult length:", len(data_occcult_clips))

# Split and create data loaders for occult
num_samples_ocult = len(data_occcult_clips)
train_size_ocult = int(0.8 * num_samples_ocult)
val_size_ocult = num_samples_ocult - train_size_ocult
train_set_ocult, val_set_ocult = random_split(data_occcult_clips, [train_size_ocult, val_size_ocult])

train_loader_clip_occult = data.DataLoader(train_set_ocult, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_clip_ocult = data.DataLoader(val_set_ocult, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

# Load cybernetics images
images_paths_ClassB = get_tag_jobs(35)
cyber_clips = get_clip_vectors(images_paths_ClassB)
#data_cyber_clips = cyber_clips
data_cyber_clips = [(clip, 0) for clip in cyber_clips]

print("Cybernetics length:", len(data_cyber_clips))

# Split and create data loaders for cybernetics
num_samples_cyber = len(data_cyber_clips)
train_size_cyber = int(0.8 * num_samples_cyber)
val_size_cyber = num_samples_cyber - train_size_cyber
train_set_cyber, val_set_cyber = random_split(data_cyber_clips, [train_size_cyber, val_size_cyber])

train_loader_clip_cyber = data.DataLoader(train_set_cyber, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_clip_cyber = data.DataLoader(val_set_cyber, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

# Set loaders
train_loader = train_loader_clip_occult
val_loader = val_loader_clip_ocult
adv_loader = train_loader_clip_cyber



print("trainloader length:", len(train_loader))


# Fetch the first batch
first_batch_data, first_batch_labels = next(iter(train_loader_clip_occult))

# Print the shape of the first batch data and labels
print(f"Shape of the first batch data: {first_batch_data.shape}")
print(f"Shape of the first batch labels: {first_batch_labels.shape}")

# Train
model = train_model(img_shape=(1,1280),
                    batch_size=train_loader.batch_size,
                    lr=0.001,
                    beta1=0.0)



# Plot


epochs = range(1, len(total_losses) + 1)  


# Create subplots grid (3 rows, 1 column)
fig, axes = plt.subplots(4, 1, figsize=(10, 24))

# Plot each loss on its own subplot
axes[0].plot(epochs, total_losses, label='Total Loss')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(True)

# axes[1].plot(epochs, class_losses, label='Classification Loss')
# axes[1].set_xlabel('Steps')
# axes[1].set_ylabel('Loss')
# axes[1].set_title('Classification Loss')
# axes[1].legend()
# axes[1].grid(True)

axes[1].plot(epochs, cdiv_losses, label='Contrastive Divergence Loss')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Loss')
axes[1].set_title('Contrastive Divergence Loss')
axes[1].legend()
axes[1].grid(True)


axes[2].plot(epochs, reg_losses , label='Regression Loss')
axes[2].set_xlabel('Steps')
axes[2].set_ylabel('Loss')
axes[2].set_title('Regression Loss')
axes[2].legend()
axes[2].grid(True)

# Plot real and fake scores on the fourth subplot
axes[3].plot(epochs, real_scores_s, label='Real Scores')
axes[3].plot(epochs, fake_scores_s, label='Fake Scores')
axes[3].set_xlabel('Steps')
axes[3].set_ylabel('Score')  # Adjust label if scores represent a different metric
axes[3].set_title('Real vs. Fake Scores')
axes[3].legend()
axes[3].grid(True)

# Adjust spacing between subplots for better visualization
plt.tight_layout()

plt.savefig("output/loss_tracking_per_step.png")

# Save the figure to a file
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# upload the graph report
minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss" +date_now+".png"
cmd.upload_data(minio_client, 'datasets', minio_path, buf)
# Remove the temporary file
os.remove("output/loss_tracking_per_step.png")
# Clear the current figure
plt.clf()




# energy_evaluation_with_pictures(val_loader,adv_loader)
# energy_evaluation_with_pictures(val_loader,val_cifarset_loader)
# energy_evaluation_with_pictures(val_loader,val_ood_loader)

#val_ood_loader
##### Value eval
print("Occult VS Cyber")
energy_evaluation(val_loader,adv_loader)
# print("Occult VS Cifar")
# energy_evaluation(val_loader,val_cifarset_loader)
# print("Occult VS SVHN")
# energy_evaluation(val_loader,val_ood_loader)












#################################################################################################### Entrainement inverse




# train_loader = train_loader_set_cyber
# val_loader = val_loader_set_cyber
# adv_loader =  val_loader_advtrain


# Train
model2 = train_model(img_shape=(3,512,512),
                    batch_size=train_loader_set_cyber.batch_size,
                    lr=0.001,
                    beta1=0.0)



# # Initialize the model
# model = Clip_NN(input_size=1280, hidden_size=512, output_size=1)

# # Create a dummy input tensor of the correct shape [1, 1280]
# input_tensor = torch.randn(1, 1280)

# # Forward pass
# output = model(input_tensor)


# print(output.shape)  # This should print torch.Size([1, 1])










# modelsave = model
# model = model2
# # Plot


# epochs = range(1, len(total_losses) + 1)  


# # Create subplots grid (3 rows, 1 column)
# fig, axes = plt.subplots(4, 1, figsize=(10, 24))

# # Plot each loss on its own subplot
# axes[0].plot(epochs, total_losses, label='Total Loss')
# axes[0].set_xlabel('Steps')
# axes[0].set_ylabel('Loss')
# axes[0].set_title('Total Loss')
# axes[0].legend()
# axes[0].grid(True)

# # axes[1].plot(epochs, class_losses, label='Classification Loss')
# # axes[1].set_xlabel('Steps')
# # axes[1].set_ylabel('Loss')
# # axes[1].set_title('Classification Loss')
# # axes[1].legend()
# # axes[1].grid(True)

# axes[1].plot(epochs, cdiv_losses, label='Contrastive Divergence Loss')
# axes[1].set_xlabel('Steps')
# axes[1].set_ylabel('Loss')
# axes[1].set_title('Contrastive Divergence Loss')
# axes[1].legend()
# axes[1].grid(True)


# axes[2].plot(epochs, reg_losses , label='Regression Loss')
# axes[2].set_xlabel('Steps')
# axes[2].set_ylabel('Loss')
# axes[2].set_title('Regression Loss')
# axes[2].legend()
# axes[2].grid(True)

# # Plot real and fake scores on the fourth subplot
# axes[3].plot(epochs, real_scores_s, label='Real Scores')
# axes[3].plot(epochs, fake_scores_s, label='Fake Scores')
# axes[3].set_xlabel('Steps')
# axes[3].set_ylabel('Score')  # Adjust label if scores represent a different metric
# axes[3].set_title('Real vs. Fake Scores')
# axes[3].legend()
# axes[3].grid(True)

# # Adjust spacing between subplots for better visualization
# plt.tight_layout()

# plt.savefig("output/loss_tracking_per_step.png")

# # Save the figure to a file
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# buf.seek(0)

# # upload the graph report
# minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss_reversed" +date_now+".png"
# cmd.upload_data(minio_client, 'datasets', minio_path, buf)
# # Remove the temporary file
# os.remove("output/loss_tracking_per_step.png")
# # Clear the current figure
# plt.clf()



# energy_evaluation_with_pictures(val_loader,adv_loader)
# energy_evaluation_with_pictures(val_loader,val_cifarset_loader)
# energy_evaluation_with_pictures(val_loader,val_ood_loader)

# #val_ood_loader
# ##### Value eval
# print("Occult VS Cyber")
# energy_evaluation(val_loader,adv_loader)
# print("Occult VS Cifar")
# energy_evaluation(val_loader,val_cifarset_loader)
# print("Occult VS SVHN")
# energy_evaluation(val_loader,val_ood_loader)










# import torchvision.transforms.functional as TF
# from torchvision.utils import make_grid
# minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
#             "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
#             None)
# minio_path="environmental/output/my_test"
# date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
# data_iter = iter(val_loader)
# images, labels = next(data_iter)
# # Create a grid of images
# image_grid = make_grid(images[:16], nrow=4, padding=2, normalize=True)

# # Convert tensor to numpy array and transpose channels
# image_grid_np = TF.to_pil_image(image_grid)
# plt.imshow(image_grid_np)
# plt.axis('off')
# plt.savefig("output/sample_from_occult.png")
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# buf.seek(0)

# # upload the graph report
# minio_path= minio_path + "/sample_from_occult" +date_now+".png"
# cmd.upload_data(minio_client, 'datasets', minio_path, buf)
# # Remove the temporary file
# os.remove("output/sample_from_occult.png")
# # Clear the current figure
# plt.clf()





#################################################################################################### Combine occult and Cyber vs all
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# train_comb, val_comb =  get_combined_adv_dataset_from_id_array([35,39],5,0)
# train_comb_loader = data.DataLoader(train_comb, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
# val_comb_loader = data.DataLoader(val_comb, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

# # ADV compare = val_loader_advtrain



# def energy_evaluation_comb(training_loader,adv_loader):
    
#     some_a = 0
#     some_b = 0
#     preci = 0
#     cpt = 0
#     # load training set images
#     test_imgs, _ = next(iter(training_loader))
    

#     # load adv set images
#     fake_imgs, _ = next(iter(adv_loader)) # val_loader_dog  val_ood_loader val val_loader_noncats val_loader
    

#     rangeX = len(test_imgs)
#     for i in range(rangeX):
#         a,b =  compare_images_value_purevalue_comb(test_imgs[i].to(model.device),fake_imgs[i].to(model.device))
#         if a > b:
#             preci += 1
#         cpt += 1
#         some_a += a
#         some_b += b

#     some_a = some_a / rangeX
#     some_b = some_b / rangeX

#     print(f"Score in distribution : {some_a:4.2f}")
#     print(f"Score OOD : {some_b:4.2f}")
#     print(f"Accuracy : ", preci , " / ",cpt)



# ################# Compare image A with image B energy wise
# @torch.no_grad()
# def compare_images_value_purevalue_comb(img1, img2):
#     # Pass the first image through the CNN model and get its score
#     score1 = model.cnn(img1.unsqueeze(0).to(model.device)).cpu()
#     score1 += modelsave(img1.unsqueeze(0).to(model.device)).cpu()
#     # Pass the second image through the CNN model and get its score
#     score2 = model.cnn(img2.unsqueeze(0).to(model.device)).cpu()
#     score2 += modelsave.cnn(img2.unsqueeze(0).to(model.device)).cpu()
#     grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, pad_value=0.5, padding=2)
#     grid = grid.permute(1, 2, 0)
#     # print(f"Score original image: {score1.item():4.2f}")
#     # print(f"Score transformed image: {score2.item():4.2f}")
#     return score1.item(), score2.item()





# def energy_evaluation_with_pictures_comb(training_loader,adv_loader):
    
#     some_a = 0
#     some_b = 0
#     # load training set images
#     test_imgs, _ = next(iter(training_loader))
    

#     # load adv set images
#     fake_imgs, _ = next(iter(adv_loader)) # val_loader_dog  val_ood_loader val val_loader_noncats val_loader
    
#     # print("tes_imgs shape : ", test_imgs.shape)
#     # print("fake_imgs shape : ", fake_imgs.shape)


#     rangeX = 64
#     for i in range(rangeX):
#         a,b =  compare_images_show_comb(test_imgs[i].to(model.device),fake_imgs[i].to(model.device))
#         some_a += a
#         some_b += b

#     some_a = some_a / rangeX
#     some_b = some_b / rangeX



# @torch.no_grad()
# def compare_images_show_comb(img1, img2):
#     imgs = torch.stack([img1, img2], dim=0).to(model.device)
#     score1, score2 = model.cnn(imgs).cpu().chunk(2, dim=0) # model.cnn(imgs)[0].cpu().chunk(2, dim=0)
#     score1b, score2b = modelsave.cnn(imgs).cpu().chunk(2, dim=0) # model.cnn(imgs)[0].cpu().chunk(2, dim=0)
#     score1 += score1b
#     score2 += score2b
#     #class1, class2 = model.cnn(imgs)[1].cpu().chunk(2, dim=0)
#     grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, pad_value=0.5, padding=2)
#     grid = grid.permute(1, 2, 0)
#     plt.figure(figsize=(4,4))
#     plt.imshow(grid)
#     plt.xticks([(img1.shape[2]+2)*(0.5+j) for j in range(2)],
#                labels=[f"ID: {score1.item():4.2f}", f"OOD: {score2.item():4.2f}"])
#     plt.yticks([])
#     plt.savefig("output/comparaison_1.png")

#     # Save the figure to a file
#     bufx = io.BytesIO()
#     plt.savefig(bufx, format='png')
#     bufx.seek(0)
#     return score1, score2




# print('combined stuff')

# energy_evaluation_with_pictures_comb(val_comb_loader,val_loader_advtrain)
# energy_evaluation_with_pictures_comb(val_comb_loader,val_cifarset_loader)
# energy_evaluation_with_pictures_comb(val_comb_loader,val_ood_loader)
# energy_evaluation_with_pictures_comb(val_comb_loader,val_loader_set_ocult)
# energy_evaluation_with_pictures_comb(val_comb_loader,val_loader_set_cyber)



# #val_ood_loader
# ##### Value eval
# print("Occult and Cyber VS Full ADV")
# energy_evaluation_comb(val_comb_loader,val_loader_advtrain)
# print("Occult and Cyber VS Cifar")
# energy_evaluation_comb(val_comb_loader,val_cifarset_loader)
# print("Occult and Cyber VS SVHN")
# energy_evaluation_comb(val_comb_loader,val_ood_loader)
# print("Full Occult VS ADV")
# energy_evaluation_comb(val_loader_set_ocult,val_loader_advtrain)
# print("Full Cyber VS ADV")
# energy_evaluation_comb(val_loader_set_cyber,val_loader_advtrain)
# print("Full Occult VS Cifar")
# energy_evaluation_comb(val_loader_set_ocult,val_cifarset_loader)
# print("Full Cyber VS Cifar")
# energy_evaluation_comb(val_loader_set_cyber,val_cifarset_loader)
# print("Full Occult VS SVHN")
# energy_evaluation_comb(val_loader_set_ocult,val_ood_loader)
# print("Full Cyber VS SVHN")
# energy_evaluation_comb(val_loader_set_cyber,val_ood_loader)