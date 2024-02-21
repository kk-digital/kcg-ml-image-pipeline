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




matplotlib.rcParams['lines.linewidth'] = 2.0

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial8"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



######### DL old model to be removed
import urllib.request
from urllib.error import HTTPError

class ToGrayscale(transforms.ToTensor):
    def __call__(self, img):
        img = super().__call__(img)  # Convert to tensor with first ToTensor()
        return img.mean(dim=0, keepdim=True)  # Average across channels

# Transformations: # don't use greyscale
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize after grayscale conversion
])



# Get Ciafr 10
# Download CIFAR-10 training dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)



# Loading the training dataset. We need to split it into a training and validation part
train_set = CIFAR10(root='./data', train=True, transform=transform, download=True)

# Loading the test set
test_set = CIFAR10(root='./data', train=False, transform=transform, download=True)


# cat_index = np.where(train_set.reshape(-1) == 3)
# dog_index= np.where(train_set.reshape(-1) == 5)

dog_idx = np.where((np.array(train_set.targets) == 5) )[0]
dog_ds = torch.utils.data.Subset(train_set, dog_idx)

cat_idx = np.where((np.array(train_set.targets) == 3))[0]
notcat_idx = np.where((np.array(train_set.targets) != 3) )[0]


# Define cat and not-cat labels
cat_target = 1
not_cat_target = 0


# Caution: This modifies the original dataset directly
for i in range(len(train_set.targets)):
    if train_set.targets[i] == 3:  # Check for cat class (index 3)
        train_set.targets[i] = cat_target
    else:
        train_set.targets[i] = not_cat_target


notcat_ds = torch.utils.data.Subset(train_set, notcat_idx)

num_samples = len(train_set)
print("the number of samples is ", num_samples)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size


train_set, val_set = random_split(train_set, [train_size, val_size])



train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
dog_loader = data.DataLoader(notcat_ds, batch_size=64, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

for images, _ in train_loader:
    # Unpack the batch
    images = images.squeeze(0)  # Assuming you want to print shape per image
    #print(f"Grayscale image shape: {images.shape}")
    break  



# ##################### NEW Classifier V2
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()

#         # Convolutional layers and activation functions
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Fully-connected layers and activation functions
#         self.fc1 = nn.Linear(64 * 8 * 8, 1024)
#         self.relu3 = nn.ReLU()

#         # Energy prediction branch
#         self.fc_energy = nn.Linear(1024, 1)  # Predict a single energy score

#         # Classification branch
#         self.fc2 = nn.Linear(1024, 1)
#         self.softmax = nn.Sigmoid() # Apply softmax for class probabilities

#     def forward(self, x):
#         # Feature extraction using convolutional layers
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)

#         # Feature processing for both branches
#         shared_features = self.relu3(self.fc1(x))

#         # Energy branch
#         energy = self.fc_energy(shared_features)  # Output energy score

#         # Classification branch
#         logits = self.fc2(shared_features)
#         probs = self.softmax(logits)  # Output class probabilities

#         return energy, probs
    



##################### NEW Classifier V2
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers and activation functions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully-connected layers and activation functions
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.relu3 = nn.ReLU()

        # Energy prediction branch
        self.fc_energy = nn.Linear(1024, 1)  # Predict a single energy score

        # Classification branch
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax for class probabilities

    def forward(self, x):
        # Feature extraction using convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)

        # Feature processing for both branches
        shared_features = self.relu3(self.fc1(x))

        # Energy branch
        energy = self.fc_energy(shared_features)  # Output energy score

        # Classification branch
        logits = self.fc2(shared_features)
        probs = self.softmax(logits)  # Output class probabilities

        return energy, probs












##### Get real images
    
def get_real_images_and_labels(n_real):
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


class Sampler:

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]


    def sample_new_exmps(self, steps=120, step_size=5):
      # Choose 80% of the batch from real images, 20% generate from scratch
      n_real = int(self.sample_size * 0.8)
      n_new = self.sample_size - n_real

      # Get real images and labels from your dataset
      real_imgs, real_labels = get_real_images_and_labels(n_real)

      # Generate new images with noise
      rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
      rand_imgs = rand_imgs.to(device)
      # Combine real and fake images with associated labels
      inp_imgs = torch.cat([real_imgs, rand_imgs], dim=0)
      labels = torch.cat([real_labels, torch.zeros(n_new).to(device)], dim=0)

      # Perform MCMC sampling
      inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

      # Add new images to the buffer and remove old ones if needed
      # ... (update buffer logic considering mixed data) ...

      return inp_imgs, labels

    # def sample_new_exmps(self, steps=60, step_size=10):
    #     """
    #     Function for getting a new batch of "fake" images.
    #     Inputs:
    #         steps - Number of iterations in the MCMC algorithm
    #         step_size - Learning rate nu in the algorithm above
    #     """
    #     # Choose 95% of the batch from the buffer, 5% generate from scratch
    #     n_new = np.random.binomial(self.sample_size, 0.05)
    #     rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
    #     old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
    #     inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

    #     # Perform MCMC sampling
    #     inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

    #     # Add new images to the buffer and remove old ones if needed
    #     self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
    #     self.examples = self.examples[:self.max_len]
    #     return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=120, step_size=5, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.010) # (0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0) # (min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = model(inp_imgs)  # Tuple containing savedx and x
            out_imgs = -out_imgs[0]  # Use the first element (savedx)
            #out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
        






total_losses = []
class_losses = []
cdiv_losses = []
reg_losses = []

real_scores_s = []
fake_scores_s = []


class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = CNNModel(**CNN_args)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
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

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        #print("the _ is ",_)
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs, fake_labels = self.sampler.sample_new_exmps(steps=60, step_size=10)
        #print("The shapes are ", real_imgs.shape)
        #print("The shapes are ", fake_imgs)
        # Pass all images through the model
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        all_scores, class_probs = self.cnn(all_imgs)

        # Separate real and fake scores and probabilities
        real_scores, fake_scores = all_scores.chunk(2, dim=0)
        real_probs, fake_probs = class_probs.chunk(2, dim=0)

        # Calculate CD loss
        cdiv_loss = fake_scores.mean() - real_scores.mean()

        # Calculate classification loss (assuming softmax output)
        class_loss = nn.CrossEntropyLoss()(real_probs, _)

        # regression loss

        reg_loss =(real_scores ** 2 + fake_scores ** 2).mean()

        # Combine losses and backpropagate
        alphaW = 0.1  # Adjust weight for cdiv_loss
        alphaY = 0.1  # Adjust weight for reg_loss
        total_loss = (alphaW * class_loss) + ((1 - alphaW) * cdiv_loss) + (alphaY * reg_loss)
        #total_loss = cdiv_loss + class_loss

        # Logging
        self.log('total loss', total_loss)
        self.log('loss_regularization', class_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', 0)
        self.log('metrics_avg_fake', 0)

        #print(('total loss', total_loss.item()))
        #print(('cls loss', class_loss.item()))
        #print(('cdiv loss', cdiv_loss.item()))
        total_losses.append(total_loss.item())
        class_losses.append(class_loss.item())
        cdiv_losses.append(cdiv_loss.item())
        reg_losses.append(reg_loss.item())

        real_scores_s.append(real_scores.mean().item())
        fake_scores_s.append(fake_scores.mean().item())



        return total_loss


    def validation_step(self, batch, batch_idx):
      # ... (existing data preprocessing) ...

      # Validate with real images only (no noise/fakes)
      real_imgs, labels = batch

      # Pass through model to get scores and probabilities
      all_scores, class_probs = self.cnn(real_imgs)

      # Calculate CD loss (optional, adjust if needed)
      cdiv = all_scores.mean()  # Modify based on scores or probabilities

      # Calculate classification metrics
      predicted_labels = torch.argmax(class_probs, dim=1)
      accuracy = (predicted_labels == labels).float().mean()
      #precision, recall, f1, _ = precision_recall_fscore(predicted_labels, labels, average='weighted')

      # Log metrics
      #print('val_accuracy', accuracy)
      self.log('val_contrastive_divergence', cdiv)
      self.log('val_accuracy', accuracy)
      #self.log('val_precision', precision)
      #self.log('val_recall', recall)
      #self.log('val_f1', f1)



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
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
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





def train_model(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=5,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("epoch")
                                   ])
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    if 5 > 99: #os.path.isfile(pretrained_filename)
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_loader, val_loader)

        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # No testing as we are more interested in other properties

    return model


from collections import namedtuple

print("################ Training started ################")
model = train_model(img_shape=(3,32,32),
                    batch_size=train_loader.batch_size,
                    lr=1e-4,
                    beta1=0.0)

print("################ Training ended ################")




############ Graph


date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
print(date_now)


minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
minio_path="environmental/output/my_test"

epochs = range(1, len(total_losses) + 1)  

# Create subplots grid (5 rows, 1 column)
fig, axes = plt.subplots(5, 1, figsize=(10, 24))

# Plot each loss on its own subplot
axes[0].plot(epochs, total_losses, label='Total Loss')
axes[0].set_xlabel('Steps')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, class_losses, label='Classification Loss')
axes[1].set_xlabel('Steps')
axes[1].set_ylabel('Loss')
axes[1].set_title('Classification Loss')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(epochs, cdiv_losses, label='Contrastive Divergence Loss')
axes[2].set_xlabel('Steps')
axes[2].set_ylabel('Loss')
axes[2].set_title('Contrastive Divergence Loss')
axes[2].legend()
axes[2].grid(True)


axes[3].plot(epochs, reg_losses , label='Regression Loss')
axes[3].set_xlabel('Steps')
axes[3].set_ylabel('Loss')
axes[3].set_title('Regression Loss')
axes[3].legend()
axes[3].grid(True)

# Plot real and fake scores on the fourth subplot
axes[4].plot(epochs, real_scores_s, label='Real Scores')
axes[4].plot(epochs, fake_scores_s, label='Fake Scores')
axes[4].set_xlabel('Steps')
axes[4].set_ylabel('Score')  # Adjust label if scores represent a different metric
axes[4].set_title('Real vs. Fake Scores')
axes[4].legend()
axes[4].grid(True)


# Adjust spacing between subplots for better visualization
plt.tight_layout()

plt.savefig("output/loss_tracking_per_step.png")

# Save the figure to a file
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# upload the graph report
minio_path= minio_path + "/loss_tracking_per_step_" +date_now+".png"
cmd.upload_data(minio_client, 'datasets', minio_path, buf)
# Remove the temporary file
os.remove("output/loss_tracking_per_step.png")
# Clear the current figure
plt.clf()







# Image generation
model.to(device)
pl.seed_everything(42)
callback = GenerateCallback(batch_size=8, vis_steps=8, num_steps=512)
imgs_per_step = callback.generate_imgs(model)
imgs_per_step = imgs_per_step.cpu()

for i in range(imgs_per_step.shape[1]):
    step_size = callback.num_steps // callback.vis_steps
    imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
    imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, pad_value=0.5, padding=2)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(8,8))
    plt.imshow(grid)
    plt.xlabel("Generation iteration")
    plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)],
               labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
    plt.yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    minio_path_i = "environmental/output/my_test/images_generation_sample_" + str(i) +"_" +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path_i, buf)





######## Average score for random images
    
with torch.no_grad():
    rand_imgs = torch.rand((128,) + model.hparams.img_shape).to(model.device)
    rand_imgs = rand_imgs * 2 - 1.0
    rand_out = model.cnn(rand_imgs)[0].mean()
    print(f"Average score for random images: {rand_out.item():4.2f}")


######## Average score for training images
def softmax_to_class(softmax_tensor):

  # Convert tensor to numpy array for easier indexing
  class_probs = softmax_tensor.detach().cpu().numpy()

  # Find the index of the element with the highest probability
  class_index = np.argmax(class_probs)

  # Map the index to the corresponding class name using a dictionary
  class_names = {
      0: "airplane",
      1: "automobile",
      2: "bird",
      3: "cat",
      4: "deer",
      5: "dog",
      6: "frog",
      7: "horse",
      8: "ship",
      9: "truck"
  }

  return class_names[class_index]




@torch.no_grad()
def compare_images(img1, img2):
    imgs = torch.stack([img1, img2], dim=0).to(model.device)
    score1, score2 = model.cnn(imgs)[0].cpu().chunk(2, dim=0)
    class1, class2 = model.cnn(imgs)[1].cpu().chunk(2, dim=0)
    grid = torchvision.utils.make_grid([img1.cpu(), img2.cpu()], nrow=2, normalize=True, pad_value=0.5, padding=2)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,4))
    plt.imshow(grid)
    plt.xticks([(img1.shape[2]+2)*(0.5+j) for j in range(2)],
               labels=["Original image", "Transformed image"])
    plt.yticks([])
    plt.show()
    print(f"Score original image: {score1.item():4.2f}")
    print(f"Score transformed image: {score2.item():4.2f}")

    print(f"Class original image: {softmax_to_class(torch.nn.functional.softmax(class1, dim=1))}")
    print(f"Class transformed image: {softmax_to_class(torch.nn.functional.softmax(class2, dim=1))}")









# Select image from training set
test_imgs, _ = next(iter(val_loader))
exmp_img = test_imgs[5].to(model.device)



# Select image from other set
from PIL import Image
img_noisy = exmp_img + torch.randn_like(exmp_img) * 0.3
img_noisy.clamp_(min=-1.0, max=1.0)



test_imgs, _ = next(iter(dog_loader))
fake_image = test_imgs[8].to(model.device)

compare_images(exmp_img, fake_image)