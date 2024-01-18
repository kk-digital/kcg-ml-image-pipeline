from __future__ import print_function
from keras.layers import Dense, Activation, Flatten, Reshape, Permute, LocallyConnected1D
from keras.layers import Convolution2D
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import callbacks
from keras.datasets import cifar10
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import profiler




batch_size = 64 #64
nb_classes = 10
nb_epoch = 50 #50
nb_runs = 1 #5
learning_rate = 0.001 # Seggested to be the best in the paper
architecture = 'Full'  # can be one of 'Shallow', 'Full', 'TreeConnect', 'Bottleneck', 'Small' or 'RandomSparse'
modelnameprefix = "Model-16-01" + architecture


# # The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Convert class vectors to binary class matrices.
# Y_train = to_categorical(y_train, nb_classes)
# Y_test = to_categorical(y_test, nb_classes)


class RandomSparseLayer(Layer):
    def __init__(self, output_dim, percentage_masked, **kwargs):
        self.output_dim = output_dim
        self.fraction_masked = percentage_masked / 100.0
        super(RandomSparseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = (input_shape[1], self.output_dim)
        self.weight_mask = np.where(np.random.rand(*weight_shape) < self.fraction_masked, np.zeros(weight_shape),
                                    np.ones(weight_shape))
        self.weight = self.add_weight(name='weight', shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        super(RandomSparseLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return Activation('relu')(K.dot(inputs, self.weight * self.weight_mask) + self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

  # The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
# Pre-processing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# for run_id in range(nb_runs):
#     model = Sequential()
#     if architecture == 'Small':
#         model.add(Convolution2D(8, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
#         model.add(Convolution2D(8, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(16, (4, 4), padding='same', strides=2, activation='relu'))
#         model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(32, (4, 4), padding='same', strides=2, activation='relu'))
#         model.add(Flatten())
#         model.add(Dense(552))
#         model.add(Dense(256))
#     else:
#         model.add(Convolution2D(64, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
#         model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(128, (4, 4), padding='same', strides=2, activation='relu'))
#         model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(Convolution2D(256, (4, 4), padding='same', strides=2, activation='relu'))
#     if architecture == 'Shallow':
#         model.add(Flatten())
#         model.add(Dense(256))
#     elif architecture == 'Full':
#         model.add(Flatten())
#         model.add(Dense(2048))
#         model.add(Dense(256))
#     elif architecture == 'TreeConnect':
#         model.add(Reshape((128, 128)))
#         model.add(LocallyConnected1D(16, 1, activation='relu'))
#         model.add(Permute((2, 1)))
#         model.add(LocallyConnected1D(16, 1, activation='relu'))
#         model.add(Flatten())
#     elif architecture == 'Bottleneck':
#         model.add(Flatten())
#         model.add(Dense(18))
#         model.add(Dense(167))
#     elif architecture == 'RandomSparse':
#         model.add(Flatten())
#         model.add(RandomSparseLayer(2048, percentage_masked=99.22))
#         model.add(RandomSparseLayer(256, percentage_masked=93.75))

#     # ADD an FC at the and (MD)
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Dense(128))
#     model.add(Dense(64))
#     # Classification
#     model.add(Dense(nb_classes))
#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(lr=learning_rate),
#                   metrics=['accuracy'])

#     # Save stats
#     tbCallback = callbacks.TensorBoard(log_dir='./cifar/' + architecture + '_run_' + str(run_id),
#                                        histogram_freq=1, write_graph=True)
#     model.summary()


# #    from tensorflow.keras.backend import set_session
# #    config = tf.ConfigProto()
# #    config.gpu_options.allow_growth = True  # Allow GPU memory usage to grow dynamically
# #    set_session(tf.Session(config=config))


#     # Allow GPU memory usage to grow dynamically
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print(e)

#     # Record the start time
#     start_time = time.time()
#     history = model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               epochs=nb_epoch,
#               validation_data=(X_test, Y_test),
#               shuffle=True,
#               callbacks=[tbCallback])
#     K.clear_session()
#     # Record the end time
#     end_time = time.time()

#     elapsed_time = end_time - start_time
#     print(f"Training time for run {run_id + 1}: {elapsed_time:.2f} seconds")

#     # Save model
#     modelname = modelnameprefix + str(run_id) + ".keras"
#     model.save(modelname)

#     import matplotlib.pyplot as plt
#     training_loss = history.history['loss']
#     validation_loss = history.history['val_loss']

#     # plt.plot(range(1, len(training_loss) + 1), training_loss,'b', label='Training loss')
#     # plt.plot(range(1, len(validation_loss) + 1), validation_loss,'r', label='Validation loss')
#     # plt.set_title('Loss per Round')
#     # plt.set_ylabel('Loss')
#     # plt.set_xlabel('Rounds')
#     # plt.legend(['Training loss', 'Validation loss'])

#     #plt.plot(range(1, len(training_loss) + 1), training_loss, 'bo', label='Training Loss')
#     #plt.plot(range(1, len(validation_loss) + 1), validation_loss, 'r', label='Validation Loss')
#     #plt.title('Training and Validation Loss')
#     #plt.xlabel('Epochs')
#     #plt.ylabel('Loss')
#     #plt.legend()
#     #plt.show()



#     start_tinference = time.time()
#     # Load the saved model


#     # Display the model architecture
#     model.summary()

#     # Run inference on a batch of 1024
#     inference_batch_size = 1024
#     predictions = model.predict(X_test[:inference_batch_size])

#     # Display the predictions
#     print("Predictions:", predictions)
#     end_time_inference = time.time()
#     elapsed_time_inference = end_time_inference - start_tinference
#     print("inference time is: ", elapsed_time_inference)



######################### functions


#Train and save model
def trainAndSaveModel(architecture,batch_size,nb_classes,nb_epoch,learning_rate,modelnameprefix):
    # The data, shuffled and split between train and test sets:
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  # Convert class vectors to binary class matrices.
  Y_train = to_categorical(y_train, nb_classes)
  Y_test = to_categorical(y_test, nb_classes)

  # Pre-processing
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255

  for run_id in range(nb_runs):
      model = Sequential()
      if architecture == 'Small':
          model.add(Convolution2D(8, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
          model.add(Convolution2D(8, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(16, (4, 4), padding='same', strides=2, activation='relu'))
          model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(32, (4, 4), padding='same', strides=2, activation='relu'))
          model.add(Flatten())
          model.add(Dense(552))
          model.add(Dense(256))
      else:
          # The CNN16384 Block from the paper
          model.add(Convolution2D(64, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
          model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(128, (4, 4), padding='same', strides=2, activation='relu'))
          model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
          model.add(Convolution2D(256, (4, 4), padding='same', strides=2, activation='relu'))
      if architecture == 'Shallow':
          model.add(Flatten())
          model.add(Dense(256))
      elif architecture == 'Full':
          model.add(Flatten())
          model.add(Dense(2048))
          model.add(Dense(256))
      elif architecture == 'TreeConnect':
          # Simulation of the TC((128, 128 - 16) - 256)
          model.add(Reshape((128, 128)))
          model.add(LocallyConnected1D(16, 1, activation='relu'))
          model.add(Permute((2, 1)))
          model.add(LocallyConnected1D(16, 1, activation='relu'))
          model.add(Flatten())
      elif architecture == 'Bottleneck':
          model.add(Flatten())
          model.add(Dense(18))
          model.add(Dense(167))
      elif architecture == 'RandomSparse':
          model.add(Flatten())
          model.add(RandomSparseLayer(2048, percentage_masked=99.22))
          model.add(RandomSparseLayer(256, percentage_masked=93.75))

      # ADD an FC at the and (MD)
      model.add(Flatten())
      #model.add(Dense(256)) # remove this ?
      #model.add(Dense(128)) # Remove this
      #model.add(Dense(64)) # remove this
      # Classification
      model.add(Dense(nb_classes))
      model.add(Activation('softmax'))

      model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['accuracy'])

      # Save stats
      tbCallback = callbacks.TensorBoard(log_dir='./cifar/' + architecture + '_run_' + str(run_id),
                                        histogram_freq=1, write_graph=True)
      model.summary()


      # Allow GPU memory usage to grow dynamically
      gpus = tf.config.experimental.list_physical_devices('GPU')
      if gpus:
          try:
              for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
          except RuntimeError as e:
              print(e)

      # Record the start time
      start_time = time.time()
      history = model.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=nb_epoch,
                validation_data=(X_test, Y_test),
                shuffle=True,
                callbacks=[tbCallback])
      K.clear_session()
      # Record the end time
      end_time = time.time()

      elapsed_time = end_time - start_time
      print(f"Training time for run {run_id + 1}: {elapsed_time:.2f} seconds")

      # Save model
      modelname = modelnameprefix + str(run_id) + ".keras"
      model.save(modelnameprefix)

    
      training_loss = history.history['loss']
      validation_loss = history.history['val_loss']
      training_accuracy = history.history['accuracy']  # Assuming you used 'accuracy' during model compilation
      validation_accuracy = history.history['val_accuracy']

      # Plotting
      plt.figure(figsize=(12, 8))

      # Plotting the loss curves
      plt.subplot(2, 1, 1)
      plt.plot(range(1, len(training_loss) + 1), training_loss, 'b', label='Training loss')
      plt.plot(range(1, len(validation_loss) + 1), validation_loss, 'r', label='Validation loss')
      plt.title('Training and Validation Loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()

      # Plotting the accuracy curves
      plt.subplot(2, 1, 2)
      plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, 'b', label='Training accuracy')
      plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, 'r', label='Validation accuracy')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.tight_layout()
      plt.show()

      # plt.plot(range(1, len(training_loss) + 1), training_loss,'b', label='Training loss')
      # plt.plot(range(1, len(validation_loss) + 1), validation_loss,'r', label='Validation loss')
      # plt.title('Training and Validation Loss')
      # plt.ylabel('Loss')
      # plt.xlabel('Epochs')
      # plt.legend(['Training loss', 'Validation loss'])

      # plt.plot(range(1, len(training_loss) + 1), training_loss, 'bo', label='Training Loss')
      # plt.plot(range(1, len(validation_loss) + 1), validation_loss, 'r', label='Validation Loss')
      # plt.title('Training and Validation Loss')
      # plt.xlabel('Epochs')
      # plt.ylabel('Loss')
      # plt.legend()
      # plt.show()



      start_tinference = time.time()
      # Load the saved model




####################################  Calculate Inference Speed     ##############################


def calculateInferenceSpeed(modelName):

    start_tinference = time.time()
    # Load the saved model
    model = load_model(modelName)

    # Display the model architecture
    model.summary()

    # Run inference on a batch of 1024
    inference_batch_size = 1024
    predictions = model.predict(X_test[:inference_batch_size])

    # Display the predictions
    print("Predictions:", predictions)
    end_time_inference = time.time()
    elapsed_time_inference = end_time_inference - start_tinference
    print("inference time is: ", elapsed_time_inference)


####################################  Calculate Inference Speed  with batch_size   ##############################
def calculateInferenceSpeedUsingBatches(modelName,batch_size):

    model = load_model(modelName)

    # Get the total number of samples
    num_samples = X_test.shape[0]
    print("number of samples", num_samples)
    # Initialize an array to store predictions
    all_predictions = []

    # Iterate over the dataset in batches
    start_tinference = time.time()
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        # Extract the current batch
        current_batch = X_test[start:end]

        # Make predictions on the current batch
        batch_predictions = model.predict(current_batch)

        # Append predictions to the result array
        all_predictions.append(batch_predictions)

    end_time_inference = time.time()
    elapsed_time_inference = end_time_inference - start_tinference
    print("inference time is: ", elapsed_time_inference)
    # Concatenate predictions from all batches
    predictions = np.concatenate(all_predictions, axis=0)
    return elapsed_time_inference


####################################  Simulate what is happening in the critical blocks   ##############################

def block_simulation_realweights(model_name):
  block1_layers = [
      Flatten(),
      Dense(2048, activation='relu'),
      Dense(256, activation='relu')
  ]

  block2_layers = [
      Reshape((128, 128)),
      LocallyConnected1D(16, 1, activation='relu'),
      Permute((2, 1)),
      LocallyConnected1D(16, 1, activation='relu'),
      Flatten()
  ]
  model = load_model(model_name)  # Replace with your model file path
  block1_weights = [layer.get_weights() for layer in block1_layers]
  block2_weights = [layer.get_weights() for layer in block2_layers]

  batch_size = 32  # Adjust as needed
  input_data = np.random.rand(batch_size, 16384)

  start_time_block1 = time.time()
  block1_output = simulate_single_block(input_data, block1_layers, block1_weights)
  end_time_block1 = time.time()

  start_time_block2 = time.time()
  block2_output = simulate_single_block(input_data, block2_layers, block2_weights)
  end_time_block2 = time.time()

  # Print execution times and compare
  print(f"Block 1 execution time: {end_time_block1 - start_time_block1:.6f} seconds")
  print(f"Block 2 execution time: {end_time_block2 - start_time_block2:.6f} seconds")



def simulate_single_block(input_data, layers, weights):
  """Simulates a block of layers."""
  output = input_data
  for layer, layer_weights in zip(layers, weights):
      output = layer(output, weights=layer_weights)
  return output



def block_simulation_dummy_data(batch_size,testingRange):
  
    
  # Define batch size (adjust as needed)
  

  # Create input data (shape might need adjustment)
  input_data = np.random.randn(batch_size, 16384)

  # Define both blocks:

  # Block 1: Flatten -> Dense(2048) -> Dense(256)
  block1 = Sequential([
      Flatten(),
      Dense(2048, activation='relu'),
      Dense(256, activation='relu')
  ])

  # Block 2: Reshape -> LocallyConnected1D -> Permute -> LocallyConnected1D -> Flatten
  block2 = Sequential([
      Reshape((128, 128)),
      LocallyConnected1D(16, 1, activation='relu'),
      Permute((2, 1)),
      LocallyConnected1D(16, 1, activation='relu'),
      Flatten()
  ])

  # Start measuring time for Block 1
  start_time_block1 = time.time()
  for _ in range(testingRange):  # Run multiple times for reliable measurement
      output_block1 = block1(input_data)
  end_time_block1 = time.time()

  # Calculate and print average inference time for Block 1
  average_time_block1 = (end_time_block1 - start_time_block1) / testingRange
  print(f"Average inference time for Block 1: {average_time_block1:.6f} seconds")

  # Start measuring time for Block 2
  start_time_block2 = time.time()
  for _ in range(testingRange):  # Run multiple times for reliable measurement
      output_block2 = block2(input_data)
  end_time_block2 = time.time()

  # Calculate and print average inference time for Block 2
  average_time_block2 = (end_time_block2 - start_time_block2) / testingRange
  print(f"Average inference time for Block 2: {average_time_block2:.6f} seconds")



################## Code starts here
# modelnameX = "modelv1.keras" #"fullconnect1.keras"
# #trainAndSaveModel("Full",64,10,2,0.001,modelnameX)
# #calculateInferenceSpeed("modelv1.h5")







########### Final testing
batch_size = 64 #64
nb_classes = 10
nb_epoch = 1 #50
nb_runs = 1 #5
learning_rate = 0.001 # Seggested to be the best in the paper
architecture = 'Full'  # can be one of 'Shallow', 'Full', 'TreeConnect', 'Bottleneck', 'Small' or 'RandomSparse'
modelnameprefix = "Model-16-01" + architecture







# #training time
# # architecture , Batch_size, number of classes,number of epochs, model name
trainAndSaveModel(architecture,64,nb_classes,50,0.001,'keras_fc_t2.keras')

#inference speed



# bs = 1024



# true_elapsed_time_inference = 0
# elapsed_time_inference = 0
# rangex = 10
# for i in range(rangex):
#  truetimeStart = time.time()
#  timeInf = 0
#  elapsed_time_inference += calculateInferenceSpeedUsingBatches('keras_fc_t2.keras',10000)
#  true_end_time_inference = time.time()
#  print("true inference time",(true_end_time_inference - truetimeStart))
#  true_elapsed_time_inference += (true_end_time_inference - truetimeStart)

# print("Average time ",elapsed_time_inference/rangex)
# print("Average time ",true_elapsed_time_inference/rangex)





#model = load_model('keras_fc_t1.keras')


#print('inference time is',calculateInferenceSpeedUsingBatches('keras_fc_t1.keras',1024)  )

# # Get the total number of samples
# num_samples = X_test.shape[0]
# print("number of samples", num_samples)
# # Initialize an array to store predictions
# all_predictions = []



# # Profile the forward pass
# with profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
#     # Iterate over the dataset in batches
#   start_tinference = time.time()
#   for start in range(0, num_samples, batch_size):
#       end = min(start + batch_size, num_samples)

#       # Extract the current batch
#       current_batch = X_test[start:end]

#       # Make predictions on the current batch
#       batch_predictions = model.predict(current_batch)

#       # Append predictions to the result array
#       all_predictions.append(batch_predictions)

#   end_time_inference = time.time()
#   elapsed_time_inference = end_time_inference - start_tinference
#   print("inference time is: ", elapsed_time_inference)
#   # Concatenate predictions from all batches
#   predictions = np.concatenate(all_predictions, axis=0)
  


# # Print the profiling results
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))



