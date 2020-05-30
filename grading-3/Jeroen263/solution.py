#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/Jeroen263'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "Albrechts, Jeroen"
student_id = "0949918"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = True


# In[ ]:


# Uncomment the following line to run in Google Colab
# get_ipython().system('pip install --quiet openml ')


# In[17]:


# Uncomment the following line to run in Google Colab
# get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU') # Check whether GPUs are available


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import openml as oml
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from tensorflow.keras.models import model_from_json


# In[19]:


from packaging import version
import sklearn
import tensorflow
sklearn_version = sklearn.__version__
tensorflow_version = tensorflow.__version__
if version.parse(sklearn_version) < version.parse("0.22.0"):
    print("scikit-learn is outdated. Please update now!")
if version.parse(tensorflow_version) < version.parse("2.1.0"):
    print("Tensorflow is outdated. This is version {}. Please update to 2.1.".format(tensorflow_version))
else:
    print("Hi{}, Looks good. You may continue :)".format(your_name.split(",")[1]))


# # Assignment 3

# ### Choice of libraries
# We recommend to use Tensorflow in this assignment since that is what we covered in the labs. If you feel confident using PyTorch (and Skorch for the scikit-learn wrapper), that is allowed too, as long as you are able to implement the requested functions and return the requested data. Read the assignment carefully and ensure that you can. Note that you may also need to do a bit more work to implement certain helper functions and wrappers.

# ### Storing and submitting files
# You must be able to store your models and submit them to GitHub Classroom. The evaluation functions used in this notebook will automatically store models for you.
# 
# If you want to run and solve the notebook on your local machine/laptop, fill in the path 'base_dir' to your assignment folder into the next cell.
# 
# If you use Colab, we recommend that you link it to your Google Drive:  
# * Upload the assignment folder to your Google Drive (+ New > Folder Upload)
# * Open Colab in a browser, open the 'Files' menu in the left sidebar, and click 'Mount Drive'
#   * At this point you may need to authenticate
# * Fill in the path to your assignment folder below
# #   * It's likely '/content/drive/My Drive/assignment-3-yourname'

# In[ ]:


# # base_dir = '/content/drive/My Drive/assignment-3-Jeroen263' # For Google Colab
#base_dir = './'


# In[21]:


#Uncomment to link Colab notebook to Google Drive
# from google.colab import drive
# drive.mount('/content/drive')


# ### Using GPUs
# While you can solve this assignment on a CPU, using a GPU will speed things up training quite a bit. If you have a local GPU, you can use that. If you don't, we recommend Google Colab. When you are in Colab:
# * In Runtime > Change runtime type, select the GPU under Hardware Accelerator
# * Run the 3rd cell on the top of this notebook to check that the GPU is found.
# 
# Note that Colab may not always have GPUs ready all the time, and may deny you a GPU when you have used them a lot. When you are temporarily 'locked out', you can switch to a non-GPU runtime or to a local instance of Jupyter running on your machine.

# ### Constraints
# * Your stored models should not be larger than 100MB when stored in file. GitHub will not allow uploading if they are.
# * When questions ask you to provide an explanation, it should be less than 500
# characters long. Some questions have a higher limit. Always answer in full sentences.
# * Don't train for more than 100 epochs, i.e. don't throw excessing computational resources at the problem. If your model hasn't converged by then, think of ways it could be made to converge faster. In this assignment you are not after the last tiny improvement, you can stop when learning curves flatten out. Do at least 5 epochs to get a reasonable learning curve.

# ### Grading
# Grading is based on the following aspects:
# * Correctness in answering the question. Carefully read the question and answer
# what is asked for. Train your models on the correct data. It should be clear on which data should be trained, but ask when in doubt. When something is not defined (e.g. the number of epochs or batch size), you can freely choose them.
# * Clarity of your explanations. Write short but precise descriptions of what you did and why. Give short but clear explanations of the observed performance. 
# After your explanation, your approach and model should make perfect sense. Refrain from using symbols as substitute for words in your explanation (e.g. no: "More layers -> more parameters" yes: "More layers mean more parameters"). 
# * Part of your grade depends on how well your model performs. When the question says 'you should at least get x%', x% will give you a good but not the maximal grade. You can get the full grade when you are close to what is the expected maximal performance. You don't need to invest lots of effort into the last tiny improvement, though. Unless specified, we look at the accuracy on the validation set. If your learning curves are very erratic we'll compute a score based on the smoothed curves (i.e. single peaks don't count).
# * The weight of each question is indicated. Take this into account when planning your time.

# ### Other tips
# * Don't wait until the last minute to do the assignment. The models take time to train, most questions will require some thinking, and some require you to read up on some new concepts.
# * Take care that you upload the results as requested. You need to submit not only the notebooks but also the trained models and learning curves (training histories). Be sure to run the verification script and check that all the results are included.
# * We provide an evaluation function that also stored models to disk. After you are done training the model, set the 'train' attribute to False so that the model doesn't train again (and loads from file instead) when you restart and rerun your notebook.
# * Explore. For many questions we'll ask you to explain your model design decisions. You cannot magically know the best solutions but you can experiment
# based on your understanding and make decisions based on both your knowledge and experiments. Your explanation is at least as important as the performance of your model.
# * Be original. We will check for plagiarism between student submissions.

# ### Data
# The [Street View House Numbers Dataset](https://www.openml.org/d/41081) contains 32-by-32 RGB images centered around a single digit of a house number appearing in Google Street View. Many of the images do contain some distractors at the sides. It consists of 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. Your goal is to build models that recognize the correct digit.

# If you use Colab, uncomment the following to cache the dataset inside the VM. This will make reloading faster if you need to restart your notebook. After longer periods of inactivity, your VM may be recycled and the cache lost, in which case the dataset will be downloaded again. Also note that this dataset is about 1Gb large, and will take even more space in memory. You may need to switch to a high-RAM environment (Colab will ask you if you hit the limit).

# In[ ]:


# Use OpenML caching in Colab
# On your local machine, it will store data in a hidden folder '~/.openml'
import os
# oml.config.cache_directory = os.path.expanduser('/content/cache')


# In[ ]:


# Download Streetview data. Takes a while (several minutes), and quite a bit of
# memory when it needs to download. After caching it loads faster.
SVHN = oml.datasets.get_dataset(41081)
X, y, _, _ = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)


# Reshape, sample and split the data

# In[ ]:


from tensorflow.keras.utils import to_categorical

Xr = X.reshape((len(X),32,32,3))
Xr = Xr / 255.
yr = to_categorical(y)


# In[ ]:


# DO NOT EDIT. DO NOT OVERWRITE THESE VARIABLES.
from sklearn.model_selection import train_test_split
# We do an 80-20 split for the training and test set, and then again a 80-20 split into training and validation data
X_train_all, X_test, y_train_all, y_test = train_test_split(Xr,yr, stratify=yr, train_size=0.8, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_all,y_train_all, stratify=y_train_all, train_size=0.8, random_state=1)
evaluation_split = X_train, X_val, y_train, y_val


# Check the formatting - and what the data looks like

# In[26]:


from random import randint

# Takes a list of row ids, and plots the corresponding images
# Use grayscale=True for plotting grayscale images
def plot_images(X, y, grayscale=False):
    fig, axes = plt.subplots(1, len(X),  figsize=(10, 5))
    for n in range(len(X)):
        if grayscale:
            axes[n].imshow(X[n], cmap='gray')
        else:
            axes[n].imshow(X[n])
        axes[n].set_xlabel((np.argmax(y[n])+1)%10) # Label is index+1
        axes[n].set_xticks(()), axes[n].set_yticks(())
    plt.show();

images = [randint(0,len(X_train)) for i in range(5)]
X_random = [X_train[i] for i in images]
y_random = [y_train[i] for i in images]
plot_images(X_random, y_random)


# ### Evaluation harness
# We provide an evaluation function 'run_evaluation' that you should use to 
# evaluate all your models. It also stores the trained models to disk so that
# your submission can be quickly verified, as well as to avoid having to train
# them over and over again. Your last run of the evaluation function (the last one
# stored to file), is the one that will be evaluated. The 'train' argument indicates whether to train or to load from disk. We have provided helper functions for saving and loading models to/from file, assuming you use TensorFlow. If you use PyTorch you'll have to adapt them.

# In[ ]:


import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # for use with tensorflow

def shout(text, verbose=1):
    """ Prints text in red. Just for fun.
    """
    if verbose>0:
        print('\033[91m'+text+'\x1b[0m')

#def load_model_from_file(base_dir, name, extension='.h5'):
#    """ Loads a model from a file. The returned model must have a 'fit' and 'summary'
#    function following the Keras API. Don't change if you use TensorFlow. Otherwise,
#    adapt as needed. 
#    Keyword arguments:
#    base_dir -- Directory where the models are stored
#    name -- Name of the model, e.g. 'question_1_1'
#    extension -- the file extension
#    """
#    try:
#        model = load_model(os.path.join(base_dir, name+extension))
#    except OSError:
#        shout("Saved model could not be found. Was it trained and stored correctly? Is the base_dir correct?")
#        return False
#    return model
#
#def save_model_to_file(model, base_dir, name, extension='.h5'):
#    """ Saves a model to file. Don't change if you use TensorFlow. Otherwise,
#    adapt as needed. 
#    Keyword arguments:
#    model -- the model to be saved
#    base_dir -- Directory where the models should be stored
#    name -- Name of the model, e.g. 'question_1_1'
#    extension -- the file extension
#    """
#    model.save(os.path.join(base_dir, name+extension))

def load_model_from_file(base_dir, name, extension='.h5'):
  """ Loads a model from a file. The returned model must have a 'fit' and 'summary'
  function following the Keras API. Don't change if you use TensorFlow. Otherwise,
  adapt as needed. 
  Keyword arguments:
    base_dir -- Directory where the models are stored
    name -- Name of the model, e.g. 'question_1_1'
    extension -- the file extension
  """
  try:
    # if a json description is available, load config and then weights
    if os.path.isfile(os.path.join(base_dir, name+'.json')):
      json_file = open(os.path.join(base_dir, name+'.json'), 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      model = model_from_json(loaded_model_json)
      model.load_weights(os.path.join(base_dir, name+extension))
    # else just load the entire model from hdf5 file
    else:
      model = load_model(os.path.join(base_dir, name+extension))
  except OSError:
    shout("Saved model could not be found. Was it trained and stored correctly? Is the base_dir correct?")
    return False
  return model

def save_model_to_file(model, base_dir, name, extension='.h5'):
  """ Saves a model to file. Don't change if you use TensorFlow. Otherwise,
  adapt as needed. 
  Keyword arguments:
    model -- the model to be saved
    base_dir -- Directory where the models should be stored
    name -- Name of the model, e.g. 'question_1_1'
    extension -- the file extension
  """
  path = os.path.join(base_dir, name+extension)
  model.save(path)
  size = os.path.getsize(path)
  # If model > 100MB, store the weights and architecture only.
  if size > 100*1024*1024:
    print("Model larger than 100MB, storing weights only.")
    model.save_weights(path)
    model_json = model.to_json()
    with open(os.path.join(base_dir, name+".json"), "w") as json_file:
        json_file.write(model_json)


# Helper function to extract min/max from the learning curves
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

# DO NOT EDIT
def run_evaluation(name, model_builder, data, base_dir, train=True, 
                   generator=False, epochs=3, batch_size=32, steps_per_epoch=60, 
                   verbose=1, **kwargs):
    """ Trains and evaluates the given model on the predefined train and test splits,
    stores the trained model and learning curves. Also prints out a summary of the 
    model and plots the learning curves.
    Keyword arguments:
    name -- the name of the model to be stored, e.g. 'question_1_1.h5'
    model_builder -- function that returns an (untrained) model. The model must 
                     have a 'fit' function that follows the Keras API. It can wrap
                     a non-Keras model as long as the 'fit' function takes the 
                     same attributes and returns the learning curves (history).
                     It also must have a 'summary' function that prints out a 
                     model summary, and a 'save' function that saves the model 
                     to disk. 
    data -- data split for evaluation. A tuple of either:
            * Numpy arrays (X_train, X_val, y_train, y_val)
            * A data generator and validation data (generator, X_val, y_val)
    base_dir -- the directory to save or read models to/from
    train -- whether or not the data should be trained. If False, the trained model
             will be loaded from disk.
    generator -- whether the data in given as a generator or not
    epochs -- the number of epochs to train for
    batch_size -- the batch size to train with
    steps_per_epoch -- steps per epoch, in case a generator is used (ignored otherwise)
    verbose -- verbosity level, 0: silent, 1: minimal,...
    kwargs -- keyword arguments that should be passed to model_builder.
              Not required, but may help you to adjust its behavior
    """
    return
    model = model_builder(**kwargs)
    if not model:
        shout("No model is returned by the model_builder")
        return
    if not hasattr(model, 'fit'):
        shout("Model is not built correctly")
        return
    learning_curves = {}
    if train and not stop_training: # Train anew
        shout("Training the model", verbose)
        if generator:
            generator, X_val, y_val = data
            history = model.fit(generator, epochs=epochs, batch_size=batch_size,
                              steps_per_epoch=steps_per_epoch, verbose=1, 
                              validation_data=(X_val, y_val))
            learning_curves = history.history
        else:
            X_train, X_val, y_train, y_val = data
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                              verbose=1, validation_data=(X_val, y_val))
            learning_curves = history.history
        shout("Saving to file", verbose)
        save_model_to_file(model, base_dir, name)
        with open(os.path.join(base_dir, name+'.p'), 'wb') as file_pi:
            pickle.dump(learning_curves, file_pi)
        shout("Model stored in "+base_dir, verbose)
    else: # Load from file
        shout("Loading model from file", verbose)
        model = load_model_from_file(base_dir, name)
        if not model:
            shout("Model not found")
            return
        learning_curves = None
        try:
            learning_curves = pickle.load(open(os.path.join(base_dir, name+'.p'), "rb"))
        except FileNotFoundError:
            shout("Learning curves not found")
            return
        shout("Success!", verbose)
    # Report
    print(model.summary())
    lc = pd.DataFrame(learning_curves)
    lc.plot(lw=2,style=['b:','r:','b-','r-']);
    plt.xlabel('epochs');
    print(lc.apply(minMax))


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[28]:


from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

def build_model_1_1():
  model = models.Sequential()
  model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dense(200, activation='relu'))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  
  model.compile(optimizer=optimizers.Adagrad(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=False, epochs=30, batch_size=32)
answer_q_1_1 = """
Exploring different numbers of filters and different amounts of layers
showed the model performs best using 3 dense hidden layers and an output layer. 
The first hidden layer has 400 filters, the second layer 200 and the
third and layer has 100 filters. Relu activation for the hidden layers
paired with softmax activation for the output layer resulted in the best accuracy.
An accuracy of 82.4% was achieved after 30 epochs with a batch
size of 32, which is reasonable although it does overfit.
"""
print("Answer is {} characters long".format(len(answer_q_1_1)))


# ### Question 1.2: Preprocessing (2 points)
# Rerun the model, but now preprocess the data first by converting the images to 
# greyscale. You can use the helper function below. If you want to do additional 
# preprocessing, you can do that here, too.
# * Store the preprocessed data as a tuple `preprocessed_split`
# * Rerun and re-evaluate your model using the preprocessed data.
#   * For the remainder of the assignment, always use the preprocessed data
# * Explain what you did and interpret the results in 'answer_q_1_2'. Is the model
#   better, if so, why?

# In[ ]:


# Luminance-preserving RGB to greyscale conversion
def rgb2gray(X):
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)


# In[30]:


# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val

# Adjusted model
def build_model_1_2():
  model = models.Sequential()
  model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dense(200, activation='relu'))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  
  model.compile(optimizer=optimizers.Adagrad(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=False, epochs=30, batch_size=32)
answer_q_1_2 = """
Conversion to grayscale was done as preprocessing. The only modification of the
model is the change of input shape. The model achieved the same accuracy
of 82.4%. This is a good result considering less data is used.
It can be noted that the learning curves are smoother using the 
preprocessed data, especially when looking at the curve describing the validation
loss. This shows that the preprocessing makes the input less noisy.
"""
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[31]:


from keras import regularizers

def build_model_1_3():
  model = models.Sequential()
  model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(400, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(200, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(100, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10, activation='softmax'))
  
  model.compile(optimizer=optimizers.Adagrad(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  return model

run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=False, epochs=30, batch_size=32)
answer_q_1_3 = """
Multiple regularization options were evaluated. Using dropout did not improve 
the model performance. Batch normalization and kernel regularization combined
allowed the model to achieve a validation accuracy of 85.4%, which is around 3%
higher than before. The model also overfits much less. The best values for the 
learning rate and batch size were already determined experimentally in previous
questions (batch size of 32 and learning rate of 0.01 using Adagrad optimizer).
"""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[32]:


from tensorflow.keras import initializers

def build_model_2_1():
  kernelinit = initializers.VarianceScaling(distribution='uniform', mode='fan_avg')
  kernelreg = regularizers.l2(0.001)

  convnet = models.Sequential()
  convnet.add(layers.BatchNormalization(input_shape=(32, 32, 1)))
  convnet.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernelinit,
                            kernel_regularizer=kernelreg))
  convnet.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernelinit,
                            kernel_regularizer=kernelreg))
  convnet.add(layers.MaxPooling2D((2, 2)))
  convnet.add(layers.BatchNormalization())
  convnet.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernelinit,
                            kernel_regularizer=kernelreg))
  convnet.add(layers.MaxPooling2D((2, 2)))
  convnet.add(layers.BatchNormalization())
  convnet.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernelinit,
                            kernel_regularizer=kernelreg))
  convnet.add(layers.MaxPooling2D((2, 2)))
  convnet.add(layers.BatchNormalization())
  convnet.add(layers.Flatten())
  convnet.add(layers.Dropout(0.5))
  convnet.add(layers.Dense(4 * 1024, activation='relu', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  convnet.add(layers.Dropout(0.5))
  convnet.add(layers.Dense(2 * 1024, activation='relu', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  convnet.add(layers.Dropout(0.5))
  convnet.add(layers.Dense(1 * 1024, activation='relu', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  convnet.add(layers.Dense(10, activation='softmax', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  
  convnet.compile(optimizer=optimizers.Adagrad(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return convnet

run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=False, epochs=30, batch_size=64)
answer_q_2_1 = """
The model is loosely inspired by the VGG model that generally performs
well on image data. The model consists of 3 blocks of convolutional layers,
each ending with maxpooling and batch normalization. The first block contains 2
convolutional layers with 64 filters. Another convolutional block is then added
containing 1 layer with 128 filters. The last convolutional block uses 1 layer
with 256 layers. A flatten layer is then added followed by a pyramid
structure of 3 dense layers. These layers consist of 4096, 2048 and 1024
filters respectively. All layers use an l2 kernelregularizer combined with a
kernel initializer using variance scaling based on a uniform distribution. This
model achieves a very high accuracy of 94.3% after 30 epochs. The model starts 
to overfit after around epoch 15.
"""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[33]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    #rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2)
    #shear_range=0.2)
    #zoom_range=0.2)
    #horizontal_flip=True)

X_train_pre = rgb2gray(X_train)
X_val_pre = rgb2gray(X_val)

augmented_train = train_datagen.flow(
        X_train_pre, y_train,
        batch_size=64)

augmented_split = (augmented_train, X_val_pre, y_val)


run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, 
               train=False, epochs=30, batch_size=64, generator=True, steps_per_epoch=len(X_train_pre)//64)
answer_q_2_2 = """
The model was evaluated using augmented data containing rotations, width and
height shifts, shear, zoom and horizontal flips. After experimenting with many
combinations of these augmentations, the best result was obtained using only
width and height shift. The accuracy on the validation set was 94.8%
(increase of 0.5%). The training accuracy was only 93.5% compared
to the previous 98.2%, meaning that there is no more overfitting and the model
might be able to improve further with more epochs.
"""
print("Answer is {} characters long".format(len(answer_q_2_2)))


# ## Part 3. Model interpretation (10 points)
# ### Question 3.1: Interpreting misclassifications (2 points)
# Study which errors are still made by your last model (model_2_2) by evaluating it on the test data. You do not need to retrain the model.
# * What is the accuracy of model_2_2 on the test data? Store this in 'test_accuracy_3_1'.
# * Plot the confusion matrix in 'plot_confusion_matrix' and discuss which classes are often confused.
# * Visualize the misclassifications in more depth by focusing on a single
# class (e.g. the number '2') and analyse which kinds of mistakes are made for that class. For instance, are the errors related to the background, noisiness, etc.? Implement the visualization in 'plot_misclassifications'.
# * Summarize your findings in 'answer_q_3_1'

# In[34]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def predict_labels(name, verbose=1):
  #shout("Loading model from file", verbose)
  model = load_model_from_file(base_dir, name)
  if not model:
      shout("Model not found")
      return
  predictions = model.predict(rgb2gray(X_test))
  return predictions

predictions = predict_labels("model_2_2")
y_test_cat = (y_test.argmax(axis=1)+1)%10
y_pred_cat = (predictions.argmax(axis=1)+1)%10
#(np.argmax(y[n])+1)%10

acc = accuracy_score(y_test_cat, y_pred_cat)
print("Accuracy:", round(acc,4))

test_accuracy_3_1 = acc


def plot_confusion_matrix():
  preds = predict_labels("model_2_2")
  y_test_cat = (y_test.argmax(axis=1)+1)%10
  y_pred_cat = (preds.argmax(axis=1)+1)%10

  labels = np.arange(0, 10, 1)
  cm = confusion_matrix(y_test_cat, y_pred_cat)
  fig, ax = plt.subplots()
  im = ax.imshow(cm)
  ax.set_xticks(labels), ax.set_yticks(labels)
  ax.set_xticklabels(labels, rotation=45, ha="right")
  ax.set_yticklabels(labels)
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  for i in range(100):
      ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")
  return

plot_confusion_matrix()

def plot_misclassifications():
  number = 5
  preds = predict_labels("model_2_2")
  y_test_cat = (y_test.argmax(axis=1)+1)%10
  y_pred_cat = (preds.argmax(axis=1)+1)%10

  X_incorrect = []
  y_incorrect = []
  for i in range(len(y_test_cat)):
    if (y_test_cat[i] == number) and not (y_test_cat[i] == y_pred_cat[i]):
      X_incorrect.append(X_test[i])
      y_incorrect.append(preds[i])
  for i in range(4):
    images = [randint(0,len(X_incorrect)-1) for i in range(5)]
    X_random = [X_incorrect[i] for i in images]
    y_random = [y_incorrect[i] for i in images]
    plot_images(X_random, y_random)
  print("Misclassified images with true label", number)
  return

plot_misclassifications()


answer_q_3_1 = """
The accuracy of the model on the test data is 94.5%, which is quite good.
The most misclassifications occur in images with true label 1. This is likely caused
by the fact that this class contains the most observations. Furthermore, images with label 5 are often
misclassified as 3 or 6. When looking at the misclassifications for the image 5, we
observe that most mistakes are made in either very blurry images or images containing
multiple digits. The number 5 might be classified often as a 3 because the bottom
part of the number has a similar curvature to that of the number 3. The upper part
of the number 5 looks similar to that of a 6 and blurry images sometimes make it
look like the bottom curve of the 5 is closed, resembling a 6.
"""
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[41]:


from tensorflow.keras import models

def plot_activations():
  name = "model_2_2"
  #indexes of convolutional layers:
  convlayers = [1,2,5,8]

  model = load_model_from_file(base_dir, name)
  if not model:
      shout("Model not found")
      return

  testdata = rgb2gray(X_test)
  testimage = np.expand_dims(testdata[0], axis=0)

  layers_out = [layer.output for layer in model.layers[:15]]
  activation_model = models.Model(inputs=model.input, outputs=layers_out)
  activations = activation_model.predict(testimage)

  images_per_row = 16

  layer_names = []
  for layer in model.layers[:10]:
      layer_names.append(layer.name)

  namelist=[]
  for i in convlayers:
    namelist.append(layer_names[i])

  activationlist=[]
  for i in convlayers:
    activationlist.append(activations[i])

  start = 0
  # Now let's display our feature maps
  for layer_name, layer_activation in zip(namelist, activationlist):

      # This is the number of features in the feature map
      n_features = layer_activation.shape[-1]

      # The feature map has shape (1, size, size, n_features)
      size = layer_activation.shape[1]

      # We will tile the activation channels in this matrix
      n_cols = n_features // images_per_row
      display_grid = np.zeros((size * n_cols, images_per_row * size))

      # We'll tile each filter into this big horizontal grid
      for col in range(n_cols):
          for row in range(images_per_row):
              channel_image = layer_activation[0,
                                              :, :,
                                              col * images_per_row + row]
              # Post-process the feature to make it visually palatable
              channel_image -= channel_image.mean()
              channel_image /= channel_image.std()
              channel_image *= 64
              channel_image += 128
              channel_image = np.clip(channel_image, 0, 255).astype('uint8')
              display_grid[col * size : (col + 1) * size,
                          row * size : (row + 1) * size] = channel_image

      # Display the grid
      scale = 1. / size
      plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
      plt.title("Activation of layer {} ({})".format(convlayers[start]+1,layer_name))
      plt.grid(False)
      plt.imshow(display_grid, aspect='auto', cmap='viridis')
      start += 1

  plt.show()
  return

# plot_activations()

answer_q_3_2 = """
The first layer consists mainly of filters that highlight the edges of the digit.
The second layer seems to use filters that have a similar function, but only
highlight certain edges of the digit like horizontal or vertical or only parts
of the edges. The third and fourth convolutional layers produce very abstract 
activations, making them difficult to analyze. The code from the notebook
shown in class was used to visualize the activations.
"""
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[43]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()
import cv2

def plot_activation_map():
  name = "model_2_2"
  index = 0
  model = load_model_from_file(base_dir, name)
  if not model:
      shout("Model not found")
      return

  testdata = rgb2gray(X_test)
  x = np.expand_dims(testdata[index], axis=0)
  preds = model.predict(x)
  pred_label = (np.argmax(preds) + 1) % 10
  actual_label = (np.argmax(y_test[index]) + 1) % 10
  print("Model predicts label as", str(pred_label), "while true label is", str(actual_label))

  # code is obtained from provided notebook
  model_output = model.output[:, np.argmax(preds)]

  # The is the output feature map of the `block5_conv3` layer,
  # the last convolutional layer in VGG16
  last_conv_layer = model.get_layer('conv2d_11')

  # This is the gradient of the "african elephant" class with regard to
  # the output feature map of `block5_conv3`
  grads = K.gradients(model_output, last_conv_layer.output)[0]

  # This is a vector of shape (512,), where each entry
  # is the mean intensity of the gradient over a specific feature map channel
  pooled_grads = K.mean(grads, axis=(0, 1, 2))

  # This function allows us to access the values of the quantities we just defined:
  # `pooled_grads` and the output feature map of `block5_conv3`,
  # given a sample image
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  # These are the values of these two quantities, as Numpy arrays,
  # given our sample image of two elephants
  pooled_grads_value, conv_layer_output_value = iterate([x])

  # We multiply each channel in the feature map array
  # by "how important this channel is" with regard to the elephant class
  for i in range(256):
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  # The channel-wise mean of the resulting feature map
  # is our heatmap of class activation

  heatmap = np.mean(conv_layer_output_value, axis=-1)

  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)

  # We use cv2 to load the original image
  img = testdata[index]

  # We resize the heatmap to have the same size as the original image
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

  # We convert the heatmap to RGB
  heatmap = np.uint8(255 * heatmap)

  # We apply the heatmap to the original image
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  # 0.4 here is a heatmap intensity factor
  superimposed_img = heatmap * 0.0025 + img
  return img, heatmap, superimposed_img 

def plot_3_3():
  img, heatmap, superimposed_img = plot_activation_map()

  fig, axes = plt.subplots(1, 3,  figsize=(15, 5))

  axes[0].imshow(img.reshape(32,32), cmap='gray')
  axes[0].set_xlabel('Original image (grayscale)')
  axes[0].set_xticks(())
  axes[0].set_yticks(())

  axes[1].matshow(heatmap)
  axes[1].set_xlabel('Class activation map')
  axes[1].set_xticks(())
  axes[1].set_yticks(())

  axes[2].imshow(superimposed_img)
  axes[2].set_xlabel('Class activation \n superimposed on image')
  axes[2].set_xticks(())
  axes[2].set_yticks(())
  plt.show()
  return

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[44]:


from tensorflow.keras.applications.vgg16 import VGG16

def build_model_4_1():
  block5_trainable=True

  kernelinit = initializers.VarianceScaling(distribution='uniform', mode='fan_avg')
  kernelreg = regularizers.l2(0.001)

  conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(32, 32, 3))

  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  model.add(layers.Dense(10, activation='softmax', kernel_initializer=kernelinit,
                           kernel_regularizer=kernelreg))
  
  conv_base.trainable = True

  set_trainable = False
  for layer in conv_base.layers:
      if layer.name == 'block5_conv1' and block5_trainable:
          set_trainable = True
      if set_trainable:
          layer.trainable = True
      else:
          layer.trainable = False
  
  model.compile(optimizer=optimizers.Adagrad(lr=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model


run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
               train=False, epochs=30, batch_size=64)
answer_q_4_1 = """
With the model completely frozen it achieves a poor accuracy of 56.8%. Unfeezing
the last convolutional block improves this to 84.6%, which is a significant
improvement but still not great. There occurs heavy overfitting, indicating that
the model might be too complex. Unfreezing all layers results in a high accuracy
(95%), but this is not the point of the assignment.
"""
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[45]:


import pickle
import gzip
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def store_embedding(X, name):  
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
    pickle.dump(X, file_pi)

def load_embedding(name):
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
    return pickle.load(file_pi)

def store_embeddings():
  name = "model_4_1"
  model = load_model_from_file(base_dir, name)
  if not model:
      shout("Model not found")
      return
  
  conv_part = model.layers[0]

  embed_model = models.Sequential()
  embed_model.add(conv_part)
  embed_model.add(layers.Flatten())
  
  conv_part.trainable = False
  embed_model.summary()

  train_output = embed_model.predict(X_train)
  test_output = embed_model.predict(X_test)
  val_output = embed_model.predict(X_val)
  store_embedding(train_output, 'train')
  store_embedding(test_output, 'test')
  store_embedding(val_output, 'val')
  return

#store_embeddings()

def generate_pipeline():
  #scaler = StandardScaler()
  #clf = RandomForestClassifier(random_state=1, n_jobs=-1)
  #clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
  clf = SVC(random_state=1)
  pca = PCA(n_components=20)

  pipeline = Pipeline(steps=[('dim_reduction', pca),
                             ('fit', clf)])
  return pipeline

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
  """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
  """
  model = pipeline.fit(X_train, y_train)
  preds = model.predict(X_test)
  acc = accuracy_score(y_test, preds)
  return acc

def evaluation_4_2(X_train, y_train, X_test, y_test):
  """ Runs 'evaluate_pipeline' with embedded versions of the input data 
  and returns the accuracy.
  """
  y_train_cat = []
  for i in y_train:
    cat = (np.argmax(i)+1)%10
    y_train_cat.append(cat)

  y_test_cat = []
  for i in y_test:
    cat = (np.argmax(i)+1)%10
    y_test_cat.append(cat)

  train_embed = load_embedding('train')
  test_embed = load_embedding('test')
  pipeline = generate_pipeline()
  acc = evaluate_pipeline(pipeline, train_embed, y_train_cat, test_embed, y_test_cat)
  return acc

print(evaluation_4_2(X_train, y_train, X_test, y_test))

answer_q_4_2 = """
The models RF, kNN and SVC were compared along with different scaling techniques to
see what model performed best on the validation data when trained on the training
data. SVC combined with PCA with 20 components performed best. It achieves an 
accuracy of 84.9% on the test data, which is slightly higher than that of only the model from question
4.1 but still lower than the model from question 2.2.
"""
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


# In[ ]:





dg_code= """
train_datagen = ImageDataGenerator(
    #rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2)"""
last_edit = 'May 26, 2020'