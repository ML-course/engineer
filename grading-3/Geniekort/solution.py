#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/Geniekort'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fill in your name using the format below and student ID number
your_name = "Kortleven, David"
student_id = "0937121"


# In[23]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = True


# In[3]:


# Uncomment the following line to run in Google Colab
# !pip install --quiet openml 


# In[4]:


# Uncomment the following line to run in Google Colab
#%tensorflow_version 2.x
import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU') # Check whether GPUs are available


# In[5]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import openml as oml
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[6]:


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

# In[7]:


# #base_dir = '/content/drive/My Drive/TestAssignment' # For Google Colab
# base_dir = './'


# In[8]:


#Uncomment to link Colab notebook to Google Drive
# #from google.colab import drive
# #drive.mount('/content/drive')


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

# In[9]:


# Use OpenML caching in Colab
# On your local machine, it will store data in a hidden folder '~/.openml'
#import os
# #oml.config.cache_directory = os.path.expanduser('/content/cache')


# In[10]:


# Download Streetview data. Takes a while (several minutes), and quite a bit of
# memory when it needs to download. After caching it loads faster.
SVHN = oml.datasets.get_dataset(41081)
X, y, _, _ = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)


# Reshape, sample and split the data

# In[11]:


from tensorflow.keras.utils import to_categorical

Xr = X.reshape((len(X),32,32,3))
Xr = Xr / 255.
yr = to_categorical(y)


# In[12]:


# DO NOT EDIT. DO NOT OVERWRITE THESE VARIABLES.
from sklearn.model_selection import train_test_split
# We do an 80-20 split for the training and test set, and then again a 80-20 split into training and validation data
X_train_all, X_test, y_train_all, y_test = train_test_split(Xr,yr, stratify=yr, train_size=0.8, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_all,y_train_all, stratify=y_train_all, train_size=0.8, random_state=1)
evaluation_split = X_train, X_val, y_train, y_val


# Check the formatting - and what the data looks like

# In[13]:


from random import randint

# Takes a list of row ids, and plots the corresponding images
# Use grayscale=True for plotting grayscale images
def plot_images(X, y, grayscale=False):
    fig, axes = plt.subplots(1, len(X),  figsize=(10, 5))
    for n in range(len(X)):
        if grayscale:
            axes[n].imshow(X[n].reshape(32, 32)/255, cmap='gray')
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

# In[14]:


import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, model_from_json # for use with tensorflow

def shout(text, verbose=1):
    """ Prints text in red. Just for fun.
    """
    if verbose>0:
        print('\033[91m'+text+'\x1b[0m')

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
                              steps_per_epoch=steps_per_epoch, verbose=verbose, 
                              validation_data=(X_val, y_val))
            learning_curves = history.history
        else:
            X_train, X_val, y_train, y_val = data
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                              verbose=verbose, validation_data=(X_val, y_val))
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


# In[15]:


# Toy usage example
# Remove before submission
from tensorflow.keras import models
from tensorflow.keras import layers 

def build_toy_model():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# First build and store
run_evaluation("toy_example", build_toy_model, evaluation_split, base_dir, 
               train=True, epochs=3, batch_size=32)


# In[16]:


# Toy usage example
# Remove before submission
# With train=False: load from file and report the same results without rerunning
run_evaluation("toy_example", build_toy_model, evaluation_split, base_dir, 
               train=False)


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[17]:


def build_model_1_1():
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.05) # 80% acc
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # 83% acc

    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(125, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=True, epochs=25, batch_size=32)
answer_q_1_1 = "I first tried RMSprop as optimizer, but gave accuracy < 65. Then tried SGD, which gave better results (+/-80%). Then tried Adam, with multiple learning rates, but 1e-3 gave best results. First I tried doing large layers (starting at 3000), these gave good results. However smaller layers (starting at 1024) gave just as good results. Probably since 3000 was just way more than necessary to model the problem. Wrt epochs, from epoch 25, there was overfitting, so I added early stopping at epoch 25."
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

# In[18]:


# Luminance-preserving RGB to greyscale conversion
def rgb2gray(X):
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)


# In[19]:


# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val


# In[20]:


images = [randint(0,len(preprocessed_split[0])) for i in range(5)]
X_random = [preprocessed_split[0][i] for i in images]
y_random = [preprocessed_split[2][i] for i in images]
# plot_images(X_random, y_random, grayscale=True)


# In[21]:


# Adjusted model
def build_model_1_2():
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.05) # 80% acc
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # 83% acc

    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dense(125, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=True, epochs=25, batch_size=32)
answer_q_1_2 = """
               I transformed every image to grayscale. I noticed that the validation accuracy went slightly up, which can be explained by the fact that the data is simplified, and the information lies mainly in the contrast between numbers and their background, not the actual color of the number. The overfitting started at around the same epoch, so I kept the same amount of epochs (25).
               """
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[22]:


from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV


def build_model_1_3():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # 83% acc

    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dropout(0.01))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.01))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.01))
    model.add(layers.Dense(250, activation='relu'))
    model.add(layers.Dropout(0.01))
    model.add(layers.Dense(125, activation='relu'))
    model.add(layers.Dropout(0.01))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])    
    return model


run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=True, epochs=40, batch_size=256)
answer_q_1_3 = "I noticed that increasing batch size speeds up the training a lot. Adding dropout layers reduces overfitting, but when adding too much dropout, performance drops. A dropout rate of 0.5 removes most performance of the model. But 0.01 really prevents overfitting, but slows down training accuracy improvement, so more epochs (around 40) are needed to achieve similar accuracy."
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[34]:


def build_model_2_1(dropout=0.2):
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  # model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
  model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
  # model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(dropout))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model
run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=True, epochs=10, batch_size=64, dropout=0.2, verbose=1)
answer_q_2_1 = """Tried to have 4 block of 2 layers of convolutional layers, this caused overfitting. So I added max pooling to the first and third block to decrease overfitting, and added one dropout layer (tuned between 0.01, 0.1, 0.2, 0.5) after the dense layer in the end, to reduce overfitting. Also I added early stopping at epoch 10, since overfitting started from that point. I also tuned the batch size, tested 32,64,128,256. 64 gave the best results, even though the overfitting was less for higher batch sizes."""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[35]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 64

# for shift_range in [True, False]:
    # print("Trying " + str(shift_range))
data_augmenter = ImageDataGenerator(
    width_shift_range=0.1,
    zoom_range=(0.9, 0.9)
)

augmented_train = data_augmenter.flow(preprocessed_split[0], preprocessed_split[2], batch_size=batch_size)
augmented_split = augmented_train, preprocessed_split[1], preprocessed_split[3]

images, labels = augmented_train[0]
plot_images(images[:5], labels[:5], grayscale=True)


run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, 
                   train=True, epochs=20, batch_size=None, dropout=0.2, verbose=1, generator=True,
                   steps_per_epoch=len(preprocessed_split[0]) / batch_size,)

answer_q_2_2 = """
              I tried to width_shift_range between 0, 0.1,0.2 and 0.5. 0.1 gave best results. I also added zooming, tuned between 0.9, 0.7, 0.5. 0.9 Gave the best results. It is understandable why this improves the accuracy. By zooming in, we reduce irrelevant cues from the image. Zooming in too much however, would remove crucial information. Flipping did not improve the accuracy, probably since the it does not provide any new usable info.
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

# In[36]:


from sklearn.metrics import accuracy_score, confusion_matrix

model = load_model_from_file(base_dir, "model_2_2")
X_test_augmented = rgb2gray(X_test)
y_pred = model.predict(X_test_augmented)

# print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
test_accuracy_3_1 = 0.9327 

def plot_confusion_matrix():
  cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
  fig, ax = plt.subplots()
  im = ax.imshow(cm)
  ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
  ax.set_xticklabels(list(np.array(range(1,11)) % 10))
  ax.set_yticklabels(list(np.array(range(1,11)) % 10))
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  for i in range(100):
    ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")
  

def plot_misclassifications(show_count=5, show_true_class=5, rows=1):
  misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]

  selected_misclassified_samples = []
  for misclassified_index in misclassified_samples:
    if(np.argmax(y_test[misclassified_index]) == show_true_class - 1):
      selected_misclassified_samples.append(misclassified_index)

  for r in range(rows):
    fig, axes = plt.subplots(1, show_count,  figsize=(40, 5))
    for nr, i in enumerate(selected_misclassified_samples[show_count*r:show_count*(r+1)]):
        axes[nr].imshow(X_test[i])
        axes[nr].set_xlabel("Predicted: %s,\n Actual : %s, \n index: %s" % (np.argmax(y_pred[i]) + 1,np.argmax(y_test[i]) + 1, i))
        axes[nr].set_xticks(()), axes[nr].set_yticks(())

plot_confusion_matrix()
plot_misclassifications(15, 5, 2)
answer_q_3_1 = """
               From the confusion matrix it becomes clear that a 7 is predicted to be a 1 or 2 relatively often. (Not the other way around). Also a 5 is predicted to be a 3 often. When looking at some of the misclassified fives, we see different types of difficulties in some of the image. In some images, the sample consists of so much noise that even for a human it is hardly possible to distinghuish the label. Another cause for errors are distracting numbers next to the number to be classified. Some wrongly classified samples also are rotated significantly (> 45 deg.). Finally I also noticed a sample in which the image shows a 4, but is (wrongly) labelled as a 5. In this case, the provided training data is wrong.
               """
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[37]:


def plot_activations():
  model = load_model_from_file(base_dir, "model_2_2")
  sample = np.expand_dims(X_test_augmented[0], axis=0)
  layers_outputs = [l.output for l in model.layers[:10]]
  layers_names = [l.name for l in model.layers[:22]]
  activation_model = models.Model(inputs=model.input, outputs=layers_outputs)

  prediction_activations = activation_model.predict(sample)
  first_layer_activation = prediction_activations[4]

  for name, activations in zip(layers_names, prediction_activations):
    plot_layer_activations(name, activations)

# Plot the activations of all filters in a layer.
def plot_layer_activations(layer_name, layer_activation):
  filters_per_row = 16
  filter_count = layer_activation.shape[-1]
  filter_size = layer_activation.shape[1]
  rows_count = filter_count // filters_per_row
  
  filter_display = np.zeros((filter_size * rows_count, filter_size * filters_per_row))
  for row in range(rows_count):
    for col in range(filters_per_row):
      channel_image = layer_activation[0, :, :, row * filters_per_row + col]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
      filter_display[row * filter_size : (row + 1) * filter_size, col * filter_size : (col + 1) * filter_size] = channel_image
  plt.figure(figsize=(1/filter_size * filter_display.shape[1],
                            1/filter_size * filter_display.shape[0]))
  plt.title(f"Activation of layer {layer_name} ")
  plt.grid(False)
  plt.imshow(filter_display, aspect='auto', cmap='viridis')

# plot_activations()
plt.show()

answer_q_3_2 = """The visualizations of the layer activations show interesting behavior. The initial layer triggers mostly on recognizable features of the 8, such as the circular shapes. A few layers deeper, the filters are also representing some higher level shapes, like 3 vertical lines, or S shapes. From there on the layers activate on unrecognizable shapes, but seem to learn something useful. In each layer there are a few filters which do not activate at all, whic might indicate something about the sample, or indicate that that filter does not learn anything useful."""
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[38]:


tf.compat.v1.disable_eager_execution()


# In[40]:


from tensorflow.keras import backend as K
import cv2

def plot_3_3():
  
  # print(y_test[0])
  model = load_model_from_file(base_dir, "model_2_2")
  sample = np.expand_dims(X_test_augmented[0], axis=0)
  # print(sample.shape)
  eight_output = model.output[:, 7]
  # print([l.name for l in model.layers])

  last_layer = model.get_layer('conv2d_15')
  grads = K.gradients(eight_output, last_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0,1,2))
  iterate = K.function([model.input], [pooled_grads, last_layer.output[0]])
  pooled_grads_value, conv_layer_output_value = iterate([sample])

  for i in range(conv_layer_output_value[:,:].shape[-1]):
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  # print(conv_layer_output_value.shape)
  
  heatmap = np.mean(conv_layer_output_value, axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  # print(heatmap)
  heatmap = cv2.resize(heatmap, (sample.shape[2], sample.shape[1]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  
  heatmap_intensity = 0.0035
  sample_with_heatmap = heatmap * heatmap_intensity + sample * (1 - heatmap_intensity)
  
  # print(sample_with_heatmap.shape)
  
  RGB_im = cv2.cvtColor(sample_with_heatmap[0].astype(np.float32), cv2.COLOR_BGR2RGB)
  plt.imshow(RGB_im)
  plt.show()

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[29]:


from tensorflow.keras.applications.vgg16 import VGG16


# In[30]:


def build_model_4_1(learning_rate=0.0001):
  convolution_layers = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  for layer in convolution_layers.layers[:6]:
      layer.trainable = False
  model = models.Sequential()
  model.add(convolution_layers)
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']) 
  return model

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
              train=True, epochs=6, batch_size=64, learning_rate=0.0001)  
                
answer_q_4_1 = """
               Unfreezing more convolutional layers gives better results, but also requires more running time. Finally I choose to defreeze the last 6 layers, and apply early stopping at epoch 6 (which gives around 94 validation accuracy). However, it is a tradeoff between time and accuracy.I also tuned learning rate, between [0.005, 0.001, 0.0005, 0.0001]. Learning rate 0.0001 gave the best results. Another tweak I tried out is the number of nodes in the dense layer. I tweaked this between 56, 128 and 256. However the impact was not significant, so I sticked to 128. Finally I added a dropout layer again, to prevent overfitting.
               """
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[31]:


import pickle
import gzip
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def store_embedding(X, name):  
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
    pickle.dump(X, file_pi)

def load_embedding(name):
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
    return pickle.load(file_pi)

def store_embeddings():
  """ Stores all necessary embeddings to file
  """
  model = load_model_from_file(base_dir, "model_4_1")
  trained_conv_base = model.layers[0]
  X_train_embed = trained_conv_base.predict(X_train)
  X_val_embed = trained_conv_base.predict(X_val)
  X_test_embed = trained_conv_base.predict(X_test)
  store_embedding(X_train_embed.reshape((X_train_embed.shape[0],X_train_embed.shape[-1])), "X_train_embedding")
  store_embedding(X_val_embed.reshape((X_val_embed.shape[0],X_val_embed.shape[-1])), "X_val_embedding")
  store_embedding(X_test_embed.reshape((X_test_embed.shape[0],X_test_embed.shape[-1])), "X_test_embedding")

def generate_pipeline():
  """ Returns an sklearn pipeline.
  """
#   return make_pipeline(StandardScaler(), SVC(kernel="rbf", max_iter=50))
  return make_pipeline(StandardScaler(), BernoulliNB(alpha=0.0001))
#   return make_pipeline(StandardScaler(), GaussianNB())
#   return make_pipeline(StandardScaler(), RandomForestClassifier(verbose=1, n_jobs=-1, n_estimators=500, max_samples=500))
#   return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
  

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
  """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
  """
#   scores = cross_val_score(pipeline, X_train, y_train, cv=3, n_jobs=-1)
#   print(f"Cross validate scores={scores}")
  pipeline.fit(X_train, y_train)
#   print("Testing...")
  y_pred = pipeline.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  # print(f"Found {accuracy} accuracy on testset")
  return accuracy


def evaluation_4_2(X_train, y_train, X_test, y_test):
  """ Runs 'evaluate_pipeline' with embedded versions of the input data 
  and returns the accuracy.
  """
  #   print("Loading embeddings")
  X_train_embedding = load_embedding("X_train_embedding")
  X_val_embedding = load_embedding("X_val_embedding")
  X_test_embedding = load_embedding("X_test_embedding")
  y_train_embedding = np.argmax(y_train, axis=1)
  y_val_embedding = np.argmax(y_val, axis=1)
  y_test_embedding = np.argmax(y_test, axis=1)
  pipeline = generate_pipeline()
  #   print("Starting to evaluate")
  #   print("Starting to evaluate")
  return evaluate_pipeline(generate_pipeline(), X_train_embedding, y_train_embedding, X_test_embedding, y_test_embedding)

  
# store_embeddings()
eval_result = evaluation_4_2(X_train, y_train, X_test, y_test)


answer_q_4_2 = """For the pipeline I tried out the following classifiers: GaussianNB and BernoulliNB (98 on crossvalidate, 94 on test) , SVC with rbf kernel (94 on cv, 82 on test), RandomForest Classifier (98 on cv, 94.6 on test). Next to that I also wanted to try a KNN classifier, however with the huge amount of samples in X this was too slow. I also tried different preprocessing steps, but none of them in/decreased performance, probably because we are already dealing with output of convolutational layers"""
print("Pipeline:",generate_pipeline())

print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
data_augmenter = ImageDataGenerator(
    width_shift_range=0.1,
    zoom_range=(0.9, 0.9)"""
last_edit = 'May 26, 2020'