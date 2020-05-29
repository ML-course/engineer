#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/fennovde'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fill in your name using the format below and student ID number
your_name = "VAN_DEN_EIJNDE, FENNO"
student_id = "0819644"


# In[2]:


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
# base_dir = '../assignment-3-fennovde'


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

# In[50]:


from tensorflow.keras.utils import to_categorical

Xr = X.reshape((len(X),32,32,3))
Xr = Xr / 255.
yr = to_categorical(y)


# In[51]:


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

# In[14]:


import os
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # for use with tensorflow
from tensorflow.keras.models import model_from_json

def shout(text, verbose=1):
    """ Prints text in red. Just for fun.
    """
    if verbose>0:
        print('\033[91m'+text+'\x1b[0m')

# def load_model_from_file(base_dir, name, extension='.h5'):
#     """ Loads a model from a file. The returned model must have a 'fit' and 'summary'
#     function following the Keras API. Don't change if you use TensorFlow. Otherwise,
#     adapt as needed. 
#     Keyword arguments:
#     base_dir -- Directory where the models are stored
#     name -- Name of the model, e.g. 'question_1_1'
#     extension -- the file extension
#     """
#     try:
#         model = load_model(os.path.join(base_dir, name+extension))
#     except OSError:
#         shout("Saved model could not be found. Was it trained and stored correctly? Is the base_dir correct?")
#         return False
#     return model
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


# def save_model_to_file(model, base_dir, name, extension='.h5'):
#     """ Saves a model to file. Don't change if you use TensorFlow. Otherwise,
#     adapt as needed. 
#     Keyword arguments:
#     model -- the model to be saved
#     base_dir -- Directory where the models should be stored
#     name -- Name of the model, e.g. 'question_1_1'
#     extension -- the file extension
#     """
#     model.save(os.path.join(base_dir, name+extension))
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
#     return pd.Series(index=['min','max'],data=[x.min(),x.max()])
    pass



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

# In[15]:


from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers 
# import winsound


# In[16]:


def build_model_1_1():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=True, epochs=50, batch_size=32)


# In[17]:


answer_q_1_1 = """
Tried different N of nodes
Rather have small layers, however this configuration increased my accuracy quite much.
If first layer is increased more, accuracy got a bit better.
No negative values so I choose relu function, sigmoid was less arratic but relu converges faster.
Softmax is ideal for taking the highest probability of a class.
Model flattens after 20 epochs without overfitting.
Tried different learning rates in factors of 10, 0.0001 was best
Accuracy of 84%, model still doesn't overfit
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

# In[18]:


# Luminance-preserving RGB to greyscale conversion
def rgb2gray(X):
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)

# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val


# In[19]:


# Adjusted model
def build_model_1_2():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[20]:


#Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=True, epochs=50, batch_size=32)


# In[21]:


answer_q_1_2 = """
Model 1.2 is almost identical als 1.1, the accuracy is a little bit better for 1.2.
By grayscaling for a computer no information is lost, it decreases computational time.
However in general color pixels have a higher dimensionality so it could overfit more easily.
"""
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[22]:


def build_model_1_3():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=True, epochs=50, batch_size=32)


# In[23]:


answer_q_1_3 = """
When adding dropout the accuracy decreased.
Batch size reduction increased the accuracy, however at the cost of heavily increasing computational time.
However it only increased performance a bit.
Adagard converged a smoother, but lower accuracy (10%).
adelta had a lower accuracy.
with stochastic gradient descent an higher learning rate can be picked and can be compensated with the momentum.
Lower learning rate gives smoother curve, but needs more epochs.
"""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[24]:


def build_model_2_1():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.35))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=False),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# In[25]:


run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=True, epochs=20, batch_size=32)


# In[26]:


answer_q_2_1 = """
Started with one convlayer (incl. zero padding) with 32 filters, added layers till overfitting.
For deeper layers, scaled the amount of filters with a factor 2.
Third conv layer overfited so added dropout, in increasing steps. to regularize deep layers more.
An extra conv layer did not increase the accuracy. So tried double (three is not significantly better) layers with maxpooling(2,2) (max 3 layers) as in VGG.
Here the model started overfitting, so i added normalization for each layer, this increased the accuracy with 5%.
Tuned the nodes of dense layer, 128 seems to be fine and added normalization which increases the accuracy a bit.
Overall quite okay 94%, so did some finetuning
filtersize, optimizers and activation.
Lastly looked at different optimizers and different epochs.
"""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[27]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented data
# preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    shear_range=5,
)
# rotation_range=


augmented_split = train_datagen.flow(preprocessed_split[0], y_train, batch_size=64) , preprocessed_split[1], preprocessed_split[3]


# In[28]:


run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, 
               train=True, epochs=50, generator=True, batch_size=None, steps_per_epoch=len(X_train)/64)

# # winsound.Beep(440,1000)


# In[29]:


answer_q_2_2 = """
Set the steps to the number of batches for the augmented split.
Housenumbers are upright, so flipping decreased accuracy.
However subtle rotations and shears increased accuracy, however no greater than 5.
Numbers are mostly in the middle, so shifting is probably usefulll, found optimum for 0.2 both height and width.
Lastly tried zooming, but dit didn't improve the accuracy more in combination wit shifting.
Got an validation accuracy of 96%, so a 1% improvement by augmentation.
"""
print("Answer is {} characters long".format(len(answer_q_2_2)))


# ## Part 3. Model interpretation (10 points)
# ### Question 3.1: Interpreting misclassifications (2 points)
# Study which errors are still made by your last model (model_2_2) by evaluating it on the test data. You do not need to retrain the model.
# * What is the accuracy of model_2_2 on the test data? Store this in 'test_accuracy_3_1'.
# * Plot the confusion matrix in 'plot_confusion_matrix' and discuss which classes are often confused.
# * Visualize the misclassifications in more depth by focusing on a single
# class (e.g. the number '2') and analyse which kinds of mistakes are made for that class. For instance, are the errors related to the background, noisiness, etc.? Implement the visualization in 'plot_misclassifications'.
# * Summarize your findings in 'answer_q_3_1' (800char)

# In[30]:


from sklearn.metrics import confusion_matrix
svhn_classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5,
                 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}

model = load_model_from_file(base_dir, "model_2_2")
y_pred = model.predict(rgb2gray(X_test))

test_accuracy_3_1 = model.evaluate(rgb2gray(X_test), y_test)[1]


# In[31]:


def plot_confusion_matrix():
    misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]  
    cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    ax.set_xticklabels(list(svhn_classes.values()), rotation=45, ha="right")
    ax.set_yticklabels(list(svhn_classes.values()))
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    for i in range(100):
        ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")

def my_plot_misclassifications(Predicted=None,True_label=None):
    misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]  
    
    fig, axes = plt.subplots(1, 5,  figsize=(10, 5))
    N=[]
    
    for i in misclassified_samples:
        if True_label==None:
            if svhn_classes[np.argmax(y_pred[i])]==Predicted:
                N.append(i)
        if Predicted==None:
            if svhn_classes[np.argmax(y_test[i])]==True_label:
                N.append(i)
        if True_label==None and Predicted==None:
                N.append(i)
        else:
            if svhn_classes[np.argmax(y_test[i])]==True_label and svhn_classes[np.argmax(y_pred[i])]==Predicted:
                N.append(i)
    if len(N)<5:
        print('Only %s misclassified predictions available'%len(N))

    n=[randint(0,len(N)) for i in range(5)]  
    for nr, i in enumerate(n):
        im=N[i-1]
        axes[nr].imshow(X_test[im])
        axes[nr].set_xlabel("Predicted: %s,\n True : %s" % (svhn_classes[np.argmax(y_pred[im])],svhn_classes[np.argmax(y_test[im])]))
        axes[nr].set_xticks(()), axes[nr].set_yticks(())
        
def plot_misclassifications():
    my_plot_misclassifications()


# In[32]:


plot_confusion_matrix()


# In[33]:


plot_misclassifications()


# In[34]:


answer_q_3_1 = """
True 5 and 7  confused with 3 and 1, these number also look a like if plotted.
The true 1 is misclassified a lot on first view, however quantity of ones in the dataset is quite high (so also more training).
Some misclassifications with multiple numbers in the picture.
Furthermore a lot misclassifications could be explained by multiple numbers in an image and background noice displaying extra contours.
Also a lot of images have the wrong labelings, which contaminates learning and predictions.
"""
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# 
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[35]:


def plot_activations():
    model = load_model_from_file(base_dir, "model_2_2")
    img_tensor = np.expand_dims(rgb2gray(X_test)[0], axis=0) 

    layer_outputs = [layer.output for layer in model.layers[:15]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers[:9]:
        layer_names.append(layer.name)
        
    for layer_index in [0,2,6,8]: #interesting layers
        print(layer_index)
        for layer_name, layer_activation in zip(layer_names[layer_index:layer_index+1], activations[layer_index:layer_index+1]):

            size = layer_activation.shape[1]

            n_cols = layer_activation.shape[-1] // 16
            grid = np.zeros((size * n_cols, 16 * size))

            for col in range(n_cols):
                for row in range(16):
                    channel_image = layer_activation[0,:, :,col * 16 + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * grid.shape[1],
                                scale * grid.shape[0]))
            plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
            plt.imshow(grid, aspect='auto', cmap='viridis')

        plt.show()


# In[36]:


# plot_activations()


# In[37]:


answer_q_3_2 = """
The model predicts the True eight accurately.
Deeper layers got lower resolution, from Layer 9 and upwards it becomes quite unreadable.
First layer does activate few filters, 1/3 of the activated ones highlight edges.
The next filter all of them highlight edges.
Hereafter it becomes a bit more vague, but you can still clearly see activations around the numbers, or in the numbers.
In the last readable layer you see, it becomes somewhat vaguer but the you still see the contours of the eight.
"""
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpose the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[38]:


from tensorflow.keras import backend as K
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import cv2


# In[39]:


def plot_3_3():
    tf.compat.v1.disable_eager_execution()
    model = load_model_from_file(base_dir, "model_2_2")

    input_img = rgb2gray(X_test)[0]
    input_img = np.expand_dims(input_img,axis=0)

    layer_output = model.output[:,0]

    last_conv_layer = model.layers[14]
    if not last_conv_layer.name.startswith('conv'):
        print('No conventional layer selected')
    else:    
        grads = K.gradients(layer_output, last_conv_layer.output)[0]
        loss = K.mean(grads,axis=(0,1,2))
        loss_v,grads_v = K.function([model.input], [loss,last_conv_layer.output[0]])([input_img])
        for i in range(32):
            grads_v[:,:,i] *= loss_v[i]
            
        heatmap = np.mean(grads_v, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap,(32,32))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        unscaled_img = 255 * X_test[0]
        unscaled_img = unscaled_img.astype(int)
        
        superimposed = heatmap * 0.5 + unscaled_img
        plt.figure(figsize=(4,4))
        plt.title("class activation map")
        plt.imshow(superimposed.astype(int),interpolation='bilinear');

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[40]:


def build_model_4_1():  
    Keras_model=tensorflow.keras.applications.vgg16.VGG16(include_top=False, 
                                                      weights='imagenet', input_tensor=None, 
                                                      input_shape=(32,32,3), pooling=None, classes=1000)
    model=models.Sequential()
    for i in Keras_model.layers:
        model.add(i)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation ='sigmoid'))

    layer_trainborder=10  #18 border dense layers
    
    for L in model.layers[:layer_trainborder]:
        L.trainable = False
    
    for L in model.layers[layer_trainborder:]:
        L.trainable = True
        
    for i, layer in enumerate(model.layers):
        print(i,layer.name, layer.trainable)

    model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])    
    return model


run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
               train=True, epochs=16, batch_size=32)


# In[41]:


answer_q_4_1 = """
To prevent dataloss played with the amount and the nodes in dense layers.
Two dense layers gave a better accuracy.
The accuracy with all the layers frozen were quite bad, unfreezing considerably increased the accuracy.
With more unfrozen the accuracy increased even more, but it was stated only a few layers.
Overall the learning rate shouldn't be to high due to overwritting the previous trained model fully.
optimal was 0.001 which gave an accuracy of 91%
"""
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[42]:


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer, MinMaxScaler


# In[78]:


import pickle
import gzip

def store_embedding(X, name):
    X=model.predict(X)  
    with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
        pickle.dump(X, file_pi)

def load_embedding(name):
    with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
        return pickle.load(file_pi)

def store_embeddings():
    """ Stores all necessary embeddings to file
    """
    model = load_model_from_file(base_dir, "model_4_1")
    for i in range(6):
        model.pop()
    print('saving X_train')
    store_embedding(X_train, 'X_train')
    print('saving X_val')
    store_embedding(X_val, 'X_val')
    print('saving X_test')
    store_embedding(X_test, 'X_test')
    

def generate_pipeline():
    """ Returns an sklearn pipeline.
    """
    scale=make_pipeline(Normalizer())
    clf=KNeighborsClassifier()
    return Pipeline(steps=[('preprocess',scale),('Classifier',clf)])

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
    """
    
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[3]))
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[3]))
        
    pipeline.fit(X_train,y_train)
    return pipeline.score(X_test,y_test)

def evaluation_4_2(X_train, y_train, X_test, y_test):
    """ Runs 'evaluate_pipeline' with embedded versions of the input data 
    and returns the accuracy.
    """
    X_train = load_embedding('X_train')
    X_test = load_embedding('X_test')
    pipeline=generate_pipeline()
    return evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)


# In[1]:


answer_q_4_2 = """
For most classifiers first need to reshape the inputs.
Tried different scalings,normalized gave the best results.
For the classifiers Kneighbors and randomforrest were the two moest promising.
Tried finetuning these classes, Kneighbors gave the best result with 90.5% accuarcy.
Randomly Kneighbour stopped working with the 4d input, couldn't find the problem.
Therefore reshaped the inputs, which gave the exact same result.
"""
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    shear_range=5,
)"""
last_edit = 'May 26, 2020'