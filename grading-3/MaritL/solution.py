#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/MaritL'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "LOONEN, MARIT"
student_id = "0961708"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = True  


# In[ ]:


# Uncomment the following line to run in Google Colab
# get_ipython().system('pip install --quiet openml ')


# In[69]:


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


# In[71]:


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

# In[72]:


#Uncomment to link Colab notebook to Google Drive
# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# # base_dir = '/content/drive/My Drive/assignment-3-MaritL' # For Google Colab
#base_dir = './'


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
# #oml.config.cache_directory = os.path.expanduser(base_dir)


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

# In[78]:


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
from tensorflow.keras.models import model_from_json

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

# In[80]:


from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models

def build_model_1_1():
  model = models.Sequential()
  model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dense(200, activation='relu'))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(50, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer= optimizers.Adagrad(learning_rate=0.005),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model


run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=False, epochs=50, batch_size=50)


answer_q_1_1 = """
              As I was unsure of effects, I experimented with number of nodes in the layers(e.g.100-50-25) and batchsize(50 to 10). Adagrad as optimizer gave a smoother learning curve, but was slow, so I changed number of epochs to 50 and a somewhat higher learning rate (0.005) to speed up the learning (accuracy=.71). I tried adding an extra layer and more nodes per layer(400-200-100-50), which resulted in .82 accuracy. My model slightly overfits (loss is smaller than val_loss)
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


# In[82]:


X_train_g = rgb2gray(X_train)
X_val_g = rgb2gray(X_val)
X_test_g = rgb2gray(X_test)

# Replace with the preprocessed data
preprocessed_split = X_train_g, X_val_g, y_train, y_val


# Adjusted model
def build_model_1_2():
  model = models.Sequential()
  model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dense(200, activation='relu'))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dense(50, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer= optimizers.Adagrad(learning_rate=0.005),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=False, epochs=50, batch_size=50)

answer_q_1_2 = """
               I used rgb2gray to convert X_train and X_val, and put these and y_train y_val in the preprocessed_split. Using this to run the model built in 1.1 (except for reshaping the input layer to fit the 32x32x1 greyscale data), it yields similar results and learning curve as 1.1: it still slightly overfits and has a comparable accuracy of .81. I expected the model to perform better with the greyscaled data as it removes complexity that is not needed to recognize numbers.
               """
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[83]:


from keras import regularizers

def build_model_1_3():
  model = models.Sequential()
  model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
  model.add(layers.Dense(400, activation='relu'))
  model.add(layers.Dropout(0.1))
  model.add(layers.Dense(200, activation='relu'))
  model.add(layers.Dropout(0.1))
  model.add(layers.Dense(100, activation='relu'))
  model.add(layers.Dropout(0.1))
  model.add(layers.Dense(50, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer= optimizers.Adagrad(learning_rate=0.005),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=False, epochs=50, batch_size=32)

answer_q_1_3 = """
               L1 regularization did not converge well, L2 was much better but still slightly overfitted and resulted in lower accuracy than before. So I tried adding dropout layers (.5/.3), but these underfitted, so I lowered this to .1. This worked well; neither over- nor underfit, with the same accuracy as the before. I tried increasingly bigger values for the dropouts(.1, .2, .3), but this caused too much information to be lost, so I stayed at .1 for all dropout layers.
               """
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[84]:


def build_model_2_1():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
  model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Dropout(0.3))
  model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Dropout(0.4))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=False, epochs=30, batch_size=64)
answer_q_2_1 = """
               I started with 3 conv-layers (32-64-128 filters) and later increased this to 6 (2x32-2x64-2x128) to allow for higher model complexity. Throughout, I experimented somewhat, but fewer filters decreased accuraccy (not enough complexity), and more filters did not increase accuracy or did not converge. To reduce overfitting and provide basic translation invariance, I added maxpooling layers between conv layers, and later (as it still overfitted) also added dropout layers. After some experimentation with dropout fraction, increasing from .2 to .5 gave the best results (no over- nor underfit). I used adam as optimizer as the learning curve was smoother than rmsprop, and faster (and more accurate) than adagrad. Bigger filter size or bigger dense layer did not work.
               """
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[85]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")
augmented_train = train_datagen.flow(X_train_g, y_train, batch_size=64)
augmented_split = augmented_train, X_val_g, y_val

#def build_model_2_2():
  #model = models.Sequential()
  #model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
  #model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
  #model.add(layers.MaxPooling2D((2,2)))
  #model.add(layers.Dropout(0.05))
  #model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  #model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
  #model.add(layers.MaxPooling2D((2,2)))
  #model.add(layers.Dropout(0.1))
  #model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  #model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
  #model.add(layers.MaxPooling2D((2,2)))
  #model.add(layers.Dropout(0.2))
  #model.add(layers.Flatten())
  #model.add(layers.Dense(128, activation='relu'))
  #model.add(layers.Dense(10, activation='softmax'))
  #model.compile(optimizer='adam',
  #            loss='categorical_crossentropy',
  #            metrics=['accuracy'])
  #return model


steps = int(X_train_g.shape[0] / 64)
run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, 
               train=False, epochs=30, steps_per_epoch=steps, generator=True)
answer_q_2_2 = """
               I used ImageDataGenerator to create variations of X_train. I found that for all values above .2 (or for shear_range; .1) the model did not coverge(but .2 yielded better accuracy than .1). Also, for horizontal and/or vertical flips, the model could not converge (logical as numbers in the data are not mirrored/turned, 5&2, 6&9 are confusing). With the best values,no flips, this model performs similar to 2.1, but when I lower dropout(comments) lower val_los is found.
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

# In[86]:


from sklearn.metrics import confusion_matrix
model = load_model_from_file(base_dir, "model_2_2")

#Evaluation on test data
evaluation = model.evaluate(X_test_g, y_test)
test_accuracy_3_1 = evaluation[1]

#Plotting the consfusion matrix
y_pred = model.predict(X_test_g)
misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]

def plot_confusion_matrix():
  labels = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9", 9: "0"}
  cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
  fig, ax = plt.subplots()
  im = ax.imshow(cm)
  ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
  ax.set_xticklabels(list(labels.values()), rotation=45, ha="right")
  ax.set_yticklabels(list(labels.values()))
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  for i in range(100):
    ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")

#Plotting some misclassifications for 'missed' 7's, as these are seem to be often classified as 1's and 2's.
def plot_misclassifications():
  misclassified_seven= []
  for i in misclassified_samples:
    if np.argmax(y_test[i])==6:
      misclassified_seven.append(i)
  
  fig, axes = plt.subplots(1, 7,  figsize=(10, 5))
  for nr, i in enumerate(misclassified_seven[:7]):
    axes[nr].imshow(X_test[i])
    axes[nr].set_xlabel("Predicted: %s,\n Actual : %s" % (((np.argmax(y_pred[i])+1)%10),((np.argmax(y_test[i])+1)%10)))
    axes[nr].set_xticks(()), axes[nr].set_yticks(())

  fig, axes = plt.subplots(1, 7,  figsize=(10, 5))
  for nr, i in enumerate(misclassified_seven[7:14]):
    axes[nr].imshow(X_test[i])
    axes[nr].set_xlabel("Predicted: %s,\n Actual : %s" % (((np.argmax(y_pred[i])+1)%10),((np.argmax(y_test[i])+1)%10)))
    axes[nr].set_xticks(()), axes[nr].set_yticks(())

  fig, axes = plt.subplots(1, 7,  figsize=(10, 5))
  for nr, i in enumerate(misclassified_seven[14:21]):
    axes[nr].imshow(X_test[i])
    axes[nr].set_xlabel("Predicted: %s,\n Actual : %s" % (((np.argmax(y_pred[i])+1)%10),((np.argmax(y_test[i])+1)%10)))
    axes[nr].set_xticks(()), axes[nr].set_yticks(())

answer_q_3_1 = """
               Accuracy on testdata is .935, which is close to val_accuracy in model 2.2: it does not overfit. To see where mistakes are, I plot the confusionmatrix. You can clearly see the diagonal line (yay), and most errors seem to be made a small amount of the time (e.g. max 30 times). But there are some exceptions: 7s are relatively often classified as 1 or 2, and 5s are quite often classified as 3. Because of the errors of the 7, I visualize the first 14 cases of actual 7s that were misclassified. Some errors are very understandable; when there is more than one number in the picture, when the picture is very vague, and some of the 7s do just really look like 1s...
               """
print("Answer is {} characters long".format(len(answer_q_3_1)))

plot_confusion_matrix()
plot_misclassifications()


# In[87]:


#Target Image, first of X_test

print(y_test[0])
print(np.argmax(y_test[0]))
print(y_pred[0])
print(np.argmax(y_pred[0]))
plt.imshow(X_test[0])
print(X_test[0].shape)


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[88]:


model.summary()


# In[89]:


from tensorflow.keras import models

def plot_activations():
  img_tensor = X_test_g[0]
  img_tensor = np.expand_dims(img_tensor, axis=0) # shape from  (32, 32, 1) to (1, 32, 32, 1)
  layer_outputs = [layer.output for layer in model.layers[:15]]
  activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
  activations = activation_model.predict(img_tensor)

  plt.rcParams['figure.dpi'] = 120
  first_layer_activation = activations[0]

  images_per_row = 16
  layer_names = []
  for layer in model.layers[:15]:
    layer_names.append(layer.name)

  # Which layers I want to show (1, 2, 5 and 6, the first four convulutional layers)
  for x in [0,1,4,5]:
    layer_index=x

    start = layer_index
    end = layer_index+1
    # Display feature maps
    for layer_name, layer_activation in zip(layer_names[start:end], activations[start:end]):
        # number of features in the feature map
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
        plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()

# Visualize the activations for the first five layers

# plot_activations()

#Answer
answer_q_3_2 = """
               I have visualized the first(32x32) and middle(16x16) conv layers; after that the resolution is too low to interpret due to max-pooling. I can see that layer1 responds to some elements such as vertical edges (e.g. 10th filter, 1st row). Layer2 already seems to combine some of the basic elements (e.g. curves). The activations in layer 5 and 6 are much more abstract, but e.g. seem to be responding to content on different sides of edges, perimeters, etc. 
               """
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[90]:


from keras import backend as K
import cv2
# from google.colab.patches import cv2_imshow
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

def plot_3_3():
  input_image = np.expand_dims(X_test_g[0], axis=0)

  with tf.GradientTape() as tape:
    last_conv_layer = model.get_layer('conv2d_89')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(input_image)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap = heatmap.reshape((8, 8))

  img = X_test_g[0]
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
  image = heatmap * 1.3 * img
  cv2_imshow(cv2.resize(image, (300, 300)))

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[91]:


from tensorflow.keras.applications.vgg16 import VGG16

def build_model_4_1():
  conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
  
  #Last three layers of the model (layers in block 5) unfrozen
  for layer in conv_base.layers:
    #print(layer.name)
    if layer.name == 'block5_conv1' or 'block5_conv2' or 'block5_conv3':
      layer.trainable = True
    else:
      layer.trainable = False

  #model with convulutional base VGG16
  model = models.Sequential()
  model.add(conv_base)
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
               train=False, epochs=15, batch_size=64)
answer_q_4_1 = """
               In model 4.1, I first froze the entire conv base and added a dense hidden layer (and dense, softmax). With some different settings (extra layer, more filters), this leaded to max .58 val_acc. When unfreezing the 1st layer of the 5th conv-block, val_acc rose to around .8. Fine tuning by unfreezing all 3 layers in the 5th block immediately raised val_acc to .93. This finally led to an accuracy similar to my previous models (although somewhat overfitting).
               """
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[92]:


import pickle
import gzip
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

model_4 = load_model_from_file(base_dir, "model_4_1")

def store_embedding(X, name):  
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
    pickle.dump(X, file_pi)

def load_embedding(name):
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
    return pickle.load(file_pi)

def store_embeddings(X_train, X_test):
  """ Stores all necessary embeddings to file
  """
  conv_base_4 = model_4.get_layer("vgg16")
  X_train = conv_base_4.predict(X_train)
  store_embedding(X_train, "X_train_e")

  X_test = conv_base_4.predict(X_test)
  store_embedding(X_test, "X_test_e")

def generate_pipeline():
  """ Returns an sklearn pipeline."""
  pipe = make_pipeline(DecisionTreeClassifier())
  return pipe

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
  """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score."""
  
  test = generate_pipeline().fit(X_train, y_train);
  score = test.score(X_test, y_test)
  return score

def evaluation_4_2(X_train, y_train, X_test, y_test):
  """ Runs 'evaluate_pipeline' with embedded versions of the input data 
  and returns the accuracy.
  """
  #Check if there are embeddings if so: load, if not: store them
  if os.path.join(base_dir, 'X_train_e_embedding.p') and (base_dir, 'X_test_e_embedding.p') ==True:
    X_train_embedded = load_embedding("X_train_e")
    X_train_embedded = X_train_embedded.reshape(63544, 1*1*512)
    X_test_embedded = load_embedding("X_test_e")
    X_test_embedded = X_test_embedded.reshape(19858, 1*1*512)
  else:
    store_embeddings(X_train, X_test)
    X_train_embedded = load_embedding("X_train_e")
    X_train_embedded = X_train_embedded.reshape(63544, 1*1*512)
    X_test_embedded = load_embedding("X_test_e")
    X_test_embedded = X_test_embedded.reshape(19858, 1*1*512)
  
  #get pipeline
  pipeline = generate_pipeline()
  
  #calculate accuracy of the (embedded)test data from the pipeline trained on the (embedded)train data
  score = evaluate_pipeline(pipeline, X_train_embedded, y_train, X_test_embedded, y_test)

  return score

print(evaluation_4_2(X_train, y_train, X_test, y_test))
answer_q_4_2 = """
              In store_embeddings, I used the conv_base of model_4_1 to predict X_test and X_train, and store_embedding to save these. In generate_pipeline, I added a simple pipeline. Evaluate_pipeline fits the pipe on embedded train data, and then evaluates on embedded test data. Evaluate_4_2 looks for the embedded data (base_dir), and otherwise uses store_embeddings, and then runs evaluate_pipeline. Using RF, I get .92 accuracy, surprisginly almost as high as the past models.
               """
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
train_datagen = ImageDataGenerator(
    rotation_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")"""
last_edit = 'May 26, 2020'