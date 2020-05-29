#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/Jolanrensen'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "Rensen, Jolan"
student_id = "0946444"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = False


# In[ ]:


# Uncomment the following line to run in Google Colab
# get_ipython().system('pip install --quiet openml ')


# In[ ]:


# Uncomment the following line to run in Google Colab
#%tensorflow_version 2.x
import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU') # Check whether GPUs are available


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import openml as oml
import numpy as np
import matplotlib.pyplot as plt
from typing import *
from openml import OpenMLDataset
from tensorflow.keras import models, layers, losses, optimizers, activations, metrics, regularizers
ndarray = np.ndarray


# In[ ]:


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


# # base_dir = '/content/drive/My Drive/TU e/Web Info/Assignment 3/' # For Google Colab
# base_dir: str = './'


# In[ ]:


#Uncomment to link Colab notebook to Google Drive
# # from google.colab import drive
# # drive.mount('/content/drive')


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
# import os
# # oml.config.cache_directory = os.path.expanduser('/content/cache')


# In[ ]:


# Download Streetview data. Takes a while (several minutes), and quite a bit of
# memory when it needs to download. After caching it loads faster.
SVHN: OpenMLDataset = oml.datasets.get_dataset(41081)

X: ndarray; y: ndarray
X, y, _, _ = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)

X;
y;


# Reshape, sample and split the data

# In[ ]:


from tensorflow.keras.utils import to_categorical

Xr: ndarray = X.reshape((len(X),32,32,3))
Xr = Xr / 255.
yr: ndarray = to_categorical(y)

Xr;
yr;


# In[ ]:


# DO NOT EDIT. DO NOT OVERWRITE THESE VARIABLES.
from sklearn.model_selection import train_test_split
# We do an 80-20 split for the training and test set, and then again a 80-20 split into training and validation data
X_train_all: ndarray; X_test: ndarray; y_train_all: ndarray; y_test: ndarray
X_train_all, X_test, y_train_all, y_test = train_test_split(Xr,yr, stratify=yr, train_size=0.8, test_size=0.2, random_state=1)

X_train_all;
X_test;
y_train_all;
y_test;


X_train: ndarray; X_val: ndarray; y_train: ndarray; y_val: ndarray
X_train, X_val, y_train, y_val = train_test_split(X_train_all,y_train_all, stratify=y_train_all, train_size=0.8, random_state=1)

X_train;
X_val;
y_train;
y_val;


evaluation_split: Tuple[ndarray, ndarray, ndarray, ndarray] = (X_train, X_val, y_train, y_val)

evaluation_split;


# Check the formatting - and what the data looks like

# In[ ]:


from random import randint

# Takes a list of row ids, and plots the corresponding images
# Use grayscale=True for plotting grayscale images
def plot_images(X: List[ndarray], y: List[ndarray], grayscale: bool=False):
    fig, axes = plt.subplots(1, len(X),  figsize=(10, 5))
    for n in range(len(X)):
        if grayscale:
            axes[n].imshow(X[n], cmap='gray')
        else:
            axes[n].imshow(X[n])
        axes[n].set_xlabel((np.argmax(y[n])+1)%10) # Label is index+1
        axes[n].set_xticks(()), axes[n].set_yticks(())
    plt.show();

images: List[int] = [randint(0,len(X_train)) for i in range(5)]
X_random: List[ndarray] = [X_train[i] for i in images]
y_random: List[ndarray] = [y_train[i] for i in images]
#plot_images(X_random, y_random, True)


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

def shout(text: any, verbose=1):
    """ Prints text in red. Just for fun.
    """
    if verbose>0:
        print('\033[91m'+text+'\x1b[0m')

def load_model_from_file(base_dir: str, name: str, extension: str ='.h5') -> models.Model:
    """ Loads a model from a file. The returned model must have a 'fit' and 'summary'
    function following the Keras API. Don't change if you use TensorFlow. Otherwise,
    adapt as needed. 
    Keyword arguments:
    base_dir -- Directory where the models are stored
    name -- Name of the model, e.g. 'question_1_1'
    extension -- the file extension
    """
    try:
        model = load_model(os.path.join(base_dir, name+extension))
    except OSError:
        shout("Saved model could not be found. Was it trained and stored correctly? Is the base_dir correct?")
        return False
    return model

def save_model_to_file(model: models.Model, base_dir: str, name: str, extension: str = '.h5'):
    """ Saves a model to file. Don't change if you use TensorFlow. Otherwise,
    adapt as needed. 
    Keyword arguments:
    model -- the model to be saved
    base_dir -- Directory where the models should be stored
    name -- Name of the model, e.g. 'question_1_1'
    extension -- the file extension
    """
    model.save(os.path.join(base_dir, name+extension))

# Helper function to extract min/max from the learning curves
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

# DO NOT EDIT
def run_evaluation(name: str, model_builder: Callable[[Optional[any]], models.Model], 
                   data: Union[Tuple[Iterator, ndarray, ndarray], Tuple[ndarray, ndarray, ndarray, ndarray]], 
                   base_dir: str, train: bool = True, generator: bool = False, epochs: int = 3, batch_size: int = 32, 
                   steps_per_epoch: int = 60, verbose: int = 1, **kwargs):
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
    generator -- whether the data is given as a generator or not
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


# In[ ]:


# Toy usage example
# Remove before submission
# from tensorflow.keras import models
# from tensorflow.keras import layers

def build_toy_model() -> models.Model:
    model: models.Sequential = models.Sequential()
    model.add(layers.Reshape(target_shape=(3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(units=10, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# First build and store
# with tf.device('/cpu:0'):
#     run_evaluation(name="toy_example", model_builder=build_toy_model, data=evaluation_split, base_dir=base_dir, 
#                train=True, epochs=3, batch_size=32)


# In[ ]:


# Toy usage example
# Remove before submission
# With train=False: load from file and report the same results without rerunning
# run_evaluation(name="toy_example", model_builder=build_toy_model, data=evaluation_split, base_dir=base_dir, 
#                train=False)


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[ ]:


def build_model_1_1() -> models.Model:
    model: models.Sequential = models.Sequential([
            layers.Reshape(target_shape=(3072,), input_shape=(32,32,3)),
            layers.Dense(units=160, activation=activations.relu),
            layers.Dense(units=80, activation=activations.relu),
            layers.Dense(units=40, activation=activations.relu),
            layers.Dense(units=20, activation=activations.sigmoid),
            layers.Dense(units=10, activation=activations.sigmoid),                          
    ])
   
    model.compile(
        optimizer=optimizers.RMSprop(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.Accuracy().name]
    )
    
    return model

# with tf.device('/gpu:0'):
# run_evaluation(name="model_1_1", model_builder=build_model_1_1, data=evaluation_split, base_dir=base_dir, 
#                train=True, epochs=20, batch_size=32)
run_evaluation(name="model_1_1", model_builder=build_model_1_1, data=evaluation_split, base_dir=base_dir, 
               train=False)

answer_q_1_1 = """
               First I reshaped the model to make it 1-dimensional.
               Next I added some Dense layers and found that the last one needed to have 10 nodes due to the input data.
               Following the pyramid shape I experimented with sigmoid, relu and softmax activation functions and found
               this setup to be the best performing, reaching over 70% accuracy (with 12 epochs).
               I used the compile function from Lab 6.
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
def rgb2gray(X: ndarray) -> ndarray:
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)


# In[ ]:


# Replace with the preprocessed data
preprocessed_split: Tuple[ndarray, ndarray, ndarray, ndarray] = (rgb2gray(X_train), rgb2gray(X_val), y_train, y_val)

# Adjusted model
def build_model_1_2() -> models.Model:
    model: models.Sequential = models.Sequential([
            layers.Reshape(target_shape=(1024,), input_shape=(32,32,1)),
            layers.Dense(units=160, activation=activations.relu),
            layers.Dense(units=80, activation=activations.relu),
            layers.Dense(units=40, activation=activations.relu),
            layers.Dense(units=20, activation=activations.sigmoid),
            layers.Dense(units=10, activation=activations.sigmoid),                       
    ])
    
    model.compile(
        optimizer=optimizers.RMSprop(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.Accuracy().name]
    )
    
    return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
# run_evaluation(name="model_1_2", model_builder=build_model_1_2, data=preprocessed_split, base_dir=base_dir, 
#                train=True, epochs=20, batch_size=32)
run_evaluation(name="model_1_2", model_builder=build_model_1_2, data=preprocessed_split, base_dir=base_dir, 
               train=False)

answer_q_1_2 = """
               The result of the grayscale version is higher than the one before. This might be because the
               network can now ignore color and just focus on shapes, as colors do not influence the resulting digit.
               It reaches 70% accuracy already in 10 epochs.
               """
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[ ]:


def build_model_1_3() -> models.Model:
    model: models.Sequential = models.Sequential([
            layers.Reshape(target_shape=(1024,), input_shape=(32,32,1)),
            layers.Dense(units=160, activation=activations.relu, bias_regularizer=regularizers.l2(.001)),
            layers.Dense(units=80, activation=activations.relu, bias_regularizer=regularizers.l2(.001)),
            layers.Dense(units=40, activation=activations.relu, bias_regularizer=regularizers.l2(.001)),
            layers.Dense(units=20, activation=activations.sigmoid),
            layers.Dense(units=10, activation=activations.sigmoid),                           
    ])
    
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=.001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.Accuracy().name]
    )
    
    
    return model

# run_evaluation(name="model_1_3", model_builder=build_model_1_3, data=preprocessed_split, base_dir=base_dir, 
#                train=True, epochs=50, batch_size=64)
run_evaluation(name="model_1_3", model_builder=build_model_1_3, data=preprocessed_split, base_dir=base_dir, 
               train=False)

answer_q_1_3 = """
               Interestingly, lowering the learning rate compared to the default (0.001), makes the accuracy improve for each epoch.
               Decreasing the batch_size increases the calculation time per epoch, but not necessarily improves the accuracy.
               Increasing the batch_size however does allow to easily go through 50 epochs and reach 81%.
               There is some slight overfitting regarding the loss however.
               """
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[ ]:


def build_model_2_1() -> models.Model:

  model: models.Sequential = models.Sequential([
      layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activations.relu, padding="same", input_shape=(32, 32, 1)),
      layers.BatchNormalization(),
      layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activations.relu, padding="same"),
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2, 2)),

      layers.Dropout(rate=.2),
      layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu, padding="same"),
      layers.BatchNormalization(),
      layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu, padding="same"),
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2, 2)),

      layers.Dropout(rate=.3),
      layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, padding="same"),
      layers.BatchNormalization(),
      layers.Conv2D(filters=128, kernel_size=(3, 3), activation=activations.relu, padding="same"),
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2, 2)),

      layers.Dropout(rate=.4),
      layers.Flatten(),
      layers.Dense(units=128, activation=activations.relu),
      layers.BatchNormalization(),

      layers.Dropout(rate=.5),
      layers.Dense(units=10, activation=activations.softmax),                           
  ])

  model.compile(
    optimizer=optimizers.RMSprop(learning_rate=.001),
    loss=losses.CategoricalCrossentropy(),
    metrics=[metrics.Accuracy().name]    
  )

  return model

# run_evaluation(name="model_2_1", model_builder=build_model_2_1, data=preprocessed_split, base_dir=base_dir, 
#                train=True, epochs=10, batch_size=32)
run_evaluation(name="model_2_1", model_builder=build_model_2_1, data=preprocessed_split, base_dir=base_dir, 
               train=False)

answer_q_2_1 = """
               I noticed a lot of overfitting using the first example from Lab 6,
               so I followed the rest of the lab, adding VGG-like additions to the model
               as well as regularization, dropout and batch normalization.
               """
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[ ]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

X_train, X_val, y_train, y_val = preprocessed_split

train_data_generator: ImageDataGenerator = ImageDataGenerator(  # 75.4%
    # width_shift_range=.5,  # 49.1%
    # height_shift_range=.5,  # 50.5%
    # horizontal_flip=True,  # 61.4%
    # vertical_flip=True,
    shear_range=10,
    zoom_range=.2,  # 59%
    rotation_range=10,
    # rescale=1  # overfitting
)

iterator_train: NumpyArrayIterator = train_data_generator.flow(x=X_train, y=y_train, batch_size=64)

augmented_split: Tuple[NumpyArrayIterator, ndarray, ndarray] = (iterator_train, X_val, y_val)

# run_evaluation(name="model_2_2", model_builder=build_model_2_1, data=augmented_split, generator=True, base_dir=base_dir, 
#                train=True, epochs=10, batch_size=32)
run_evaluation(name="model_2_2", model_builder=build_model_2_1, data=augmented_split, generator=True, base_dir=base_dir, 
               train=False)

answer_q_2_2 = """
              The effects of width shift and height shift are comparable. Combined, however, the effects are that the result gets much worse.
              The same holds for horizontal- vs vertical flips, except that combining the two only decreases the accuracy by a third (compared to having just one of the two active).
              Shearing has almost no negative effect on the accuracy, however there does appear to be overfitting.
              Zooming up to 0.5 seems doable, however, increasing the zoom range worsens the results.
              Rotations at 20 degrees are very well handled, yielding almost no negative effect at 20 degrees, however slightly more at higher ones.
              Rescaling does not affect the accuracy at all, but it makes the model overfit enourmously, just like no augmentation of the data does.
              Shearing, rotating and zooming work fine together, decreasing the amount of overfitting a lot.
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

# In[ ]:


from sklearn.metrics import confusion_matrix
from matplotlib import image, figure, axes
import random

classes: List[str] = [str(x + 1)[-1] for x in range(10)]
print(classes)
model: models.Model = load_model_from_file(base_dir=base_dir, name="model_2_2")
X_test_gray: ndarray = rgb2gray(X_test)

result: List[float] = model.evaluate(x=X_test_gray, y=y_test)
test_accuracy_3_1: float = result[1]

y_pred: ndarray = model.predict(x=X_test_gray)
misclassified_samples: ndarray = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]

def plot_confusion_matrix():
  cm: ndarray = confusion_matrix(
      y_true=np.argmax(y_test, axis=1), 
      y_pred=np.argmax(y_pred, axis=1)
  )

  fig: figure.Figure
  ax: axes.Axes
  fig, ax = plt.subplots()

  im: image.AxesImage = ax.imshow(X=cm)
  ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
  ax.set_xticklabels(classes, rotation=45, ha="right")
  ax.set_yticklabels(classes)
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  for i in range(100):
      ax.text(int(i / 10), i % 10, cm[i % 10, int(i / 10)], ha="center", va="center", color="w")


plot_confusion_matrix()

def plot_misclassified_as_1():
  fig: figure.Figure
  ax: axes.Axes
  fig, ax = plt.subplots(1, 5, figsize=(10, 5))

  misclassified_as_1: List[int] = list(filter(lambda i: classes[np.argmax(y_pred[i])] == "1", misclassified_samples))

  for nr, i in enumerate(random.choices(misclassified_as_1, k=5)[:5]):
    ax[nr].imshow(X=X_test[i])
    ax[nr].set_xlabel(f"Predicted: {classes[np.argmax(y_pred[i])]},\n Actual : {classes[np.argmax(y_test[i])]}")
    ax[nr].set_xticks(())
    ax[nr].set_yticks(())

  plt.show()

def plot_5_3_misclassifications():
  fig: figure.Figure
  ax: axes.Axes
  fig, ax = plt.subplots(1, 5, figsize=(10, 5))

  misclassified_as_1: List[int] = list(
      filter(
          lambda i: classes[np.argmax(y_pred[i])] == "5" and classes[np.argmax(y_test[i])] == "3" or classes[np.argmax(y_pred[i])] == "3" and classes[np.argmax(y_test[i])] == "5", 
          misclassified_samples
      )
  )

  for nr, i in enumerate(random.choices(misclassified_as_1, k=5)[:5]):
    ax[nr].imshow(X=X_test[i])
    ax[nr].set_xlabel(f"Predicted: {classes[np.argmax(y_pred[i])]},\n Actual : {classes[np.argmax(y_test[i])]}")
    ax[nr].set_xticks(())
    ax[nr].set_yticks(())

  plt.show()

def plot_misclassifications():
  plot_misclassified_as_1()
  plot_5_3_misclassifications()


plot_misclassifications()

answer_q_3_1 = """
               From the confusion matrix can be seen that 1 is predicted often when that is not the true value. 
               This happens for all numbers, however, it happens the most for 7, 3 and 4. Aside from this, predicting a 6 when it
               is actually a 8, or predicting a 3 when it's actually a 5 (or vice versa) also seems to happen a lot.
               In the case for 1, by plotting some misclassifications, it can be seen that often 1 does appear in the same picture, 
               however not in the center, so the actual label is different. Sometimes the area between two numbers is also seen as a 1,
               as well als just overall blurry pictures with a vague line in them.
               The confusion between 3 and 5 often seems to originate when the top half of the 3 or 5 is smaller than the bottom half.
               """

print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[ ]:


model: models.Model = load_model_from_file(base_dir=base_dir, name="model_2_2")
model.summary()

img_tensor_color: ndarray = np.expand_dims(X_test[0], axis=0)
img_tensor: ndarray = np.expand_dims(rgb2gray(X_test)[0], axis=0)

# Extracts the outputs of the layers:
layer_outputs: List[tf.Tensor] = [layer.output for layer in model.layers]
# Creates a model that will return these outputs, given the model input:
activation_model: models.Model = models.Model(inputs=model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
all_activations: List[ndarray] = activation_model.predict(img_tensor)


def plot_activations():
  plt.rcParams['figure.dpi'] = 120

  f: figure.Figure; ax1: axes.Axes
  f, ax1 = plt.subplots(sharey=True)

  ax1.imshow(X=img_tensor_color[0])
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1.set_xlabel('Input image')

  # for i in [0, 2, 4, 6, 8, 10, 12, 16]:
  #   first_layer_activation: ndarray = activations[i]

  #   f: figure.Figure; ax2: axes.Axes
  #   f, ax2 = plt.subplots(sharey=True)

  #   ax2.matshow(Z=first_layer_activation[0, :, :, 2], cmap='viridis')
  #   ax2.set_xticks([])
  #   ax2.set_yticks([])
  #   ax2.set_xlabel(f'Activation of filter {i + 1}');


  images_per_row: int = 15
  layer_names: List[str] = [layer.name for layer in model.layers]

  def plot_layer(layer_index: int):
    start: int = layer_index
    end: int = layer_index + 1

    # Now let's display our feature maps
    layer_name: str; layer_activation: ndarray
    for layer_name, layer_activation in zip(layer_names[start:end], all_activations[start:end]):

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
              channel_image = layer_activation[0, :, :, col * images_per_row + row]
              # Post-process the feature to make it visually palatable
              channel_image -= channel_image.mean()
              channel_image /= channel_image.std()
              channel_image *= 64
              channel_image += 128
              channel_image = np.clip(channel_image, 0, 255).astype('uint8')
              display_grid[
                           col * size : (col + 1) * size,
                           row * size : (row + 1) * size
              ] = channel_image

      # Display the grid
      scale = 1. / size
      plt.figure(figsize=(scale * display_grid.shape[1],
                          scale * display_grid.shape[0]))
      plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
      plt.grid(False)
      plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


  for i in [0, 2, 6, 9, 10, 12, 14, 17]:
    plot_layer(layer_index=i)


# plot_activations()

answer_q_3_2 = """
               While it gets harder to see from filter 11 onwards, it is clear that the model is focussing on
               the correct shape of the number 8 while discarding the number 5 that is not centered.
               It's curious to see that around filter 7 in some cases the colors are inverted, meaning that while at first the number 8
               was colored in, now it's the background that is colored in.
               """

print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[ ]:


import cv2
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import GraphExecutionFunction

tf.compat.v1.disable_eager_execution()

K.clear_session()
model: models.Model = load_model_from_file(base_dir=base_dir, name="model_2_2")
model.summary()

img_tensor: ndarray = np.expand_dims(rgb2gray(X_test)[0], axis=0)

# # Extracts the outputs of the layers:
# layer_outputs: List[tf.Tensor] = [layer.output for layer in model.layers]
# # Creates a model that will return these outputs, given the model input:
# activation_model: models.Model = models.Model(inputs=model.input, outputs=layer_outputs)

# # This will return a list of 5 Numpy arrays:
# # one array per layer activation
# activations: List[ndarray] = activation_model.predict(img_tensor)

def plot_activation_map():
  last_conv_layer: layers.Layer = model.get_layer('conv2d_312')
  model_output: tf.Tensor = model.output[:, 9]
  grads: tf.Tensor = K.gradients(model_output, last_conv_layer.output)[0]
  pooled_grads: tf.Tensor = K.mean(grads, axis=(0, 1, 2))
  iterate: GraphExecutionFunction = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  pooled_grads_value: ndarray; conv_layer_output_value: ndarray
  pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

  for i in range(128):
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  preds: ndarray = model.predict(img_tensor)
  print('Predicted:', (np.argmax(preds) + 1) % 10)

  heatmap: ndarray = np.mean(conv_layer_output_value, axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  plt.matshow(heatmap)
  plt.show()

  x: ndarray = X_test[0]

  heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  # plt.matshow(heatmap)
  # plt.show()

  superimposed_img = heatmap * .4 + np.uint8(255 * x)

  cv2.imwrite('./heatmap.jpg', superimposed_img)
  img = cv2.imread('./heatmap.jpg')

  RGB_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  plt.rcParams['figure.dpi'] = 120
  plt.imshow(RGB_im)
  plt.title('Class activation map')
  plt.xticks([])
  plt.yticks([])
  plt.show()

# plot_activation_map()

def plot_3_3():
  plot_activation_map()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16


def build_model_4_1() -> models.Model:
  conv_base: models.Model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

  conv_base.summary()

  model: models.Sequential = models.Sequential([
      conv_base,
      layers.Flatten(),
      layers.Dense(units=10, activation=activations.sigmoid),                  
  ])

  conv_base.trainable = False

  for layer in ["block5_pool", "block5_conv1", "block5_conv2", "block5_conv3", 
                "block4_pool", "block4_conv1", "block4_conv2", "block4_conv3"]:
    conv_base.get_layer(layer).trainable = True

  model.compile(
          optimizer=optimizers.RMSprop(),
          loss=losses.CategoricalCrossentropy(),
          metrics=[metrics.Accuracy().name]
      )
  
  model.summary()

  return model


# run_evaluation(name="model_4_1", model_builder=build_model_4_1, data=evaluation_split, base_dir=base_dir, 
#                train=True, epochs=10, batch_size=32)
run_evaluation(name="model_4_1", model_builder=build_model_4_1, data=evaluation_split, base_dir=base_dir, 
               train=False)

answer_q_4_1 = """
                It takes a very long time to train the model having its convolutional base frozen with an accuracy that only slowly gets better, however not much highter than 55%.
                Unfreezing the last (or last two) convolutional blocks of the convolutional base doesn't do anything notable.
              
               """
               
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[28]:


import pickle
import gzip
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_validate, KFold



# from Assignment 2
def flexible_pipeline(
        categorical: List[str], 
        clf: Union[ClassifierMixin, BaseEstimator], 
        scaler: Optional[TransformerMixin] = StandardScaler(), 
        encoder: Optional[TransformerMixin] = OneHotEncoder()
) -> Pipeline:
    """ Returns a pipeline that imputes all missing values, encodes categorical features and scales numeric ones
    Keyword arguments:
    categorical -- A list of categorical column names. Example: ['gender', 'country'].
    clf -- any scikit-learn classifier
    scaler -- any scikit-learn feature scaling method (Optional)
    encoder -- any scikit-learn category encoding method (Optional)
    Returns: a scikit-learn pipeline which preprocesses the data and then runs the classifier
    """
    categorical_pipe: Pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        encoder
    )
    numerical_pipe: Pipeline = make_pipeline(
        SimpleImputer(strategy="mean")
    )
    if scaler is not None:
        numerical_pipe.steps.insert(1, ["scaler", scaler])
    
    transform: ColumnTransformer = make_column_transformer((categorical_pipe, categorical), remainder=numerical_pipe)
    return Pipeline(steps=[("preprocess", transform), ("classify", clf)])


def store_embedding(X: ndarray, name: str):  
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
    pickle.dump(X, file_pi)


def load_embedding(name: str) -> ndarray:
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
    return pickle.load(file_pi)


def store_embeddings():
  """ Stores all necessary embeddings to file
  """
  model: models.Model = load_model_from_file(base_dir=base_dir, name="model_4_1")
  model._layers.pop()

  model.compile(
            optimizer=optimizers.RMSprop(),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.Accuracy().name]
        )

  model.summary()

  store_embedding(X=model.predict(X_train_all), name="X_train_all")
  store_embedding(X=model.predict(X_test), name="X_test")


# store_embeddings()


def generate_pipeline() -> Pipeline:
  """ Returns an sklearn pipeline.
  """
  return flexible_pipeline(
      categorical=[str(x + 1)[-1] for x in range(10)],
      clf=SVC(kernel="rbf", random_state=1),
  )


def evaluate_pipeline(pipeline: Pipeline, X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray) -> float:
  """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
  """
  pipeline.fit(X_train, y_train)
  score: Dict[str, np.ndarray] = cross_validate(
      estimator=pipeline,
      X=X_test,
      y=y_test,
      cv=KFold(n_splits=3, shuffle=True),
      scoring="roc_auc"
  )

  return np.mean(score["test_score"])



def evaluation_4_2(X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray) -> float:
  """ Runs 'evaluate_pipeline' with embedded versions of the input data 
  and returns the accuracy.
  """

  # pipeline: Pipeline = generate_pipeline()
  # 
  # return evaluate_pipeline(
  #     pipeline=pipeline,
  #     X_train=X_train,
  #     y_train=y_train,
  #     X_test=X_test,
  #     y_test=y_test
  # )
  pass


# acc = evaluation_4_2(
#     X_train=load_embedding("X_train_all"),
#     y_train=y_train_all,
#     X_test=load_embedding("X_test"),
#     y_test=y_test
# )



answer_q_4_2 = """
               Your answer 
               """

print("Pipeline:", generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
train_data_generator: ImageDataGenerator = ImageDataGenerator(  # 75.4%
    # width_shift_range=.5,  # 49.1%
    # height_shift_range=.5,  # 50.5%
    # horizontal_flip=True,  # 61.4%
    # vertical_flip=True,
    shear_range=10,
    zoom_range=.2,  # 59%
    rotation_range=10,
    # rescale=1  # overfitting
)"""
last_edit = 'May 26, 2020'