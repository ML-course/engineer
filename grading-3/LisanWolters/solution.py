#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/LisanWolters'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fill in your name using the format below and student ID number
your_name = "Wolters, L.J."
student_id = "0959459"


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
# base_dir = 'C:/Users/s153944/Documents/uni jaar 5/2IMM15 Web information retrieval and data mining/assignment 3/assignment-3-LisanWolters'


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
# import os
# # oml.config.cache_directory = os.path.expanduser('/content/cache')


# In[8]:


# Download Streetview data. Takes a while (several minutes), and quite a bit of
# memory when it needs to download. After caching it loads faster.
SVHN = oml.datasets.get_dataset(41081)
X, y, _, _ = SVHN.get_data(dataset_format='array',
    target=SVHN.default_target_attribute)


# Reshape, sample and split the data

# In[9]:


from tensorflow.keras.utils import to_categorical

Xr = X.reshape((len(X),32,32,3))
Xr = Xr / 255.
yr = to_categorical(y)


# In[10]:


# DO NOT EDIT. DO NOT OVERWRITE THESE VARIABLES.
from sklearn.model_selection import train_test_split
# We do an 80-20 split for the training and test set, and then again a 80-20 split into training and validation data
X_train_all, X_test, y_train_all, y_test = train_test_split(Xr,yr, stratify=yr, train_size=0.8, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_all,y_train_all, stratify=y_train_all, train_size=0.8, random_state=1)
evaluation_split = X_train, X_val, y_train, y_val


# Check the formatting - and what the data looks like

# In[14]:


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

# In[11]:


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
    model.save(os.path.join(base_dir, name+extension))

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


# In[16]:


# # Toy usage example
# # Remove before submission
# from tensorflow.keras import models
# from tensorflow.keras import layers 

# def build_toy_model():
#     model = models.Sequential()
#     model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
#     model.add(layers.Dense(10, activation='relu'))
#     model.add(layers.Dense(10, activation='softmax'))
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # First build and store
# run_evaluation("toy_example", build_toy_model, evaluation_split, base_dir, 
#                train=True, epochs=3, batch_size=32)


# In[17]:


# # Toy usage example
# # Remove before submission
# # With train=False: load from file and report the same results without rerunning
# run_evaluation("toy_example", build_toy_model, evaluation_split, base_dir, 
#                train=False)


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[18]:


from tensorflow.keras import models
from tensorflow.keras import layers 

def build_model_1_1():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) #softmax
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model  #rmsprop

run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=False, epochs=45, batch_size=32)
answer_q_1_1 = """I ended with an softmax layer, as this would give a probabilty for each possible option, which sum to 1. I have a sigmoid layer in between as it is in the range 0 to 1 which can push it to a probability already. As optimizer I tried adamax and rmspop. The accurancy is 0.83, which is a reasonable performance. Accuracy and validation loss have stabilized for multiple epochs. Therefore I am not underfitting. And I am also not overfitting as the validation loss is not increasing. """
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

# In[14]:


# Luminance-preserving RGB to greyscale conversion
def rgb2gray(X):
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)


# In[34]:


# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val

# Adjusted model
def build_model_1_2():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) #softmax
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
  

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=False, epochs=45, batch_size=32)
answer_q_1_2 = """For assignment 1.2 I used the predefined rgb2gray method to preprocess the data. For the training data set this reduced the minimal loss a little bit. For the validating set the loss and accurancy are comparable to the values without preprocessing. Next to that is the curve in the lines to show the validation and the loss are behaving the same in model 1.1 and 1.2, therefore the model is performing the same."""
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[21]:


from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

def build_model_1_3():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='sigmoid'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) 
    model.compile(optimizer=optimizers.Adamax(lr=0.005),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    return model


run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=False, epochs=45, batch_size=64)
answer_q_1_3 = """I have tried to add dropout layers after layer 2 and 4, with different dropout rates, but than the accurancy decreased. The batchsize of 64 worked best, together with a learning rate of 0.005 for adamax. Batchnormalization and l2 regularization had a negative influence on the loss. Not keeping the pyramid shape in the dense layers increased the loss. In the previous assignment my model was not really overfitting, therefore after regularization it became less accurate."""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[22]:


def build_model_2_1():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',  input_shape=(32, 32, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
  
run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=False, epochs=10, batch_size=32)
answer_q_2_1 = """The optimizer rmsprop and adamax gave the same results. At the end I have a dense layer with 32 nodes. This dense layer is to have a layer which looks at the global data instead of more local data, which is the case for convolutional layers. The dropout increases each time it is used after a convolutional layer, the dropout after the dense layer is again a smaller dropout. The dropout is used to break up accidental not significant learned patterns. Max pooling is used to reduce the dimension of the data. This highlights the most present feature, which is in practice mostly the best option for image classification. The batch normalization layers adjust and scales the amount by what the hidden unit values shift around, it normalizes the data. We have a fit which is neither over or underfitting."""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[32]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
       rotation_range=5,        # Rotate image up to 5 degrees
       width_shift_range=0.05,    # Shift image left-right up to 5% of image width
       height_shift_range=0.05,   # Shift image up-down up to 5% of image height
       shear_range=0.1,          # Shear (slant) the image up to 0.1 degrees
       #zoom_range=0.1,           # Zoom in up to 10%
      # horizontal_flip=False,    # Horizontally flip the image
      fill_mode='nearest')

aug_data = datagen.flow(preprocessed_split[0],preprocessed_split[2], batch_size = 32768)
augXtrain, augYtrain = aug_data.next()
augmented_split = augXtrain, preprocessed_split[1], augYtrain, preprocessed_split[3]

# plot_images(augXtrain[0:5, :, :, 0], augYtrain[0:5], grayscale = True)
# plot_images(augXtrain[5:10, :, :, 0], augYtrain[5:10], grayscale = True)
# plot_images(augXtrain[10:15, :, :, 0], augYtrain[10:15], grayscale = True)
run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, 
               train=False, epochs=15, batch_size=None)
answer_q_2_2 = """I did not manage to improve my previous model with data augmentation. The accuracy and loss is slightly more constant through the epochs. Other augmentations give the same performance, or causes that the model did not train. No training can be due to the fact that the picture gets too blurry and edges are less distinguishable. The current augmentation has a small rotation, shear and shift. I have set a higher batch size to increase accuracy as we need to distinguish multiple classes . """
print("Answer is {} characters long".format(len(answer_q_2_2)))


# ## Part 3. Model interpretation (10 points)
# ### Question 3.1: Interpreting misclassifications (2 points)
# Study which errors are still made by your last model (model_2_2) by evaluating it on the test data. You do not need to retrain the model.
# * What is the accuracy of model_2_2 on the test data? Store this in 'test_accuracy_3_1'.
# * Plot the confusion matrix in 'plot_confusion_matrix' and discuss which classes are often confused.
# * Visualize the misclassifications in more depth by focusing on a single
# class (e.g. the number '2') and analyse which kinds of mistakes are made for that class. For instance, are the errors related to the background, noisiness, etc.? Implement the visualization in 'plot_misclassifications'.
# * Summarize your findings in 'answer_q_3_1'

# In[18]:


from sklearn.metrics import confusion_matrix

test_accuracy_3_1 = 0.9201
def plot_confusion_matrix():
  model = load_model_from_file(base_dir, "model_2_2")
  #model.evaluate(rgb2gray(X_test), y_test)
  
  y_pred = model.predict(rgb2gray(X_test))
  #misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]

  cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
  fig, ax = plt.subplots()
  im = ax.imshow(cm)
  ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
  ax.set_xticklabels([1,2,3,4,5,6,7,8,9,0])
  ax.set_yticklabels([1,2,3,4,5,6,7,8,9,0])
  ax.set_ylabel('True')
  ax.set_xlabel('Predicted')
  for i in range(100):
      ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")
      #ax.text((int(i/10)+1)%10,i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")

def plot_misclassifications():
  model = load_model_from_file(base_dir, "model_2_2")
  
  y_pred = model.predict(rgb2gray(X_test))
  misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]

  # Visualize the (first five) misclassifications, together with the predicted and actual class
  #X_test_gray = rgb2gray(X_test)
  fig, axes = plt.subplots(5, 5,  figsize=(40, 20))

  toshow = [misclassified_samples[3], misclassified_samples[31],  misclassified_samples[51], misclassified_samples[77], misclassified_samples[86], misclassified_samples[96], misclassified_samples[109],  misclassified_samples[125], misclassified_samples[129], misclassified_samples[133], misclassified_samples[142], misclassified_samples[151],  misclassified_samples[158], misclassified_samples[161], misclassified_samples[162], misclassified_samples[172], misclassified_samples[173],  misclassified_samples[181], misclassified_samples[185], misclassified_samples[197], misclassified_samples[199], misclassified_samples[210],  misclassified_samples[220], misclassified_samples[239], misclassified_samples[249]]
  for nr, i in enumerate(toshow):
      axes[int(nr/5), int(nr%5)].imshow(X_test[i])
      axes[int(nr/5), int(nr%5)].set_xlabel("Predicted: %s,\n Actual : %s" % ((np.argmax(y_pred[i])+1)%10,(np.argmax(y_test[i])+1)%10))
      axes[int(nr/5), int(nr%5)].set_xticks(()), axes[int(nr/5), int(nr%5)].set_yticks(()) 
  plt.show(); 

  # for i in range (0,500):
  #   if((np.argmax(y_test[misclassified_samples[i]])+1)%10 == 9):
  #     print(i , ":", (np.argmax(y_test[misclassified_samples[i]])+1)%10)

answer_q_3_1 = """The mistakes which occur the most are that 5 is expected when it was a 3 or 1 is predicted when it was a 7. These mistakes make sense, as these are also mixed up by humans more often. I focussed on the cases where the real number was a 9, but it was not expected. In most of these cases a 0 or 8 was predicted. Most of the times that a 0 was expected the 9 did not have a clear line next to the round part of the 9 or there was no clear horizontal line in the middle. At cases where an 8 was predicted, the end part of the bottum line has a vertical part which is pointing to the top attached. On different picture there are visible multiple numbers, which makes it difficult to recognize only one number. Most of the picture are also blurry and therefore difficult to assign the correct number."""
print("Answer is {} characters long".format(len(answer_q_3_1)))
plot_confusion_matrix()
plot_misclassifications()


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[25]:


def plot_activations():
  np.seterr(divide='ignore', invalid='ignore')
  model = load_model_from_file(base_dir, "model_2_2")
  #model.summary()
  images_per_row = 16

  img_tensor = rgb2gray(X_test)[0]
  img_tensor = np.expand_dims(img_tensor, axis=0) 

  layer_outputs = [layer.output for layer in model.layers[:]]  
  # Creates a model that will return these outputs, given the model input:
  activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
  activations = activation_model.predict(img_tensor)

  layer_names = []
  for layer in model.layers[:]: 
      layer_names.append(layer.name)

  # Now let's display our feature maps
  for i in  [0,2,6,8,12,14]: 
    layer_index = i
    start = i
    end = i+1
    for layer_name, layer_activation in zip(layer_names[start:end], activations[start:end]):
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
        plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()    

answer_q_3_2 = """In layer 1 and 3 the model really focusses on the outlines of the number. From layer 7 onwards there are visualized more abstract patterns. In layer 7 and 9, you can see clear the round top and bottom of the 8 on some of the picture, while on other pictures the two dots which are in the middle of the 8 are highlighted. In layer 9 in for example the picture on the left bottom the 8 and 5 are fairly clearly highlighted. In later layers the visualization become to abstract for me to interpret."""
print("Answer is {} characters long".format(len(answer_q_3_2)))
# plot_activations()


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[15]:


# get_ipython().system('pip install opencv-python')
import cv2 

tf.compat.v1.disable_eager_execution()

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K

def plot_3_3():
  #!pip install opencv-python
  K.clear_session()

  model = load_model_from_file(base_dir, "model_2_2") 
  X_test_gray = rgb2gray(X_test)
  number = 0
  img = X_test_gray[number] 

  # `x` is a float32 Numpy array of shape (32, 32, 3)
  x = image.img_to_array(img)
  # We add a dimension to transform our array into a "batch"
  # of size (1, 32, 32, 3)
  x = np.expand_dims(x, axis=0)

  x_output = model.output[:, 7] #number -1 

  # The is the output feature map of the 'conv2d_41' layer,
  last_conv_layer = model.get_layer('conv2d_17') 

  #tf.compat.v1.disable_eager_execution()

  # This is the gradient of the first grayscaled x_test element with regard to the output feature map of `conv2d_41`
  grads = K.gradients(x_output, last_conv_layer.output)[0]

  # This is a vector, of shape (132,) where each entry is the mean intensity of the gradient over a specific feature map channel
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  #print(pooled_grads.shape)

  # This function allows us to access the values of the quantities we just defined:
  # `pooled_grads` and the output feature map of `conv2d_41`, given a sample image
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  # These are the values of these two quantities, as Numpy arrays, given our sample image of the number 8
  pooled_grads_value, conv_layer_output_value = iterate([x])

  # We multiply each channel in the feature map array by "how important this channel is" with regard to the number 8 class
  for i in range(132): 
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  heatmap = np.mean(conv_layer_output_value, axis=-1)

  # plot the heatmap
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  # We resize the heatmap to have the same size as the original image
  heatmap = cv2.resize(heatmap, (32, 32))
  fig, axs = plt.subplots(1, 3, figsize=(10, 5))
  axs[0].matshow(heatmap, cmap = 'gray')
  axs[0].set_title('Grayscaled activation map')
  axs[0].axis('off')

  # We convert the heatmap to RGB
  heatmap = np.uint8(255 * heatmap)

  # We apply the heatmap to the original image
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 

  # 0.35 here is a heatmap intensity factor
  superimposed_img = img *255 + heatmap * 0.35
  cv2.imwrite(base_dir +'/test.jpg', superimposed_img)
  img = cv2.imread(base_dir +'/test.jpg') 

  RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  axs[1].imshow((X_test_gray[number, :, :, 0]* 255).astype(np.uint8), cmap='gray') 
  axs[1].set_title('Grayscaled image')
  axs[1].axis('off')
  axs[2].imshow(RGB_image)
  # axs[2].imshow((superimposed_img * 255).astype(np.uint8))
  axs[2].set_title('Imposed activation map')
  axs[2].axis('off')
 
  pass

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[27]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers

def build_model_4_1():
    model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32,32,3), pooling=None, classes=1000)
    conv_base = model
    #model.summary()
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='sigmoid'))
#     model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) #softmax
    # conv_base.trainable = False
    for layer in conv_base.layers:
      if layer.name == 'block5_conv3' or layer.name == 'block5_conv2' or layer.name == 'block5_conv11': 
          layer.trainable = True
      else:
          layer.trainable = False
    model.compile(optimizer='adamax',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir,
               train=False, epochs=10, batch_size=64)
answer_q_4_1 = """When I freeze the convolutional base the accuracy will get around 45 percent, even with multiple dense layers, which is low. This can be due to the fact that the VGG16 is trained on different data for which different features are important. It works better to unfreeze the last convolutional layers, as the features which are important for number recognitioncan get a higher weight. I have choosen to unfreeze block5. The model start overfitting after a few epochs, but the accuracy is reasonable."""
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,  GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn import pipeline

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
  #model.summary()
  vgg16layer = model.get_layer(name='vgg16')
  # vgg16layer.summary()
  XtrainOutput = vgg16layer.predict(X_train)
  XtestOutput = vgg16layer.predict(X_test) 
  store_embedding(XtrainOutput, 'XTraining')
  store_embedding(XtestOutput, 'XTesting')
  pass

def generate_pipeline():
  """ Returns an sklearn pipeline.
  """
  pipe = make_pipeline(MinMaxScaler(), LinearSVC(C=1)) 
  return(pipe)


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
  """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
  """
  # param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100]} 
  
  # grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
  # grid.fit(X_train, y_train)
  # print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
  # print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
  # print("Best parameters: {}".format(grid.best_params_)) 
    
  pipeline.fit(X_train, y_train)
  # trainscore = pipeline.score(X_train, y_train)
  # # print('trainscore:', trainscore)
  # testscore = pipeline.score(X_test, y_test)
  # # print('testscore:', testscore)

  cvs = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
  trainscore = cross_val_score(pipeline, X_train, y_train, cv=cvs)
  # print(trainscore)
  # print(np.mean(trainscore))
  testscore = cross_val_score(pipeline, X_test, y_test, cv=cvs)
  # print(testscore)
  # print(np.mean(testscore))

  accuracy = {
      "train accuracy": np.mean(trainscore),
      "test accuracy": np.mean(testscore)
  }
  return(accuracy)
  

def evaluation_4_2(X_train, y_train, X_test, y_test):
  """ Runs 'evaluate_pipeline' with embedded versions of the input data 
  and returns the accuracy.
  """
  xTrain = load_embedding('XTraining')
  xTest = load_embedding('XTesting')
  # print(xTrain.shape)
  # print(xTest.shape)
  xTrain = xTrain.reshape((63544,-1))
  xTest = xTest.reshape((19858,-1))
  # print(xTrain.shape)
  # print(xTest.shape)
  # print(y_train.shape)
  y_trainn = [0] * len(y_train)
  for i in range(len(y_train)):
    y_trainn[i] = ((np.argmax(y_train[i])+1)%10)
  
  y_testt = [0] * len(y_test)
  for i in range(len(y_test)):
    y_testt[i] = ((np.argmax(y_test[i])+1)%10)
  
  pipeline = generate_pipeline()
  accuracy = evaluate_pipeline(pipeline, xTrain, y_trainn, xTest, y_testt)
  # print(accuracy)
  return(accuracy)

answer_q_4_2 = """I tried pipelines with MinMaxscalar, Normalizer and Standardscalar and classifiers LinearSVC, LR and RFC.  When cross_val_score was used, all methods resulted in almost the same accuracy. When the score method of a pipeline was used, the training accuracy with the RFC was 99%, but the test accuracy was just as now 78%. Grid search showed that C=1 is the best option for LinearSVC. This model cannot beat my best model, as my best model has an accuracy of 95% and this only has an accuracy of 78%."""
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))
# store_embeddings()
# evaluation_4_2(X_train, y_train, X_test, y_test)


dg_code= """
datagen = ImageDataGenerator(
       rotation_range=5,        # Rotate image up to 5 degrees
       width_shift_range=0.05,    # Shift image left-right up to 5% of image width
       height_shift_range=0.05,   # Shift image up-down up to 5% of image height
       shear_range=0.1,          # Shear (slant) the image up to 0.1 degrees"""
last_edit = 'May 26, 2020'