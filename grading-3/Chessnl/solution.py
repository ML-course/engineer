#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/Chessnl'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "Schols, Jeroen"
student_id = "0997216"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = True


# In[3]:


# Uncomment the following line to run in Google Colab
# get_ipython().system('pip install --quiet openml ')


# In[4]:


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

# In[ ]:


# # base_dir = '/content/drive/My Drive/assignment-3-Chessnl' # For Google Colab


# In[8]:


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
# oml.config.cache_directory = os.path.expanduser('/content/drive/My Drive/assignment-3-Chessnl/cache')


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


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[ ]:


from tensorflow.keras import layers
from tensorflow.keras import models

def build_model_1_1():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(768, activation='sigmoid'))
    model.add(layers.Dense(192, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[16]:


run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, train=False, epochs=50, batch_size=256)


# In[17]:


answer_q_1_1 = """
A batch size of 256 is used to reduce overfitting and speedup training.
Two hidden layers as one results in underfitting and three dont yield a significant improvement.
Sigmoid for hidden layers as relu required more layers to properly train.
Softmax for output layer to represent confidence in predictions made.
Categorical crossentropy as loss, as this is a multi-class classifcation problem.

Obtained reasonable accuracy on training (.846) and validation (.780), which is slightly overfitting.
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


# In[ ]:


# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train), rgb2gray(X_val), y_train, y_val


# In[ ]:


# Adjusted model
def build_model_1_2():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32)))
    model.add(layers.Dense(768, activation='sigmoid'))
    model.add(layers.Dense(192, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[21]:


# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, train=False, epochs=50, batch_size=256)


# In[22]:


answer_q_1_2 = """
The same model as before was used, except for the input layer, as the new input has a lower dimensionality.
The training accuracy remained about equal whereas the validation accuracy increased.
Likely, the model overfits less as inputs will no longer be identifiable by a few specific values.
This, as in the new model each input has a higher similarity amongst the images (by a reduction in dimensionality),
which makes it more difficult to rember specific training set images (i.e. overfit).
"""
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[23]:


from keras import regularizers
from tensorflow.keras import optimizers

def build_model_1_3():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32)))
    model.add(layers.Dense(768, kernel_regularizer=regularizers.l1(0.000001), activation='sigmoid'))
    model.add(layers.Dense(192, kernel_regularizer=regularizers.l1(0.000001), activation='sigmoid'))
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.000001), activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.0015), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[24]:


run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, train=False, epochs=25, batch_size=32)


# In[25]:


answer_q_1_3 = """
L1 regularization is added to the hidden layers as these have many edges.
L2 regularization is added to the output layer as to add more regularization.
Previously we used larger batch sizes, 256 downto 32, to avoid overfitting which is now done by regularization.
We increase the learning rate as this model trains slower.
We stop at 25 epochs after which the model has converged.
With this the training accuracy decreased and the validation accuracy increased.
This means the model overfits less.
"""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[ ]:


def build_model_2_1():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l1(0.0001), activation='sigmoid'))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001), activation='sigmoid'))
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.0001), activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[27]:


run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, train=False, epochs=50, batch_size=256)


# In[28]:


answer_q_2_1 = """
The model consists twice of:
Two convolution layers of 64 3x3 relu filters. 64 filters were used as 32 resulted in underfitting and 128 in overfitting.
A 2x2 max-pool layer to reduce the dimensionality of preceding layers.
A dropout layer with .4 dropout probability, being a form of regularization.
No batch normalization was used as this didnt improve the training process.

The result is flattened and passed through:
A dense 128 nodes sigmoid layer with L1 regularization.
A dense 32 nodes sigmoid layer with L2 regularization.
A softmax output layer with L2 regularization.

Regularization was used to avoid overfitting. L1 was only used for the first layer, as this layer had many parameters (204928).
The resulting model achieves a validation accuracy of .931 and doesnt seem overfitted.
"""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[ ]:


# Helper function to check whether the ImageDataGenerator produces proper images
def display_gen_results(gen):
    print("original:")
    plot_images(X_test[:5], y_test[:5])
    print("generated:")
    generated = [gen.random_transform(X_test[i]) for i in range(5)]
    plot_images(generated, y_test[:5])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data

# We ran the ImageDataGenerator with various parameters, the results of these runs are added below

# when generated data is not augmented.
# gen = ImageDataGenerator()
#          loss  accuracy  val_loss  val_accuracy
# min  0.569203  0.185482  0.486569      0.190974
# max  2.382035  0.887500  2.287641      0.912381

# rotation_range: Int. Degree range for random rotations.
# destroys too much information when trained with large rotation angles
# gen = ImageDataGenerator(rotation_range=10, fill_mode='nearest')
#
# rotation_range=15 fails to train
#
# rotation_range=10 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.606337  0.181380  0.491250      0.190974
# max  2.380957  0.865495  2.294832      0.905080
# which is worse then when the data is not augmented
#
# rotation_range=5 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.562052  0.181641  0.468191      0.148360
# max  2.376009  0.878628  2.274635      0.906527
# which is worse then when the data is not augmented

# width_shift_range: Float < 1. Fraction of total width.
# gen = ImageDataGenerator(width_shift_range=.1, fill_mode='nearest')
#
# width_shift_range=.2 fails to train
#
# width_shift_range=.1 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.638460  0.181966  0.507796      0.190974
# max  2.375387  0.863997  2.282804      0.906276
# which is worse then when the data is not augmented

# height_shift_range: Float < 1. Fraction of total height.
# gen = ImageDataGenerator(height_shift_range=.2, fill_mode='nearest')
#
# height_shift_range=.3 fails to train
#
# height_shift_range=.2 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.760010  0.181380  0.569466      0.149682
# max  2.396762  0.833333  2.284032      0.895449
# which is worse then when the data is not augmented
#
# height_shift_range=.1 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.637017  0.182292  0.496172       0.14836
# max  2.343125  0.855409  2.276494       0.90338
# which is worse then when the data is not augmented

# shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees).
# gen = ImageDataGenerator(shear_range=10, fill_mode='nearest')
#
# shear_range=15 fails to train
#
# shear_range=10 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.579438  0.184115  0.505483      0.190974
# max  2.380259  0.880475  2.273360      0.905646
# which is worse then when the data is not augmented
#
# shear_range=5 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.570666  0.182357  0.489449      0.190974
# max  2.345636  0.877799  2.267689      0.905268
# which is worse then when the data is not augmented

# zoom_range: Float. [lower, upper] = [1-zoom_range, 1+zoom_range].
# gen = ImageDataGenerator(zoom_range=.26, fill_mode='nearest')
#
# zoom_range=1 fails to train
#
# zoom_range=.75 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  1.013821  0.177344  0.638806      0.148360
# max  2.395909  0.740820  2.278160      0.868383
# which is worse then when the data is not augmented
#
# zoom_range=.5 produces:
#         loss  accuracy  val_loss  val_accuracy
# min  0.78867  0.178841  0.534608      0.190974
# max  2.37587  0.816953  2.280705      0.900485
# which is worse then when the data is not augmented
#
# zoom_range=.25 produces:
#          loss  accuracy  val_loss  val_accuracy
# min  0.629426  0.184180  0.495804      0.190974
# max  2.373202  0.865234  2.278559      0.908353
# which is worse then when the data is not augmented

# horizontal_flip: Boolean. Randomly flip inputs horizontally.
# gen = ImageDataGenerator(horizontal_flip=True)
#          loss  accuracy  val_loss  val_accuracy
# min  0.717133  0.182422  0.588294      0.190974
# max  2.380940  0.839844  2.287966      0.880657
# which is worse then when the data is not augmented

# vertical_flip: Boolean. Randomly flip inputs vertically.
# gen = ImageDataGenerator(vertical_flip=True)
#          loss  accuracy  val_loss  val_accuracy
# min  0.752271  0.183789  0.642908      0.190974
# max  2.379966  0.835286  2.292966      0.873670
# which is worse then when the data is not augmented

# combinations of augmentations
# gen = ImageDataGenerator(rotation_range=10, width_shift_range=.1, height_shift_range=.2, shear_range=10, zoom_range=.5, fill_mode='nearest')
#          loss  accuracy  val_loss  val_accuracy
# min  1.193136  0.178451  0.618295      0.190974
# max  2.389267  0.680343  2.282429      0.883553
# which is worse then when the data is not augmented
#
# gen = ImageDataGenerator(rotation_range=5, width_shift_range=.05, height_shift_range=.1, shear_range=5, zoom_range=.25, fill_mode='nearest')
#          loss  accuracy  val_loss  val_accuracy
# min  0.874287  0.182161  0.565931      0.148360
# max  2.367892  0.786939  2.283235      0.892617
# which is worse then when the data is not augmented
#
# gen = ImageDataGenerator(rotation_range=2, width_shift_range=.025, height_shift_range=.05, shear_range=2, zoom_range=.125, fill_mode='nearest')
#          loss  accuracy  val_loss  val_accuracy
# min  0.635776  0.180924  0.484432      0.148360
# max  2.377078  0.864380  2.278965      0.910367
# which is worse then when the data is not augmented

# we end up using an ImageDataGenerator that does not use any augmentation, as this method performed best
gen = ImageDataGenerator()
augmented_split = [gen.flow(preprocessed_split[0], preprocessed_split[2], batch_size=256), preprocessed_split[1], preprocessed_split[3]]

# uncomment to display the result of the generator
# note that the non-parameterized ImageDataGenerator does not yield any augmentation
# display_gen_results(gen)


# In[31]:


run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, train=False, generator=True, epochs=50, batch_size=512)


# In[32]:


answer_q_2_2 = """
The following augmentations were tested and converged, however all reduced the accuracy.
- upto 10% width-shifts
- upto 20% ratio height-shifts
- zooming upto 75%
- rotating or shearing upto 10 degrees
- vertical and horizontal flipping
Combinations of these underperformed as well.

Augmenting the training set makes it less representative of the validation set
and should only be used when data is scarce or to reduce overfitting.
As neither holds for our model, we dont use any augmentation.
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

# In[33]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def plot_confusion_matrix():
    # load model and perform predictions
    model = load_model_from_file(base_dir, "model_2_2")
    y_pred = np.argmax(model.predict(rgb2gray(X_test)), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    ax.set_ylabel('True')
    ax.set_yticklabels([(i+1)%10 for i in range(10)])
    ax.set_xlabel('Predicted')
    ax.set_xticklabels([(i+1)%10 for i in range(10)])
    im = ax.imshow(cm)
    fig.set_size_inches(6,6)
    for i in range(100): ax.text(int(i/10), i%10, cm[i%10,int(i/10)], ha="center", va="center", color="w")

    # calculate and return the accuracy score
    return accuracy_score(y_true, y_pred)

test_accuracy_3_1 = plot_confusion_matrix()
print("Model has an accuracy of {} on the test data".format(test_accuracy_3_1))


# In[34]:


def plot_misclassifications():
    # load model and perform predictions
    model = load_model_from_file(base_dir, "model_2_2")
    prediction = model.predict(rgb2gray(X_test))
    y_pred = np.argmax(prediction, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # find all misclassifications for number 2 and display the first 25 with the corresponding prediction
    misclass = [i for i in range(len(y_pred)) if y_pred[i] != y_true[i] and y_true[i] == 1]
    for i in range(min(5,int(len(misclass)/5))):
        plot_images(X_test[misclass[i*5:(i+1)*5]], prediction[misclass[i*5:(i+1)*5]])

plot_misclassifications() 


# In[35]:


answer_q_3_1 = """
The model has a test accuracy of .900.
This was as expected as the validation accuracy of the last few training epochs average around this value as well.
The most common misclassifications are classifying 6 as 5 and 7 as 1.
We consider this a reasonable mistake as (arguably) these numbers share many similarities.

We analyzed 25 misclassifcations of number 2.
Most of these are extremely blurry images.
Furthermore, does it appear that cursive/decorative fonts are more difficult to classify.
"""
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[36]:


images_per_row = 16
layer_names = []

def plot_activation_for_layer(layer_index, activations, use_image_background):
    start = layer_index
    end = layer_index+1

    for layer_name, layer_activation in zip(layer_names[start:end], activations[start:end]):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')                
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
        plt.grid(False)

        if use_image_background: # superimpose image with greyscale input image when use_image_background is true
            repeating_image = np.zeros((32 * n_cols, images_per_row * 32))
            for col in range(n_cols):
                for row in range(images_per_row):
                    repeating_image[col * 32 : (col + 1) * 32, row * 32 : (row + 1) * 32] = rgb2gray(X_test[:1])[0, :, :, 0]
            plt.imshow(repeating_image, cmap='gray', extent=[0, 32*size*images_per_row, 0, 32*size*n_cols])
            plt.imshow(display_grid, aspect='auto', alpha=0.75, cmap='viridis', extent=[0, 32*size*images_per_row, 0, 32*size*n_cols])
        else: # else only display the activations
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

# helper function to reduce code duplication between plot_activations() and plot_3_3()
def plot_activations_helper(use_image_background):
    # define an activation model with the outputs being the output of all layers in model_2_2
    model = load_model_from_file(base_dir, "model_2_2")
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    for layer in model.layers:
        layer_names.append(layer.name)

    # let the activation_model predict for the first image
    img_tensor = rgb2gray(X_test)[0]
    img_tensor = np.expand_dims(img_tensor, axis=0) 
    activations = activation_model.predict(img_tensor)

    # plot the activations for the following layers, being the convolutional layers
    plot_activation_for_layer(0, activations, use_image_background);
    plot_activation_for_layer(1, activations, use_image_background);
    plot_activation_for_layer(4, activations, use_image_background);
    plot_activation_for_layer(5, activations, use_image_background);

def plot_activations():
    plot_activations_helper(False)

# plot_activations()


# In[37]:


answer_q_3_2 = """
The first layer seems to perform edge-detection.
The second layer appears to perform some additional/refined edge-detection.
The third layer seems to detect larger contours and shapes.
For the fourth layer it isn't entirely clear what is being focussed on.

I would consider the first and third layer to learn something useful (edges, contours and shapes).
For the second and especially the fourth layer it is less clear what has been learned and whether this is useful.
"""
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[38]:


def plot_3_3():
    plot_activations_helper(True)

# plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16

def build_model_4_1():
    # define a model with VGG16 base, dense layer and an output layer
    input_tensor = layers.Input(shape=(32,32,3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # freeze everthing in the base model except the last block
    for layer in base_model.layers[:15]:
        layer.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[40]:


run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, train=False, epochs=30, batch_size=1024)


# In[41]:


answer_q_4_1 = """
We build a model with VGG16 trained on ImageNet as base,
followed by a 128 node dense relu layer and a 10 node softmax output layer.

When trained with all VGG16 layer parameters frozen, this resulted in poor validation accuracy (.557).

Hence, we unfroze the final block (block5) of layers and retrained.
To avoid catastrophic forgetting we used a large batch size.
This resulted in a validation accuracy of .808.
However, this model was overfitting, as the training accuracy was .924.
"""
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[ ]:


import pickle
import gzip
from sklearn.pipeline import Pipeline

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
    activation = [model.layers[18].output]
    activation_model = models.Model(inputs=model.input, outputs=activation)

    activations = [activation_model.predict([X_train_all[i:i+1]]).reshape((512)) for i in range(len(X_train_all))]
    store_embedding(activations, 'X_train_all')

    activations = [activation_model.predict([X_test[i:i+1]]).reshape((512)) for i in range(len(X_test))]
    store_embedding(activations, 'X_test')

    activations = [activation_model.predict([X_train[i:i+1]]).reshape((512)) for i in range(len(X_train))]
    store_embedding(activations, 'X_train')

    activations = [activation_model.predict([X_val[i:i+1]]).reshape((512)) for i in range(len(X_val))]
    store_embedding(activations, 'X_val')

# uncomment to recalculate embeddings (warning: slow)
# store_embeddings()


# In[ ]:


from sklearn.preprocessing import StandardScaler, Normalizer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# helper function to simplify creation of a pipeline
def make_pipeline(scaler, classifier):
    if scaler is not None :
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', classifier)
        ])
    return pipeline

# perform a grid search to find the best pipeline to use
def grid_search_pipelines():
    X_train_embedded = np.array(load_embedding('X_train'))
    X_val_embedded = np.array(load_embedding('X_val'))
    with gzip.open(os.path.join(base_dir, 'results'), 'rb') as file_pi:
        results = pickle.load(file_pi)

    for scaler in [None, Normalizer(), StandardScaler(), PowerTransformer()]:
        # Random Forest
        for d in np.logspace(1, 8, base=2, num=8):
            for n in np.linspace(10, 100, num=10):
                classifier = RandomForestClassifier(n_estimators=int(n), max_depth=int(d), random_state=1)
                pipeline = make_pipeline(scaler, classifier)
                results.append(str('RandomForest with scaler = ' + str(scaler).split('(')[0] + ', n_estimators = ' + str(int(n)) + ', max_depth = ' + str(int(d)) + ', accuracy = ' + str(evaluate_pipeline(pipeline, X_train_embedded, np.argmax(y_train, axis=1), X_val_embedded, np.argmax(y_val, axis=1)))))

        # logistic regression
        for c in np.logspace(-6, 3, num=10):
            classifier = LogisticRegression(C = c, random_state=1)
            pipeline = make_pipeline(scaler, classifier)
            results.append(str('LogRegress with scaler = ' + str(scaler).split('(')[0] + ', C = ' + str(c) + ', accuracy = ' + str(evaluate_pipeline(pipeline, X_train_embedded, np.argmax(y_train, axis=1), X_val_embedded, np.argmax(y_val, axis=1)))))

        # SVM with RBF kernel
        for c in np.logspace(-9, 9, num=7):
            for g in np.logspace(-9, 9, num=7):
                classifier = SVC(kernel='rbf', C=c, gamma=g, random_state=1)
                pipeline = make_pipeline(scaler, classifier)
                results.append(str('SVM with scaler = ' + str(scaler).split('(')[0] + ', C = ' + str(c) + ', gamma = ' + str(g) + ', accuracy = ' + str(evaluate_pipeline(pipeline, X_train_embedded, np.argmax(y_train, axis=1), X_val_embedded, np.argmax(y_val, axis=1)))))

    with gzip.open(os.path.join(base_dir, 'results'), 'wb') as file_pi:
        pickle.dump(results, file_pi)

def print_grid_search_pipelines_results():
    with gzip.open(os.path.join(base_dir, 'results'), 'rb') as file_pi:
        results = pickle.load(file_pi)
        for res in results:
            print(res)

# uncomment to redo grid search for optimal pipeline (warning: very slow)
# results = grid_search_pipelines()

# uncomment to print the results of the grid search
# print_grid_search_pipelines_results()


# In[ ]:


from sklearn.metrics import accuracy_score

def generate_pipeline():
    """ Returns an sklearn pipeline.
    """
    return make_pipeline(None, RandomForestClassifier(n_estimators = 300, max_depth = 64, random_state=1))

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """ Evaluates the given pipeline, trained on the given embedded training set and 
        evaluated on the supplied embedded test set. Returns the accuracy score.
    """
    pipeline.fit(X_train, y_train)
    return accuracy_score(y_test, pipeline.predict(X_test))

def evaluation_4_2(X_train, y_train, X_test, y_test):
    """ Runs 'evaluate_pipeline' with embedded versions of the input data 
        and returns the accuracy.
    """
    X_train_all_embedded = load_embedding('X_train_all')
    X_test_embedded = load_embedding('X_test')
    pipeline = generate_pipeline()
    return evaluate_pipeline(pipeline, X_train_all_embedded, np.argmax(y_train_all, axis=1), X_test_embedded, np.argmax(y_test, axis=1))

# uncomment to calculate the accuracy of the chosen pipeline
# when dependent code has not been change, this should return  0.8243025480914493
# evaluation_4_2(X_train_all, y_train_all, X_test, y_test)


# In[45]:


answer_q_4_2 = """
We performed a grid search for Random Forest, Logistic Regression and SVM with RBF as classifiers
and Normalizer, StandardScaler, and PowerTransformer as preprocessers.

The best validation accuracies were:
0.816 for SVM with RBF when normalized, with C = 1e6 and gamma = 1e-6.
0.823 for Logistic regression when power-transformed, with C = 0.
0.825 for Random Forest without scaling, with 300 estimators and max_depth = 64.

The latter was chosen which gave a 0.824 test accuracy.
"""
print("Pipeline:", generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
# Helper function to check whether the ImageDataGenerator produces proper images
def display_gen_results(gen):"""
last_edit = 'May 26, 2020'