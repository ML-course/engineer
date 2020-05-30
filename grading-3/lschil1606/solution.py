#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/lschil1606'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fill in your name using the format below and student ID number
your_name = "Schilders, Lars"
student_id = "0908729"


# In[2]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = True


# In[3]:


# Uncomment the following line to run in Google Colab
# !pip install --quiet openml 


# In[4]:


# Uncomment the following line to run in Google Colab
# %tensorflow_version 2.x
import tensorflow as tf
# # tf.config.experimental.list_physical_devices('GPU') # Check whether GPUs are available


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import openml as oml
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[18]:


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


# # base_dir = '/content/drive/My Drive/Colab Notebooks/assignment-3-lschil1606' # For Google Colab
# base_dir = './'


# In[20]:


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





# In[ ]:


# Use OpenML caching in Colab
# On your local machine, it will store data in a hidden folder '~/.openml'
# import os
# # oml.config.cache_directory = os.path.expanduser('/content/cache')


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

# In[ ]:





# In[25]:


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


from tensorflow.keras import models
from tensorflow.keras import layers 


# ## Part 1. Dense networks (10 points)
# 
# ### Question 1.1: Baseline model (4 points)
# - Build a dense network (with only dense layers) of at least 3 layers that is shaped like a pyramid: The first layer must have many nodes, and every subsequent layer must have increasingly fewer nodes, e.g. half as many. Implement a function 'build_model_1_1' that returns this model.
# - You can explore different settings, but don't use any preprocessing or regularization yet. You should be able to achieve at least 70% accuracy, but more is of course better. Unless otherwise stated, you can use accuracy as the evaluation metric in all questions.
# * Add a small description of your design choices (max. 500 characters) in 'answer_q_1_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - The name of the model should be 'model_1_1'. Evaluate it using the 'run_evaluation' function. For this question, you should not use more than 50 epochs.

# In[28]:


def build_model_1_1():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=False, epochs=50, batch_size=32)


# In[ ]:


answer_q_1_1 = """
               I start with 1024 nodes and halve that amount in the next two layers. More layers were not better.
               I used relu as activation over sigmoid, which resulted in no learning.
               I used 10 output nodes with
               softmax to get a probability for every class prediction.
               SGD worked better than rmsprop and Adadelta as optimizer. Categorical cross-entropy 
               is used as loss function to get multi-class predictions.
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


X_train_preprocess = rgb2gray(X_train)
X_val_preprocess = rgb2gray(X_val)
X_test_preprocess = rgb2gray(X_test)


# In[ ]:


# Replace with the preprocessed data
preprocessed_split = X_train_preprocess, X_val_preprocess, y_train, y_val


# In[32]:


# Adjusted model
def build_model_1_2():
    model = models.Sequential()
    model.add(layers.Reshape((32*32,), input_shape=(32,32,1)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=False, epochs=50, batch_size=32)


# In[33]:


answer_q_1_2 = """
               By converting X_train and X_val to grey scale the model only slightly improved. 
               It did improve run time as the dimension of the data is reduced by having only 1 color channel instead of 3.
               The loss curve is slightly improved, as the model now regularizes better because of the dimensionality reduction.
               """
print("Answer is {} characters long".format(len(answer_q_1_2)))


# In[ ]:





# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[ ]:


from tensorflow.keras import optimizers, regularizers


# In[35]:


def build_model_1_3():
    model = models.Sequential()
    model.add(layers.Reshape((32*32,), input_shape=(32,32,1)))
    model.add(layers.Dense(1024,  activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512,   activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256,   activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(learning_rate=0.05),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=True, epochs=50, batch_size=64)


# In[36]:


answer_q_1_3 = """
               I varied the learning rate and the batch size, where batchsize=64 and 
               learning rate=0.05 sped up the training and worked as well as the default values. 
               L1 and L2 regularizers did not improve the model.
               Constant dropout after every layer regularized better. Batch Normalization resulted in no improvement.
               The model is regularized well, as the accuracy curves almost completely overlap.
               """
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[37]:


def build_model_2_1():
    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(32,32,1)))
    model.add(layers.Conv2D(32,(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128,(3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128,(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256,(3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256,(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer = 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=False, epochs=12, batch_size=32)


# In[38]:


answer_q_2_1 = """
               I use 4 blocks with 2 convolutional layers, 2x2 MaxPooling and 20% Dropout each, 
               starting with 32 3x3 filters and double the filters every block, 
               because 2x2 pooling halves the resolution in every dimension. Starting with 64 filters only resulted in a longer computation time.
               Increasing dropout every time had worse results.
               Zero padding is used to not lose information at the border of the pictures.
               As output layer a fully connected layer with 128 nodes and again softmax activated with 10 nodes.
               RMSProp was a much better optimizer compared to SGD, Adadelta and Adam. 
               Train and test accuracy almost completely overlap, implying good regularization
               """
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[40]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range = 5,
    rotation_range = 5,
    zoom_range = [0.95,1.05],
    horizontal_flip = False,
    vertical_flip = False)

batch_size = 32

it_train = train_datagen.flow(X_train_preprocess, y=y_train, batch_size=batch_size)

augmented_split = (it_train, X_val_preprocess, y_val)

steps_per_epoch = X_train_preprocess.shape[0]/batch_size

run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, generator=True,
               train=True, steps_per_epoch = steps_per_epoch, epochs=20, batch_size=batch_size)


# In[41]:


answer_q_2_2 = """
               Horizontal and vertical flip would not make sense, as mirrored numbers could look like other numbers (6 and 9 for example).
               The images were nicely centered already, such that larger values of height and width shift do not work well.
               The model did not improve significantly (only 0.4%)
               Steps per epoch is put on number of samples divided by the batchsize. 
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


# In[ ]:





# In[ ]:





# In[43]:


import random
model = load_model_from_file(base_dir, "model_2_2")
y_pred = model.predict(X_test_preprocess)
test_accuracy_3_1 = model.evaluate(X_test_preprocess, y_test)[1]
print("Test accuracy: " +str(test_accuracy_3_1))
def plot_confusion_matrix():
    cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10), rotation=45, ha="right")
    ax.set_yticklabels(np.arange(10))
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    for i in range(100):
        ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")

def plot_misclassifications():
    class_label = 7
    class_samples = np.nonzero((np.argmax(y_test, axis=1)+1)%10 == class_label)[0]
    misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]
    misclassified_samples_class = np.array([x for x in class_samples if x in misclassified_samples])
    nrRows = 5
    nrCols = 10
    plt.figure(figsize=(nrCols*2, nrRows*3))
    for nr, i in enumerate(random.choices(misclassified_samples_class,k=nrCols*nrRows)):
        plt.subplot(nrRows,nrCols,nr+1)
        plt.imshow(X_test[i], cmap='gray')
        plt.xlabel("Predicted: %s,\n Actual : %s" % ((np.argmax(y_pred[i])+1)%10, (np.argmax(y_test[i])+1)%10 ))
        plt.xticks(()), plt.yticks(())

    plt.show();


# In[44]:


#plot_misclassifications()


# In[ ]:





# In[45]:


#plot_confusion_matrix()


# In[46]:


answer_q_3_1 = """
               0 is the prediction that is wrong most times, while 4, 5 and 7 are labels that are most often wrongly predicted.
               The most common confusion is label 4 while predicting a 2. Looking at the misclassifications,
               some labels of the test set are wrong, while the model predicted them correct. This means that 
               the real accuracy could be even higher. Images with skewed or multiple numbers seem harder to predict
               """
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[ ]:





# In[47]:


from tensorflow.keras import models

def plot_activations():
    img_tensor = X_test_preprocess[0]
    img_tensor = np.expand_dims(img_tensor, axis=0) 

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    images_per_row = 16

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
    conv_layers = np.nonzero([l.startswith('conv') for l in layer_names])[0]
    
    for layer_index in conv_layers:

        start = layer_index
        end = layer_index+1
        # Now let's display our feature maps
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



#plot_activations()


# In[48]:


answer_q_3_2 = """
               In the first two layers you can see that some filters pay attention to edges or round shapes or some only to background.
               In the last layer almost every filter responds to a different shape, which indicates that the model is learning something.
               Some filters have very low activation values which implies that these filters do not respond well to label 8. 
               """
print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[ ]:


from tensorflow.keras import backend as K


# In[50]:


import cv2

def plot_3_3():
    tf.compat.v1.disable_eager_execution()
    model = load_model_from_file(base_dir,'model_2_2')
    X_output = model.output[:,0]
    img_tensor = X_test_preprocess[0]
    img_tensor = np.expand_dims(img_tensor, axis=0)

    #retrieve last convolutional layer
    last_conv_layer = model.layers[np.nonzero([lay.name.startswith('conv') for lay in model.layers])[0].max()]

    grads = K.gradients(X_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0,1,2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

    for i in range(32):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)


    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap = np.uint8(255 * heatmap)
    print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    img_tensor_color = (255 * X_test[0]).astype(int)

    superimposed_img = heatmap * 0.4 + img_tensor_color

    plt.subplots(1,2)
    plt.subplot(121)
    plt.imshow(img_tensor_color)
    plt.title('Original (colored) image')
    plt.subplot(122)
    plt.title('Class activation heatmap')
    plt.imshow(heatmap)

    plt.figure(figsize=(6,6))
    plt.title("Class activation map superimposed on image")
    plt.imshow(superimposed_img.astype(int))
    plt.colorbar()

#plot_3_3()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[ ]:


from tensorflow.keras.applications import vgg16


# In[56]:


def build_model_4_1():
    model_vgg16 = models.Sequential()
    pretrained_model = vgg16.VGG16(include_top=False, input_shape=(32,32,3))

    for idx, layer in enumerate(pretrained_model.layers):
        if layer.name.startswith('block5_conv') or layer.name.startswith('block4_conv'):
            layer.trainable=True
        else:
            layer.trainable = False
        model_vgg16.add(layer)
    
    model_vgg16.add(layers.BatchNormalization())
    model_vgg16.add(layers.Dropout(0.2))
    model_vgg16.add(layers.Flatten())
    model_vgg16.add(layers.Dense(512, kernel_regularizer = regularizers.l1(), activation='relu'))
    model_vgg16.add(layers.BatchNormalization())
    model_vgg16.add(layers.Dropout(0.2))
    model_vgg16.add(layers.Dense(10, activation='softmax'))
    model_vgg16.compile(optimizer = optimizers.RMSprop(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    for layer in model_vgg16.layers:
        print(layer, layer.trainable)

    return model_vgg16

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
               train=True, epochs=20, batch_size=32)


# In[ ]:





# In[53]:


answer_q_4_1 = """
                Unfreezing last two blocks worked a lot better than freezing the convolutional base. 
                I used a small learning rate of 1e-4, 
                as the model should not overshoot the minimum the pretrained model was in.
                The loss in the first epoch is very high, implying that the pretrained model is bad, however the accuracy is good.
                The model still overfits, despite the dropout, batch normalization and L1 regularizer.
               """
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[ ]:


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[ ]:


import pickle
import gzip

def store_embedding(X, name):  
    with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
        pickle.dump(X, file_pi)

def load_embedding(name):
    with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
        return pickle.load(file_pi)

def store_embeddings():
    """ Stores all necessary embeddings to file
    """
    print("Loading model from file")
    model_4_1 = load_model_from_file(base_dir, "model_4_1")
    print("Loaded model from file.")
    conv_base = models.Sequential()
    for layer in model_4_1.layers[:-7]:
        conv_base.add(layer)
    
    conv_base.add(layers.Flatten())
    conv_base.compile(optimizer = 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Generating X_test embedding...")
    emb_X_test = conv_base.predict(X_test)
    print("Generating X_train embedding...")
    emb_X_train = conv_base.predict(X_train)

    print("Store X_test embedding...")
    store_embedding(emb_X_test, 'X_test')
    print("Store X_train embedding...")
    store_embedding(emb_X_train, 'X_train')


def generate_pipeline():
    """ Returns an sklearn pipeline.
    """
    return make_pipeline(StandardScaler(), SVC())

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
    """
    pipeline.fit(X_train, y_train)
    return pipeline.score(X_test, y_test)


def evaluation_4_2(X_train, y_train, X_test, y_test):
    """ Runs 'evaluate_pipeline' with embedded versions of the input data 
    and returns the accuracy.
    """
    X_train_emb = load_embedding('X_train')
    X_test_emb = load_embedding('X_test')
    pl = generate_pipeline()
    return evaluate_pipeline(pl, X_train_emb, np.argmax(y_train,axis=1), X_test_emb, np.argmax(y_test,axis=1))



# In[60]:


#evaluation_4_2(X_train, y_train, X_test, y_test)


# In[59]:


answer_q_4_2 = """
               I tried a StandardScaler and MinMaxScaler as preprocessors and 
               RandomForest and SVC as classifiers. All combinations gave approximately the same results.
               I am off of my best model so far by 2%. 
               """
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





dg_code= """
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range = 5,
    rotation_range = 5,
    zoom_range = [0.95,1.05],
    horizontal_flip = False,
    vertical_flip = False)"""
last_edit = 'May 26, 2020'