#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/manonww'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "Wientjes, Manon"
student_id = "1398903"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = False


# In[ ]:


# Uncomment the following line to run in Google Colab
# !pip install --quiet openml 


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
import sklearn


# In[ ]:


from packaging import version
#import sklearn
#import tensorflow
sklearn_version = sklearn.__version__
tensorflow_version = tf.__version__
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


# #base_dir = '/content/drive/My Drive/TestAssignment' # For Google Colab
# base_dir = './'


# In[ ]:


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

# In[ ]:


# Use OpenML caching in Colab
# On your local machine, it will store data in a hidden folder '~/.openml'
#import os
# #oml.config.cache_directory = os.path.expanduser('/content/cache')


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
#import numpy as np
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

# In[ ]:


from tensorflow.keras import models
from tensorflow.keras import layers 

def build_model_1_1():
    model = models.Sequential()
    model.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, 
               train=False, epochs=50, batch_size=32)
answer_q_1_1 = """I increased the model's complexity until the validation loss didn't decrease anymore. I always
made sure that the layers aren't too narrow to not lose information.
Besides, I choose an optimizer. Most optimizers were performing equally well.
However, SGD converges the quickest. Therefore, SGD was chosen as the optimizer.
The model starts to overfit around 20 epochs. Hence, regularization should be considered."""
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


def rgb2gray(X):
    return np.expand_dims(np.dot(X, [0.2990, 0.5870, 0.1140]), axis=3)

# Replace with the preprocessed data
preprocessed_split = rgb2gray(X_train).astype(np.float32), rgb2gray(X_val).astype(np.float32), y_train, y_val


# In[ ]:


# Adjusted model
def build_model_1_2():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, 
               train=False, epochs=50, batch_size=32)
answer_q_1_2 = """For preprocessing I converted the images to grayscale and float32.
The model is better since it needs less epochs to reach the same accuracy. 
This is caused by the fact that we need to learn less hyperparameters
and no information is lost by converting it to grayscale. Since color is
irrelevant for our images.
Converting it to float32 requires less memory. Hence, a lower training time.
Since the values are in range 0-1, no more preprocessing is needed.
               """
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[ ]:


from tensorflow.keras import optimizers

def build_model_1_3():
    model = models.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, 
               train=False, epochs=50, batch_size=32)
answer_q_1_3 = """I optimized the most interesting hyper parameters by doing grid search. Since these parameters are all related.
The values were chosen as typical hyper parameter values. From this search the following values were optimal:
learning_rate=0.01, momentum=0.3, batch_size=32 and epochs=50. After trying out different (combinations) of
regularization methods, dropout at rate 0.05 performs best. The same accuracy is reached, but the model isn't overfitting anymore.
"""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# In[ ]:


# # I run grid search only on the train dataset, since otherwise validation data will leak to the train dataset. This
# # will make further decisions incorrect
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

# def make_model(learning_rate=0.01, momentum=0.0):
#     model = models.Sequential()
#     model.add(layers.Reshape((1024,), input_shape=(32,32,1)))
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(32, activation='relu'))
#     model.add(layers.Dense(10, activation='softmax'))

#     model.compile(optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# clf = KerasClassifier(make_model)
# param_grid = {'epochs': [20,30,50],  
#               'batch_size': [32,64],
#               'learning_rate': [0.001, 0.01, 0.1], 
#               'momentum': [0.0, 0.3, 0.5, 0.9],
#               'verbose' : [0]}
# grid = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=3, return_train_score=True)
# y_grid = (np.argmax(preprocessed_split[2], axis=1)+1)%10
# grid.fit(preprocessed_split[0], y_grid)


# In[ ]:


# res = pd.DataFrame(grid.cv_results_)
# res.pivot_table(index=["param_epochs", "param_batch_size", "param_learning_rate", "param_momentum"],
#                 values=['mean_train_score', "mean_test_score"])


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[ ]:


def build_model_2_1():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, 
               train=False, epochs=15, batch_size=64)
answer_q_2_1 = """Model complexity was increased up to a point where the validation error didn't decrease anymore.
This way an accuracy of not more than 92% could be obtained. Hence, I considered making the model
more complex and then adding regularization to make it not overfit. The best architecture is a VGG 
like model. The filter size is increased everytime to compensate for the max pooling. The padding is added to not
loose the boundary pixels.
First I added dropout because it's an effective regularization method and tuned the dropout rates. 
This performed well. After it I also added batch normalization which made it train faster. Using the
early stopping method at epoch 15 gives the best result. The model isn't overfitting and a reasonable accuracy is obtained."""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmented_split = preprocessed_split

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.1)

it_train = train_datagen.flow(augmented_split[0], augmented_split[2], batch_size=64)

steps = int(augmented_split[0].shape[0] / 64)

run_evaluation("model_2_2", build_model_2_1, (it_train, augmented_split[1], augmented_split[3]), base_dir, 
               train=False, generator=True, epochs=30, batch_size=None, steps_per_epoch=steps)
answer_q_2_2 = """If we apply too much data augmentation we'll loose too much information since the image size
is only 32*32 and information will be destroyed. Hence, only small data augmentations are applied. Where shifting and flipping isn't applied
since this decreased accuracy, probably because the important number is centered and easily confused if flipped. The performance didn't
increase since not much variation in the data is generated. However, it's more sure about its predictions"""
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

model_test = load_model_from_file(base_dir, "model_2_2")
X_test_gray = rgb2gray(X_test).astype(np.float32)
test_accuracy_3_1 = model_test.evaluate(X_test_gray, y_test)[1]

def plot_confusion_matrix():
    model = load_model_from_file(base_dir, "model_2_2")
    
    X_test_gray = rgb2gray(X_test).astype(np.float32)
    y_pred = model.predict(X_test_gray)
    cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.array(list(range(1,11)))%10, rotation=45, ha="right")
    ax.set_yticklabels(np.array(list(range(1,11)))%10)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    for i in range(100):
        ax.text(int(i/10),i%10,cm[i%10,int(i/10)], ha="center", va="center", color="w")

def plot_misclassifications():
    model = load_model_from_file(base_dir, "model_2_2")
    
    X_test_gray = rgb2gray(X_test).astype(np.float32)
    y_pred = model.predict(X_test_gray)
    nine = np.nonzero(np.logical_and((np.argmax(y_pred, axis=1) == 9), (np.argmax(y_test, axis=1) == 8)))[0]
    one = np.nonzero(np.logical_and((np.argmax(y_pred, axis=1) == 0), (np.argmax(y_test, axis=1) != 0)))[0]
    two = np.nonzero(np.logical_and((np.argmax(y_pred, axis=1) == 6), (np.argmax(y_test, axis=1) == 1)))[0]
    
    
    np.random.seed(12621)
    nine_r = np.random.choice(nine, size=3)
    one_r = np.random.choice(one, size=11)
    two_r = np.random.choice(two, size=6)
    misclassified_samples = np.concatenate((nine_r, one_r, two_r), axis=0)
    fig, axes = plt.subplots(2, 10,  figsize=(30, 5))
    for nr, i in enumerate(misclassified_samples):
        im = np.squeeze(X_test_gray[i])
        axes[int(nr/10)][nr%10].imshow(im, cmap='gray')
        axes[int(nr/10)][nr%10].set_xlabel("Predicted: %s,\n Actual : %s" % (((np.argmax(y_pred[i])+1)%10),((np.argmax(y_test[i])+1)%10)))
        axes[int(nr/10)][nr%10].set_xticks(()), axes[int(nr/10)][nr%10].set_yticks(())

    plt.show();

    
answer_q_3_1 = """It can be seen that some obvious confusions, where parts of the numbers have the same shape,
are often made by the model, e.g. 6-8, 1-7.
However, the model also has some unexpected behavior. Many errors are made by predicting numbers as a 1. It seems 
that the model predicts 1 if it's unsure. By plotting some misclassifications, I choose to look at misclassifications
that gets predicted as 1 and two obvious misclassification (9-0 and 2-7), it can be seen that there is erroneous data and 
most images are really blurry and/or bright or dark. Some images are predicted wrong since the true label isn't
centered in the image. Besides, if unsure it predicts 1 if any pixels are on a vertical line."""
print("Answer is {} characters long".format(len(answer_q_3_1)))


# In[ ]:


plot_confusion_matrix()


# In[ ]:


plot_misclassifications()


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[ ]:


def plot_activation(layer_index, activations, layer_names, images_per_row=16):
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


# In[ ]:


def plot_activations():
    model = load_model_from_file(base_dir, "model_2_2")
    X_test_gray = rgb2gray(X_test).astype(np.float32)
    
    img_tensor = X_test_gray[0]
    img_tensor = np.expand_dims(img_tensor, axis=0) 

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
        
    for nr, layer in enumerate(layer_names):
        if layer.startswith('conv2d'):
            plot_activation(nr, activations, layer_names)
             
    
answer_q_3_2 = """It can be observed that in the first layer not all filters react to the 8. This is good since 
then these filters let the image unchanged and this information is getting deeper in the network. Besides, this means
that not all filter reply to all different images. This is also preferable since it means that the network learned to classify. In layer 2 and 3 it can be seen
that the filter respond to the shape of the 8. In the next layers the resolution is lower and abstract patterns are observed."""
print("Answer is {} characters long".format(len(answer_q_3_2)))


# In[ ]:


# plot_activations()


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[ ]:


import tensorflow.keras.backend as K
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import cv2

def plot_activation_map():
    model = load_model_from_file(base_dir, "model_2_2")
    
    X_test_gray = rgb2gray(X_test).astype(np.float32)
    img_tensor = X_test_gray[0]
    x = np.expand_dims(img_tensor, axis=0) 
    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_95')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((8, 8))
    
    heatmap = cv2.resize(heatmap, (img_tensor.shape[1], img_tensor.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img = heatmap * 0.1 + img_tensor
    
    plt.imshow(img)
    
# plot_activation_map()


# In[ ]:


def plot_3_3():
    plot_activation_map()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[ ]:


from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(32, 32, 3))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name.startswith('block3'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
def build_model_4_1():
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, 
               train=False, epochs=15, batch_size=64)
answer_q_4_1 = """Not unfreezing the last layers gives a really bad overfitting model, since
our new task differs from imagenet dataset (where I can't find numbers).
By increasing the number of layers that are unfreezed, unfreezing
blocks 3, 4, and 5 gives the best performance if stopping at 15 epochs. While doing this a small learning rate was used to prevent catastrophic forgetting.
Performance is lower than 2.1/2.2. Because no regularization is applied. Since this doesn't help for learning a good embedding."""
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[ ]:


#import pickle
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
    model = load_model_from_file(base_dir, "model_4_1")
    base = model.get_layer('vgg16')
    
    X_train_em = base.predict(X_train)
    store_embedding(X_train_em, 'train_em')
    
    X_val_em = base.predict(X_val)
    store_embedding(X_val_em, 'val_em')
    
    X_test_em = base.predict(X_test)
    store_embedding(X_test_em, 'test_em')


def generate_pipeline():
    """ Returns an sklearn pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    clf = SVC()
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.99, svd_solver='full')),
        ('classifier', clf)
    ])
    return pipeline

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """ Evaluates the given pipeline, trained on the given embedded training set and 
      evaluated on the supplied embedded test set. Returns the accuracy score.
    """    
    pipeline.fit(X_train, y_train)
    
    accuracy = pipeline.score(X_test, y_test)
    return accuracy

def evaluation_4_2(X_train, y_train, X_test, y_test):
    """ Runs 'evaluate_pipeline' with embedded versions of the input data 
    and returns the accuracy.
    """
    if os.path.isfile(os.path.join(base_dir, 'train_em_embedding.p')):
        X_train_em = load_embedding('train_em')
        X_test_em = load_embedding('test_em')
    else:
        store_embeddings()
        X_train_em = load_embedding('train_em')
        X_test_em = load_embedding('test_em')
    
    y_auto = (np.argmax(y_train, axis=1)+1)%10
    X_auto = np.reshape(X_train_em, (63544, 512))

    y_auto_test = (np.argmax(y_test, axis=1)+1)%10
    X_auto_test = np.reshape(X_test_em, (19858, 512))
    
    pipeline = generate_pipeline()
    acc = evaluate_pipeline(pipeline, X_auto, y_auto, X_auto_test, y_auto_test)
    return acc

# For images on non-deep learning techniques it hold that:
# Non-deep learning techniques don't consider locality of features and do not deal with translation invariance. They 
# consider the order of the features as irrelevant. Therefore, it's much harder to build an image classifier.

answer_q_4_2 = """I first ran automl for 90 minutes. This gave BernouilliNB without preprocessing as the best pipeline. Which got only 81% accuracy on the validation set. 
This seemed very unlikely. Hence, I run some models myself. SVC with PCA gave me 89% accuracy. It doesn't beat the best model thus far. Although the embeddings should have a 
structure that non-deep learning techniques could learn efficiently. However, it is also a fact that CNNs are best at image classification"""
print("Pipeline:",generate_pipeline())
print("Answer is {} characters long".format(len(answer_q_4_2)))


# In[ ]:


# import logging
# import openml as oml
# import gama

# model = load_model_from_file(base_dir, "model_4_1")
# base = model.get_layer('vgg16')
# X_train_em = base.predict(X_train)
# X_val_em = base.predict(X_val)

# from gama import GamaClassifier
# automl = GamaClassifier(
#     max_total_time=5400, # in seconds
#     n_jobs=1,  
#     scoring='accuracy', 
#     verbosity=logging.INFO, 
#     keep_analysis_log="no2.log",  
# )

# y_auto = (np.argmax(y_train, axis=1)+1)%10
# X_auto = np.reshape(X_train_em, (63544, 512))

# y_auto_val = (np.argmax(y_val, axis=1)+1)%10
# X_auto_val = np.reshape(X_val_em, (15887, 512))

# automl.fit(X_auto, y_auto)

# automl.score(X_auto_val, y_auto_val)


# In[ ]:


# def transform_evaluations(df):
#     """ The GamaReport was initially developed for use within GAMA tooling.
#     For this reason it contains some hard to interpret, useless or internal data.
#     For clarity, we filter this out for you.
#     """
#     df = df.drop(['id', 'length_cummax', 'relative_end'], axis=1)
#     df['length'] = -df['length']
#     return df

# from gama.logging.GamaReport import GamaReport
# report = GamaReport(logfile="no2.log")
# evaluations = transform_evaluations(report.evaluations)
# evaluations.sort_values('accuracy', ascending=False)


dg_code= """
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.1)"""
last_edit = 'May 26, 2020'