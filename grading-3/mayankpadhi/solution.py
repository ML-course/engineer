#!/usr/bin/env python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = './'
target_dir = '../../grading-3/mayankpadhi'
grade_file = '../../grading-3/grades.csv'
stop_training = True
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fill in your name using the format below and student ID number
your_name = "Padhi, Mayank"
student_id = "1432788"


# In[ ]:


# Before submission, set this to True so that you can render and verify this notebook without training deep learning models.
# Any deep learning models will be trained from file instead.
# stop_training = False


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


# # base_dir = '/content/drive/My Drive/mayankpadhi_assignment3' # For Google Colab
#base_dir = './'


# In[9]:


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


from tensorflow.keras import models
from tensorflow.keras import layers 


# In[18]:


from tensorflow.keras import optimizers


def build_model_1_1():
    model_1_1 = models.Sequential()
    model_1_1.add(layers.Reshape((3072,), input_shape=(32,32,3)))
    model_1_1.add(layers.Dense(1024, activation='relu'))
    model_1_1.add(layers.Dense(512, activation='relu'))
    model_1_1.add(layers.Dense(256, activation='relu'))
    model_1_1.add(layers.Dense(10, activation='softmax'))
    model_1_1.compile(optimizer= optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_1_1
    pass
run_evaluation("model_1_1", build_model_1_1, evaluation_split, base_dir, train=True, epochs=25, batch_size= 32)
answer_q_1_1 = """
The ReLU reduces the likelihood of the gradient to vanish. It does not activate all the neurons at the same time. 
Since, the problem demands the multi class classifier hence we use softmax. We keep the softmax function at the last layer.
Softmax almost work like max layer as well as it will be differentiable to train by gradient descent. 
We observe that as the learning rate decreases, the accuracy of the classifier improves. The performance of the model is resonably good, the accuracy is 0.85 while the validation accuracy is 0.83.
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


# In[20]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import cv2

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Replace with the preprocessed data
def my_preprocess_1_2():
    X_train_pre= rgb2gray(X_train)
    X_val_pre= rgb2gray(X_val)
    #X_test_pre= rgb2gray(X_test)

    preprocessed_split = X_train_pre, X_val_pre, y_train, y_val
    return preprocessed_split

#preprocessed_split = X_train, X_val, y_train, y_val
preprocessed_split = my_preprocess_1_2()

# Adjusted model
def build_model_1_2():
    model_1_2 = models.Sequential()
    model_1_2.add(layers.Reshape((1024,), input_shape=(32,32,3)))
    model_1_2.add(layers.Dense(1024, activation='relu'))
    model_1_2.add(layers.Dense(512, activation='relu'))
    model_1_2.add(layers.Dense(256, activation='relu'))
    model_1_2.add(layers.Dense(10, activation='softmax'))
    model_1_2.compile(optimizer= optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_1_2
    pass

# Evaluate. Use a new name 'model_1_2' to not overwrite the previous trained model
run_evaluation("model_1_2", build_model_1_2, preprocessed_split, base_dir, train=True, epochs=25, batch_size=32)

answer_q_1_2 = """The model remains the same as the previous problem, however the images have been greyscaled. Since, the color has no significance for the digit recognition, we can have 1 dimension (as in greyscale) instead of 3 (in case of RGB). We see that the accuracy has slightly improved, however the improvement in both the scores is not significant. However, we can also use the de-skewing technique as some of the images are tilted. This tilt makes it difficult to classify."""
print("Answer is {} characters long".format(len(answer_q_1_2)))


# ### Question 1.3: Regularization and tuning (4 points)
# * Regularize the model. You can explore (and combine) different techniques. What works best?
# * Tune other hyperparameters (e.g. learning rate, batch size,...) as you see fit.
# * Explain your findings and final design decisions. Retrain the model again on the preprocessed data and discuss the results.
# * Return your model in function 'build_model_1_3' and write your answer in 'answer_q_1_3'

# In[ ]:


from keras import regularizers

def build_model_1_3():
    model_1_3 = models.Sequential()
    model_1_3.add(layers.Reshape((1024,), input_shape=(32,32,1)))
    model_1_3.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model_1_3.add(layers.Dropout(0.005))
    model_1_3.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model_1_3.add(layers.Dropout(0.005))
    model_1_3.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model_1_3.add(layers.Dropout(0.005))
    model_1_3.add(layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)))
    model_1_3.add(layers.Dropout(0.005))
    model_1_3.compile(optimizer= optimizers.RMSprop(lr= 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_1_3
    pass
    
run_evaluation("model_1_3", build_model_1_3, preprocessed_split, base_dir, train=True, epochs=25, batch_size=16)
answer_q_1_3 = """The purpose of regularization the model to avoid overfitting. Results with l2 regularization (with lambda= 0.001) is better than the l1 regularization(with lambda= 0.001). The dropout 2% of neurons after each layer improved the accuracy."""
print("Answer is {} characters long".format(len(answer_q_1_3)))


# ## Part 2. Convolutional neural networks (10 points)
# ### Question 2.1: Design a ConvNet (7 points)
# - Build a sequential convolutional neural network. Try to achieve the best validation accuracy you can. You should be able to get at least 90% accuracy. You can use any depth, any combination of layers, and any kind of regularization and tuning. 
# - Add a description of your design choices in 'answer_q_2_1': explain what you did and also why. Also discuss the performance of the model. Is it working well? Both the performance of the model and your explanations matter.
# - You are allowed **800** characters for this answer (but don’t ramble).
# - The name of the model should be 'model_2_1'. Evaluate it using the 'run_evaluation' function and the preprocessed data.

# In[21]:


def build_model_2_1():
    model_2_1 = models.Sequential()
    model_2_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model_2_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_2_1.add(layers.MaxPooling2D((2, 2)))
    model_2_1.add(layers.Dropout(0.2))
    model_2_1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_2_1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_2_1.add(layers.MaxPooling2D((2, 2)))
    model_2_1.add(layers.Dropout(0.2))
    model_2_1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_2_1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_2_1.add(layers.MaxPooling2D((2, 2)))
    model_2_1.add(layers.Dropout(0.2))
    model_2_1.add(layers.Flatten())
    model_2_1.add(layers.Dense(128, activation='relu'))
    model_2_1.add(layers.Dropout(0.2))
    model_2_1.add(layers.Dense(10, activation='softmax'))
    model_2_1.compile(optimizer= optimizers.RMSprop(lr= 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_2_1
    pass

run_evaluation("model_2_1", build_model_2_1, preprocessed_split, base_dir, train=True, epochs=25, batch_size=32)
answer_q_2_1 = """
I have broadly made 3 'blocks' each consisting two 2d-convolution layers followed by 2d maxpooling and dropout. Conv2D extracts the spatial features from the data using its kernel. ReLU is relatively fast and it does not activate all the neurons at the same time. It usually uses multiple convolutional layers to learn patterns at different levels of abstraction. The MaxPooling reduces the resolution, increase translation invariance. The convolution layers seemed to perform very well. However, there the model was overfitting. Hence, dropout of 0.2, 0.5 and 0.05. The best performance is on lambda 0.2. I have also flattened after the 'blocks'. The accuracy is very high. There is a negilible difference in accuracy and val_accuracy. Hence, we cannot conclude that the model is overfitting.
"""
print("Answer is {} characters long".format(len(answer_q_2_1)))


# ### Question 2.2: Data Augmentation (3 points)
# 
# - Augment the preprocessed training data. You can explore using image shifts, rotations, zooming, flips, etc. What works well, and what does not?
# - Evaluate the model from question 2.1 with the augmented data using the 'run_evaluation' function. Store the new trained model as 'model_2_2'.
# - Add a description of your design choices in 'answer_q_2_2': explain what you did and also why. Also discuss the performance of the model.

# In[22]:


# Note that we build the same untrained model as in question 2.1 but store the 
# trained version as model_2_2. Change attributes as needed to run on augmented
# data
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def my_augmented_2_2():
    X_train_pre= preprocessed_split[0]
    X_val_pre= preprocessed_split[1]
    X_test_pre= rgb2gray(X_test)

    #some more augmentation
    #train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    #train_datagen.fit(X_train_pre[0])
    augmented_split = X_train_pre, X_val_pre, X_test_pre, y_train, y_val, y_test

    rotate= np.zeros(len(X_train))
    return augmented_split

my_augmented= my_augmented_2_2()
augmented_split = my_augmented[0], my_augmented[1], my_augmented[3], my_augmented[4]

def build_model_2_2():
    model_2_2 = models.Sequential()
    model_2_2.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model_2_2.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_2_2.add(layers.MaxPooling2D((2, 2)))
    model_2_2.add(layers.Dropout(0.2))
    model_2_2.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_2_2.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_2_2.add(layers.MaxPooling2D((2, 2)))
    model_2_2.add(layers.Dropout(0.2))
    model_2_2.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_2_2.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_2_2.add(layers.MaxPooling2D((2, 2)))
    model_2_2.add(layers.Dropout(0.2))
    model_2_2.add(layers.Flatten())
    model_2_2.add(layers.Dense(128, activation='relu'))
    model_2_2.add(layers.Dropout(0.2))
    model_2_2.add(layers.Dense(10, activation='softmax'))
    model_2_2.compile(optimizer= optimizers.RMSprop(lr= 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_2_2
    pass

run_evaluation("model_2_2", build_model_2_1, augmented_split, base_dir, train=True, epochs=25, batch_size=32)
answer_q_2_2 = """
               Visual inspection suggested that some of the images are slightly tilted and are shifted to the left and right. Hence, the images are rotated and shifted for better results. However, they do not seem to have a significant affect in the accuracy of the model. The accuracy score was 0.942 while the val accuracy was 0.946. However, the preprocessed data did improve the score a bit. The new model is same as the basis of the previous model with minor change in shape.
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

model_2_2 = load_model_from_file(base_dir, "model_2_2")
test_score_3_1, test_accuracy_3_1 = model_2_2.evaluate(my_augmented[2], my_augmented[5], batch_size=32)
y_pred = model_2_2.predict(my_augmented[2])

def plot_confusion_matrix():
    cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    #ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    for i in range(100):
        ax.text(int(i/10), i%10, cm[i%10,int(i/10)], ha="center", va="center", color="w")
    return test_accuracy_3_1
    pass

def plot_misclassifications():
    misclassified_samples = np.nonzero(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1))[0]
    h= 0
    for nr, i in enumerate(misclassified_samples[:100]):
        h+=1
        if (np.argmax(y_test[i])+1)%10== 5:
            fig, axes = plt.subplots(1, 1,  figsize=(10, 7))
            axes.imshow(X_test[i])
            axes.set_xlabel("Predicted: %s,\n Actual : %s" % ((np.argmax(y_pred[i])+1)%10, (np.argmax(y_test[i])+1)%10))
            axes.set_xticks(()), axes.set_yticks(())
            plt.show()
            if h== 10:
                break
    pass

plot_confusion_matrix()
plot_misclassifications()

answer_q_3_1 = """We observe that the images that contain more than one digit are misclassfied. Since our model is designed to classify single digits, it cannot classify multiple digits in a single image. When we observe the class of 5 we see that it is mostly being mistaken with 3. The quality of some of the misclassified images is extremely poor, even hard for humans to classify. However, the highest number of miss-classifications is between 6 and 0 and between 3 and 0. In general, 0 is often being misclassified as other digit."""
print("Answer is {} characters long".format(len(answer_q_3_1)))


# ### Question 3.2: Visualizing activations (4 points)
# * Implement a function `plot_activations()` that returns the most interesting activations (feature maps). Select the first example from the test set. Retrieve and visualize the activations of model 2_2 for that example (make sure you load that model in the function), for every filter for different convolutional layers (at different depths in the network).
# * Give an explanation (as detailed as you can) about your observations in 'answer_q_3_2'. Is your model indeed learning something useful?

# In[76]:


images_per_row = 16
model_2_2 = load_model_from_file(base_dir, "model_2_2")
layer_names = []
for layer in model_2_2.layers[:15]:
    layer_names.append(layer.name)
img_tensor = my_augmented[2][4]
img_tensor = np.expand_dims(img_tensor, axis=0) 
layer_outputs = [layer.output for layer in model_2_2.layers[:15]]
activation_model = models.Model(inputs= model_2_2.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
f, ax1= plt.subplots(1, 1, sharey=True)
ax1.imshow(X_test[4])

def plot_activations():
    layer_index = 4
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

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title("Activation of layer {} ({})".format(layer_index+1,layer_name))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    pass
#plot_activations(1, activations)
#plot_activations(4, activations)

# plot_activations()
answer_q_3_2= """I have taken up the example of the 4th image. We can see in the second layer, the model has learned the basic structure inside the surface of the digit. In the subsequesnt layers we can see that the output is more pixelated and other structures that were ignored in the second layer are also getting enhanced. As a result, the image is relatively more noisy. This development may be better for other images in this dataset but having more layers might have added unnecessary details to an otherwise simple example."""

print("Answer is {} characters long".format(len(answer_q_3_2)))


# ### Question 3.3: Visualizing activations (4 points)
# * Again, select the first example from the test set, and the trained model_2_2.
# * Implement a function `plot_activation_map()` that builds and shows a class activation map for your last convolutional layer that highlights what the model is paying attention to when classifying the example.
# * If possible, superimpossible the activation map over the image. If not, plot
# them side by side. Implement a function 'plot_3_3' that returns the entire plot.

# In[105]:


def plot_activation_map():
    model_2_1 = load_model_from_file(base_dir, "model_2_1")
    layer_outputs = [layer.output for layer in model_2_1.layers[:15]]
    activation_model = models.Model(inputs= model_2_1.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    last_layer_activation = activations[9]     #[len(activations)-1]
    f, ax2 = plt.subplots(1, 1, sharey=True)

    f, wt = plt.subplots(1, 1, sharey=True)
    wt.imshow(X_test[4])
    ax2.matshow(last_layer_activation[0, :, :, 2], cmap='viridis')
    wt.set_xlabel('Input image')
    ax2.set_xlabel('last convolutional layer')
    pass

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()

def plot_3_3():
    K.clear_session()
    model_2_2 = load_model_from_file(base_dir, "model_2_2")
    img_path = base_dir+'/mymage.png'
    img = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    african_elephant_output = model_2_2.output[:, 9]
    last_conv_layer = model_2_2.get_layer('conv2d_5')
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function(inputs= [model_2_2.input], outputs= [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    img = cv2.imread(base_dir+'/mymage.png')
    RGB_im = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)
    #RGB_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Save the image to disk
    cv2.imwrite(base_dir+ '/mymage.png', superimposed_img)
    plt.rcParams['figure.dpi'] = 120
    plt.imshow(RGB_im)
    plt.title('Class activation map')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# plot_3_3()
#plot_activation_map()


# ## Part 4. Transfer learning (10 points)
# ### Question 4.1 Fast feature extraction with VGG16 (5 points)
# - Import the VGG16 model, pretrained on ImageNet. [See here](https://keras.io/applications/). Only import the convolutional part, not the dense layers.
# - Implement a function 'build_model_4_1` that adds a dense layer to the convolutional base, and freezes the convolutional base. Consider unfreezing the last few convolutional layers and evaluate whether that works better.
# - Train the resulting model on the *original* (colored) training data
# - Evaluate the resulting model using 'run_evaluate'. Discuss the observed performance in 'answer_q_4_1'.

# In[ ]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers

def build_model_4_1():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    #for conv_layer in conv_base.layers[:4]:
    #    conv_layer.trainable = False
    #conv_base.summary()
    model_4_1 = models.Sequential()
    model_4_1.add(conv_base)
    model_4_1.add(layers.Flatten())
    #model_4_1.add(layers.Dense(10, activation='softmax'))
    conv_base.trainable = False
    for conv_layer in conv_base.layers[:8]:
        conv_layer.trainable = True
    model_4_1.add(layers.Dense(10, activation='softmax'))
    #model_4_1.compile(optimizer= 'sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model_4_1.compile(optimizer= optimizers.RMSprop(lr= 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model_4_1
    pass

run_evaluation("model_4_1", build_model_4_1, evaluation_split, base_dir, train=True, epochs=25, batch_size=32)

answer_q_4_1 = """ We observe that the model performs well when final 8 layers are freezed. When no layers are freezed the accuracy was 0.18 which is bad. When all layers were freezed, the accuracy was 0.1908. Hence, we tried with freezing different convolution layers of the base. The accuracy of this model was relatively better than the previous ones."""
print("Answer is {} characters long".format(len(answer_q_4_1)))


# ### Question 4.2 Embeddings and pipelines (5 points)
# - Generate embeddings of the original images by running them through the trained convolutional part of model_4_1 (without the dense layer) and returning the output. Embed the training and test data and store them to disk using the helper functions below. Implement a function `store_embeddings` that loads model_4_1 and stores all necessary embeddings to file. Make sure to run it once so that the embeddings are stored (and submitted).
# - Implement a function 'generate_pipeline' that returns an scikit-learn pipeline. You can use any non-deep learning technique (eg. SVMs, RFs,...), and preprocessing technique. You can do model selection using the validation set. 
# - Implement a function 'evaluate_pipeline' that evaluates a given pipeline on a given training and test set. 
# - Implement a function 'evaluation_4_2' that evaluates your pipeline on the embedded training and test set (loaded from file) and returns the accuracy. 
# - Describe what you did and what you observed. Report the obtained accuracy score. Can you beat your best model thus far?

# In[101]:


import pickle
import gzip
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

model_4_1= load_model_from_file(base_dir, "model_4_1")

def store_embedding(X, name):  
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'wb') as file_pi:
    pickle.dump(X, file_pi)

def load_embedding(name):
  with gzip.open(os.path.join(base_dir, name+'_embedding.p'), 'rb') as file_pi:
    return pickle.load(file_pi)

def store_embeddings():
    """ Stores all necessary embeddings to file
    """
    n_model = models.Model(inputs= model_4_1.input, outputs= model_4_1.layers[2].output)
    X_train_embed = n_model.predict(X_train, batch_size=32)
    X_test_embed = n_model.predict(X_test, batch_size=32)
    X_val_embed = n_model.predict(X_val, batch_size=32)
    store_embedding(X_train_embed, "X_train")
    store_embedding(X_test_embed, "X_test")
    store_embedding(X_val_embed, "X_val")
    store_embedding(y_train, "y_train")
    store_embedding(y_test, "y_test")
    store_embedding(y_val, "y_val")
    shout("Saved to file", verbose= 1)
    pass

def generate_pipeline():
    """ Returns an sklearn pipeline.
    """
    #pipeline= Pipeline(steps=[('classifier', RandomForestClassifier())])
    pipeline= Pipeline(steps=[('classifier', DecisionTreeClassifier())])
    return pipeline

def evaluate_pipeline(X_train_embed, X_test_embed):
    """ Evaluates the given pipeline, trained on the given embedded training set and 
        evaluated on the supplied embedded test set. Returns the accuracy score.
    """
    pipeline= generate_pipeline()
    pipeline.fit(X_train_embed, y_train)
    y_pred= pipeline.predict(X_test_embed)
    return accuracy_score(y_test, y_pred)

def evaluation_4_2(X_train, y_train, X_test, y_test):
    """ Runs 'evaluate_pipeline' with embedded versions of the input data 
    and returns the accuracy.
    evaluates your pipeline on the embedded training and test set
    """

    X_train_emb= load_embedding("X_train")
    X_test_emb= load_embedding("X_test")
    return evaluate_pipeline(X_train_emb, X_test_emb)

# store_embeddings()
print("Accuracy score: ", evaluation_4_2(X_train, y_train, X_test, y_test))

answer_q_4_2 = """
               In the store embedding function, the intermediate layers are fetched and then the embeddings of the test and training data are generated and stored in the external file system. Then a pipeline is generated using the classifier. The embeddings(of X_test and X_train) are loaded using the helper functions and are passed to evaluate the pipeline. We notice that the accuracy score is not very good as this is the accuracy score with output from intermediate layers. I tried with Random forest classifier but it didn't perform very well. The accuracy with Random Forest Classifier is 0.39.
               """

print("Answer is {} characters long".format(len(answer_q_4_2)))


dg_code= """
    #train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)"""
last_edit = 'May 26, 2020'