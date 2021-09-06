#!/usr/bin/env python
# coding: utf-8

# # Load dependencies

# In[1]:


import numpy as np
# Better to fix the seed in the beginning:
seed = 666
np.random.seed(seed)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale

from keras import backend 
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Dense


# # Utility functions (data preprocessing and KFold cross-validation)

# In[2]:


### Scale and center features, transform labels into a one-hot encoding vector:
def preprocess_data(X, y):
### TO DO ###
    X_out = scale(X)
    y_out = to_categorical(y)
    return X_out, y_out

### Training history plot function: (this function is finished, nothing to add !)
def print_training_history(training_history, fig_idx):
    epoch_absciss = range(1, len(training_history.history['loss'])+1)
    plt.figure(fig_idx, figsize=(10, 5))
    plt.suptitle("MLP model assessment")
    plt.subplot(1, 2, 1)
    plt.plot(epoch_absciss, training_history.history['loss'])
    plt.plot(epoch_absciss, training_history.history['val_loss'])
    plt.title("Train/Validation loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train loss', 'Validation loss'], loc='best')
    plt.subplot(1, 2, 2)
    plt.plot(epoch_absciss, training_history.history['accuracy'])
    plt.plot(epoch_absciss, training_history.history['val_accuracy'])
    plt.title("Train/Validation accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train accuracy', 'Validation accuracy'], loc='best')
    plt.show()
    
### Select a MLP model on a list of hyper-parameters instances, via Kfold cross-validation:
def KFold_model_selection(X, y, fixed_hyper_parameters, hyper_parameters_instances, num_folds, seed):
### TO DO ###
    def KFold_split(X, Y, num_folds, seed):
        KFold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        X_train_folds = []
        X_val_folds = []
        Y_train_folds = []
        Y_val_folds = []
        for (kth_fold_train_idxs, kth_fold_val_idxs) in KFold_splitter.split(X, Y):
            X_train_folds.append(X[kth_fold_train_idxs])
            X_val_folds.append(X[kth_fold_val_idxs])
            Y_train_folds.append(Y[kth_fold_train_idxs])
            Y_val_folds.append(Y[kth_fold_val_idxs])
        return X_train_folds, X_val_folds, Y_train_folds, Y_val_folds
    
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    X_train_folds, X_val_folds, Y_train_folds, Y_val_folds = KFold_split(X_train_val, Y_train_val, num_folds, seed)
    mean_val_MSEs = []
    for hyper_parameters_instance in hyper_parameters_instances:
        print("\nNow preprocessing hyper-parameter instance", hyper_parameters_instance)
        mean_val_MSE = perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds,
                                       fixed_hyper_parameters, 
                                       hyper_parameters_instance)
        print("Mean validation MSE:", mean_val_MSE)
        mean_val_MSEs.append(mean_val_MSE)
    best_instance_idx = mean_val_MSEs.index(min(mean_val_MSEs))
    best_hyper_parameters_instance = hyper_parameters_instances[best_instance_idx]
    print("\n\nBest hyper-parameter instance:", best_hyper_parameters_instance)
    best_model_test_MSE = assess_MLP(X_train_val, X_test, Y_train_val, Y_test,
                                       fixed_hyper_parameters,
                                       hyper_parameters_instances[best_instance_idx])
    print("Test MSE:", best_model_test_MSE)
                                       
    return

### KFold cross-validation of a MLP model with given hyper-parameters:
def perform_KFold_CV(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds, fixed_hyper_parameters, hyper_parameters_instance):
### TO DO ###
    val_fold_MSEs = []
    # For each fold, assess a surrogate model with fixed hyper-parameters:
    cmpt = 0
    for X_train_fold, X_val_fold, Y_train_fold, Y_val_fold in zip(X_train_folds, X_val_folds, Y_train_folds, Y_val_folds):
        val_fold_MSE = assess_MLP(X_train_fold, X_val_fold, Y_train_fold, Y_val_fold, fixed_hyper_parameters, hyper_parameters_instance)
        cmpt += 1
#         print("Surrogate model", str(cmpt) + "/" + str(len(X_val_folds)), "validation MSE:", val_fold_MSE)
        val_fold_MSEs.append(val_fold_MSE)
    # Compute the mean validation MSE between all the folds:
    mean_val_MSE = np.mean(val_fold_MSE)
    return mean_val_MSE
### Fit and evaluate a MLP model with given hyper-parameters:
def assess_MLP(X_train, X_test, y_train, y_test, fixed_hyper_parameters, hyper_parameters_instance, verbose=False):
### TO DO ###
    in_shape = X_train.shape[1]
    num_y_classes = y_train.shape[1]
    myMLP = build_MLP(in_shape, num_y_classes, hyper_parameters_instance)
    myMLP.fit(X_train, y_train,              batch_size=fixed_hyper_parameters["train batch size"],              epochs=fixed_hyper_parameters["epochs"])
    mytest_loss, mytest_accuracy = myMLP.evaluate(X_test, y_test, fixed_hyper_parameters["train batch size"])
    return mytest_loss


# # MLP (multi-layer perceptron) builder

# In[3]:


### Build a simple fully-connected MLP with SGD model:
def build_MLP(input_shape, num_classes, hyper_parameters_instance=None): #add hyper parameters here
    MLP = Sequential()
    # Hidden layers (fully connected/dense):
    if hyper_parameters_instance==None:
        MLP.add(Dense(10, activation='relu'))
    else:
        if hyper_parameters_instance["HiddenLayerActivationRelu"]==True:
            MLP.add(Dense(10, activation='relu'))
            if hyper_parameters_instance["Flag"]==True:
                MLP.add(Dense(10, activation='relu'))
        else:
            MLP.add(Dense(10, activation='sigmoid'))
            if hyper_parameters_instance["Flag"]==True:
                MLP.add(Dense(10, activation='sigmoid'))
    # Output layer (fully-connected/dense):
    MLP.add(Dense(units=num_classes, activation='softmax'))
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    MLP.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
    return MLP


# # Load and preprocess the Wine dataset

# In[4]:


# Load the Wine dataset:
X = load_wine().data
y = load_wine().target
# Get the shape of the individual feature vectors in the dataset:
input_shape = X.shape[1]
# Get the number of classes:
num_classes = (np.unique(y)).shape[0]
# Preprocess data: (implement the preprocess_data function)
X, y = preprocess_data(X, y)


# # Train, validate and evaluate a MLP model, and plot the results:

# In[5]:


# Number of epochs:
num_epochs = 20
# Train batch size:
train_batch_size = 16
# Split data into train/val/test sets:
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.2, random_state = 0)

# Load an MLP:
model = build_MLP(input_shape, num_classes)
# print(model.summary())
# Train and validate MLP, store the training history in a variable:
training_history = model.fit(X_train, Y_train,
                             batch_size = train_batch_size, epochs = num_epochs,
                             validation_data = [X_val , Y_val])
model.summary()
# Evaluate the model:
test_loss, test_accuracy = model.evaluate(X_test, Y_test, train_batch_size)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
# Plot training history:
# print_training_history(training_history, fig_idx=1)


# # Model selection of our MLP

# In[7]:


# Number of folds in KFold cross-validation:
num_folds = 5
# Number of epochs:
num_epochs = 5
# Train batch size:
train_batch_size = 16
# Create the list of hyper-parameters instances:
hyper_parameters_instances = [{"Flag": True, "HiddenLayerActivationRelu": True},
                              {"Flag": True, "HiddenLayerActivationRelu": False},
                              {"Flag": False, "HiddenLayerActivationRelu": True},
                              {"Flag": False, "HiddenLayerActivationRelu": False}]
# Also store the fixed hyper-parameters:
fixed_hyper_parameters = {"epochs": num_epochs, 
                          "train batch size": train_batch_size}
# Select model with KFold cross-validation:
KFold_model_selection(X, y, fixed_hyper_parameters, hyper_parameters_instances, num_folds, seed)

