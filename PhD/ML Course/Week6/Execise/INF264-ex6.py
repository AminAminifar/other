#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import seed
from random import random
from random import randrange
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


plt.rcParams['figure.figsize'] = [20.0, 7.0]
plt.rcParams.update({'font.size': 22,})
sns.set_palette('viridis')
sns.set_style('white')
sns.set_context('talk', font_scale=0.8)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# # 1. Bootstrap

# In[3]:


#TODO
# Create a random subsample from the dataset with replacement
def subsample(X,y, ratio=0.8):
    
    # pick a random subsample of X and the corresponding y
    num_of_instances = X.shape[0]
    indices = np.random.randint(num_of_instances, size=round(num_of_instances*ratio))
    sample_X, sample_y = X[indices,:], y[indices]
    return sample_X, sample_y


# In[4]:


#TODO
# Bootstrap Aggregation Algorithm
def bagging(X_train, y_train, X_test, n_clfs, Classifier):
    clfs = list()
    for i in range(n_clfs):
        # train the clfs on the train subsamples with random_state = seed and add them to the list
        Classifier.random_state = seed
        features, labels = subsample(X_train,y_train)
        clfs.append(Classifier().fit(features, labels))
    
    index = 0
    y_ = [None] * X_test.shape[0]
    for row in X_test:
        row = row.reshape(1,-1)
        # predict for each of the classifiers
        predicted_y =  list()
        for i in range(0, len(clfs)):
            predicted_y.append(int(clfs[i].predict(row))) # int classes
        #pick the prediction with the highest number
        temp = np.argmax(np.bincount(predicted_y))
        y_[index] = temp
        index = index + 1
                    
    return(y_)


# In[5]:


def KFold_split(X, y, num_folds, seed):
    KFold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    X_train_folds = []
    X_val_folds = []
    y_train_folds = []
    y_val_folds = []
    for (kth_fold_train_idxs, kth_fold_val_idxs) in KFold_splitter.split(X, y):
        X_train_folds.append(X[kth_fold_train_idxs])
        X_val_folds.append(X[kth_fold_val_idxs])
        y_train_folds.append(y[kth_fold_train_idxs])
        y_val_folds.append(y[kth_fold_val_idxs])
    return X_train_folds, X_val_folds, y_train_folds, y_val_folds


# In[6]:


#TODO
def evaluate_algorithm(X_train_val, y_train_val, num_folds, seed, algorithm, *args):
    # Extract train and validation folds:
    X_train_folds, X_val_folds, y_train_folds, y_val_folds = KFold_split(X_train_val, y_train_val, num_folds, seed)
    scores = list()
    
    for X_train_fold, X_val_fold, y_train_fold, y_val_fold in zip(X_train_folds, X_val_folds
                                                                  , y_train_folds, y_val_folds):
        predictions = algorithm(X_train_fold, y_train_fold, X_val_fold, *args)
        scores.append(accuracy_score(y_val_fold, predictions))#compute the accuracy
    return scores


# In[7]:



# Test bagging on the sonar dataset
seed = 2
# load and prepare data
filename = 'sonar.all-data'
dataset = pd.read_csv(filename,header=None)
X = dataset.iloc[:,:-1].to_numpy()
y = (dataset.iloc[:,-1].to_numpy()=='M').astype(int)
# evaluate algorithm
num_folds = 5
sample_size = 0.8
random.seed(seed)
# Extract a test set:
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=seed)




# For each hyper-parameter instance, do KFold cross validation:
for n_trees in [10, 50, 100, 150, 200]:
    scores = evaluate_algorithm(X_train_val, y_train_val, num_folds, seed, bagging, n_trees, DecisionTreeClassifier) #compute scores for the n_trees
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f' % (sum(scores)/float(len(scores))))
    


# In[8]:


Best_n = 50
print('Test set Accuracy: %.3f' %(accuracy_score(bagging(X_train_val,y_train_val,X_test,Best_n
                                                             ,DecisionTreeClassifier),y_test)))


# # 2. Missing numerical values

# In[9]:


def load_diabetes():
    dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
    print(dataset.describe())
    return dataset


# In[10]:


# print the first 20 rows of data
dataset = load_diabetes()
print(dataset.head(20))


# In[11]:


#TODO
#Count the number of zero values in column indeces [1,2,3,4,5]
num_of_zeros = [None] * 5
for i in [1,2,3,4,5]:
    thelist = dataset.values[:,i].tolist()
    frequencies = np.array(np.unique(thelist, return_counts=True)).T
    index = np.where(frequencies[:,0] == 0)
    if index[0].size != 0:
        num_of_zeros[i-1] = int(frequencies[index[0],1])
print(num_of_zeros)


# In[12]:


#TODO
# mark zero values as missing or NaN
for i in [1,2,3,4,5]:
    for j in range(dataset.values.shape[0]):
        if dataset.values[j,i]==0:
            dataset[i][j] = np.nan
# count the number of NaN values in each column
print(dataset.isnull().sum())


# In[13]:


dataset.head(20)


# In[14]:


#TODO
#delete rows contating NAN values from the dataset using .dropna(inplace = True) built-in function
# print (dataset.shape)
dataset.dropna(inplace=True)
print (dataset.shape)


# In[15]:


values = dataset.values
X = values[:,0:8]
y = values[:,8]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=seed)


# In[16]:


#TODO
#complete the function to 
#evaluate the MLP model on the test set
def evaluate_MLP(X_train, y_train,X_valid,y_valid,seed=7):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    result = model.predict(X_valid)
    print(accuracy_score(result,y_valid))


# In[17]:


evaluate_MLP(X_train, y_train,X_valid,y_valid,7)


# In[18]:


dataset = load_diabetes()
#TODO
# Mark zero values of column indices [1,2,3,4,5] as missing or NaN
for i in [1,2,3,4,5]:
    for j in range(dataset.values.shape[0]):
        if dataset.values[j,i]==0:
            dataset[i][j] = np.nan
# This time fill missing values with mean column values using .fillna(dataset.mean(),inplace=True)
dataset.fillna(dataset.mean(),inplace=True)
# count the number of NaN values in each column
print(dataset.isnull().sum())


# In[19]:


dataset.head(20)


# In[20]:


values = dataset.values
X = values[:,0:8]
y = values[:,8]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=seed)


# In[21]:


X_train.shape


# In[22]:


evaluate_MLP(X_train, y_train,X_valid,y_valid,7)


# # 3. Imbalanced Data

# In[23]:


df = pd.read_csv('creditcard.csv')
print(df.shape)
df.head()


# In[24]:


print(df.Class.value_counts())


# In[25]:


# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(df.Class, palette='viridis')
g.set_xticklabels(['Not Fraud', 'Fraud'])
g.set_yticklabels([])

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Transactions', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
plt.show()


# In[26]:


# TODO
# print percentage of samples where target == 1
df.Class.value_counts()[1]/df.Class.value_counts()[0]


# In[27]:


# Prepare data for modeling
# Separate input features and target
y = df.Class
X = df.drop('Class', axis=1)

# setting up validation and training sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=27)


# In[28]:


#TODO
# Train a DummyClassifier to predict with 'most_frequent' strategy
dummy = DummyClassifier()
dummy.fit(X_train,y_train)
dummy_pred = dummy.predict(X_valid)

# checking unique labels
print('Unique predicted labels: ', (np.unique(dummy_pred)))

# checking accuracy
print('Validation score: ', accuracy_score(y_valid, dummy_pred))


# In[29]:


def evaluate_imbalanced(y_valid, lr_pred):
    # Checking accuracy
    print('Accuracy: ', accuracy_score(y_valid, lr_pred))
    #recall score
    print('Recall: ',recall_score(y_valid, lr_pred))
    #precision score
    print('Precision: ', precision_score(y_valid, lr_pred))
    # f1 score
    print('F1 score: ',f1_score(y_valid, lr_pred))
    # confusion matrix
    print('ConfMat')
    print(pd.DataFrame(confusion_matrix(y_valid, lr_pred)))


# In[30]:


#TODO
# Train a LogisticRegressio model with solver as 'liblinear' on the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)
 
# Predict on validation set
lr_pred = lr.predict(X_valid)


# In[31]:


evaluate_imbalanced(y_valid, lr_pred)


# In[32]:


from sklearn.utils import resample


# In[33]:


y = df.Class
X = df.drop('Class', axis=1)

# setting up validation and training sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=27)


# In[34]:


# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)
X.head()


# In[35]:


# TODO
# separate minority and majority classes
not_fraud =  pd.DataFrame(df, index=np.where(df.values[:,-1] == 0)[0])
fraud = pd.DataFrame(df, index=np.where(df.values[:,-1] == 1)[0])

# upsample minority using resample and n_samples equal to the size of majority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=not_fraud.shape[0], # match number in majority class
                          random_state=27) # reproducible result


# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
upsampled.Class.value_counts()


# In[36]:


#TODO
# trying logistic regression again with the balanced dataset
y = upsampled.Class
X = upsampled.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# y_train = ...
# X_train = ...

# Train a logistic regression with solver 'liblinear' on the train data
upsampled = LogisticRegression(solver='liblinear')
upsampled.fit(X_train, y_train)
# predict on the test data
upsampled_pred = upsampled.predict(X_test)


# In[37]:


evaluate_imbalanced(y_test, upsampled_pred)


# In[38]:


# still using our separated classes fraud and not_fraud from above
# TODO
# downsample majority
not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = fraud.shape[0], # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

# checking counts
downsampled.Class.value_counts()


# In[39]:


# TODO
# trying logistic regression again with the undersampled dataset
y = downsampled.Class
X = downsampled.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# X_train = ...
# y_train = ...

# Train a logistic regression with solver 'liblinear' on the train data
undersampled = LogisticRegression(solver='liblinear')
undersampled.fit(X_train, y_train)
undersampled_pred = undersampled.predict(X_test)


# In[40]:


evaluate_imbalanced(y_test, undersampled_pred)


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


df = pd.read_csv('creditcard.csv')
y = df.Class
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# In[43]:


# train a random forest model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# predict on test set
rfc_pred = rfc.predict(X_test)


# In[44]:


evaluate_imbalanced(y_test,rfc_pred)

