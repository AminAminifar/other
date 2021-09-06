from __future__ import division
import pandas as pd
import numpy as np
import csv
# from Crypto.PublicKey import RSA
from Crypto import Random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import math

from Crypto.Random import random
from Crypto.PublicKey import ElGamal
from Crypto.Util.number import GCD
from Crypto.Hash import SHA

import binascii

# print"start"
#
# message = 100
# random_generator = Random.new().read
# key = ElGamal.generate(1024, random_generator)
# publickey = key.publickey()
# k = random.StrongRandom().randint(1,key.p-1)
# cipher = publickey.encrypt(long(message), k)[0]
#
# print("Encrypted:",cipher)
#
#
# print"end"

reader = csv.reader(open('Adult Data Set/CSV/Train_Data.csv', "rb"), delimiter=",")
x = list(reader)
Data = np.array(x).astype("float")

reader = csv.reader(open('Adult Data Set/CSV/Test_Data.csv', "rb"), delimiter=",")
x = list(reader)
TestData = np.array(x).astype("float")
print "Data Imported!"
DSlength_D1 = np.shape(Data)[0]  # number of instances
DSlength_D2 = np.shape(Data)[1]  # number of features

TestDSlength_D1 = np.shape(TestData)[0]  # number of instances
TestDSlength_D2 = np.shape(TestData)[1]  # number of features


##################
# STRAT 10RUN LOOP
##################
Global_Accuracy_E_N = 0
Number_of_Runs = 10
Accuracy_E_N = [None] * Number_of_Runs
for run_number in range(Number_of_Runs):
    print('RUN NUMBER: '+str(run_number)+' started!')

    # Shuffle Data
    indx = np.arange(DSlength_D1)  # create a array with indexes for Data
    np.random.shuffle(indx)
    Data = Data[indx, :]

    pd.DataFrame(Data).to_csv('Adult Data Set/CSV/ElGamal/10run/Adult_sh_' + str(run_number) + '.csv', header=False,
                              index=False)
    print "Shuffled Data Saved!"

    # start Encryption
    random_generator = Random.new().read
    key = ElGamal.generate(1024, random_generator)
    publickey = key.publickey()
    k = random.StrongRandom().randint(1,key.p-1)

    # Initialize Data_e (Data_encrypted)

    Data_e = [None] * DSlength_D1
    for i in range(DSlength_D1):
        Data_e[i] = [None] * (DSlength_D2 - 1)

    # Store Encrypted Data in Data_e
    print("progress (Encryption of training dataset):")
    for i in range(0, DSlength_D1):
        for j in range(0, (DSlength_D2 - 1)):
            Data_e[i][j] = publickey.encrypt(long(Data[i][j]), k)[1]
        if np.mod(i, np.floor(DSlength_D1 / 10)) == 0:
            print str(round((i / DSlength_D1) * 100)) + "%"

    TestData_e = [None] * TestDSlength_D1
    for i in range(TestDSlength_D1):
        TestData_e[i] = [None] * (TestDSlength_D2 - 1)

    # Store Encrypted TestData in Data_e
    print("progress(Encryption of test dataset):")
    for i in range(0, TestDSlength_D1):
        for j in range(0, (TestDSlength_D2 - 1)):
            TestData_e[i][j] = publickey.encrypt(long(TestData[i][j]), k)[1]
        if np.mod(i, np.floor(TestDSlength_D1 / 10)) == 0:
            print str(round((i / TestDSlength_D1) * 100)) + "%"

    print "Encryption Done!"

    Data_e = np.array(Data_e)  # These stuff are done to make the matrix work with numpy functions
    TestData_e = np.array(TestData_e)  # These stuff are done to make the matrix work with numpy functions
    # Save Data and TestData
    # pd.DataFrame(Data_e).to_csv("Adult Data Set/CSV/ElGamal/Adult_Encrypted.csv", header=False, index=False)#ADDRESS
    # pd.DataFrame(TestData_e).to_csv("Adult Data Set/CSV/ElGamal/Test_Adult_Encrypted.csv", header=False, index=False)#ADDRESS
    #
    # print "Data Saved!"

    # start Normalization
    # Data_e = np.array(Data_e) #These stuff are done to make the matrix work with numpy functions

    # Initialize N_Data_e (Normalized_Data_encrypted)
    N_Data_e = [None] * DSlength_D1
    for i in range(DSlength_D1):
        N_Data_e[i] = [None] * (DSlength_D2 - 1)
    N_Data_e = np.array(N_Data_e)

    # Initialize N_TestData_e (Normalized_TestData_encrypted)
    N_TestData_e = [None] * TestDSlength_D1
    for i in range(TestDSlength_D1):
        N_TestData_e[i] = [None] * (TestDSlength_D2 - 1)
    N_TestData_e = np.array(N_TestData_e)

    # Store Normalized Data in N_Data_e
    for j in range(0, (DSlength_D2 - 1)):
        column1 = Data_e[:, j]
        column2 = TestData_e[:, j]
        max = long(np.amax([np.amax(column1),np.amax(column2)]))
        min = long(np.amin([np.amin(column1),np.amin(column2)]))
        column_range = long(max - min)
        if (column_range==0):
            for i in range(0, DSlength_D1):
                N_Data_e[i][j] = 0
            for i in range(0, TestDSlength_D1):
                N_TestData_e[i][j] = 0
        else:
            for i in range(0, DSlength_D1):
                N_Data_e[i][j] = (Data_e[i][j] - min) / column_range
            for i in range(0, TestDSlength_D1):
                N_TestData_e[i][j] = (TestData_e[i][j] - min) / column_range
    print "Normalization Done!"

    # Adding the target lables at the end of matrix of N_E_Data
    y_train = [None] * DSlength_D1
    for i in range(0, (DSlength_D1)):
        y_train[i] = int(Data[i, (DSlength_D2 - 1)])
    y_train = np.array(y_train)
    y_train2 = [None] * DSlength_D1
    for i in range(DSlength_D1):
        y_train2[i] = [None]
    y_train2 = np.array(y_train2)
    y_train2[:, 0] = y_train
    N_E_Data_withLables = np.concatenate((N_Data_e, y_train2), axis=1)

    # Save Normalized Data
    pd.DataFrame(N_E_Data_withLables).to_csv('Adult Data Set/CSV/ElGamal/10run/Adult_Encrypted_Normalized'+str(run_number)+'.csv', header=False, index=False)
    pd.DataFrame(N_TestData_e).to_csv('Adult Data Set/CSV/ElGamal/10run/Test_Adult_Encrypted_Normalized'+str(run_number)+'.csv', header=False, index=False)

    print "Normalized Data Saved!"



    # Classification
    y_train = [None] * DSlength_D1
    for i in range(0, (DSlength_D1)):
        y_train[i] = int(Data[i, (DSlength_D2 - 1)])
    y_train = np.array(y_train)

    y_test = [None] * TestDSlength_D1
    for i in range(0, (TestDSlength_D1)):
        y_test[i] = int(TestData[i, (TestDSlength_D2 - 1)])
    y_test = np.array(y_test)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(N_Data_e, y_train)
    y_pred = clf.predict(N_TestData_e)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    tp = 0
    tn = 0
    ttp = 0
    ttn = 0

    for i in range(0, TestDSlength_D1):
        if ((y_test[i] == 1) & (y_pred[i] == 1)):
            tp = tp + 1
        if ((y_test[i] == 2) & (y_pred[i] == 2)):
            tn = tn + 1
        if (y_test[i] == 1):
            ttp = ttp + 1
        if (y_test[i] == 2):
            ttn = ttn + 1

    tpr = tp / ttp
    tnr = tn / ttn
    gmean = math.sqrt(tpr * tnr)
    print(gmean)
    Global_Accuracy_E_N = Global_Accuracy_E_N + gmean;
    Accuracy_E_N[run_number] = gmean

Global_Accuracy_E_N = Global_Accuracy_E_N/Number_of_Runs
pd.DataFrame(Accuracy_E_N).to_csv('Adult Data Set/CSV/ElGamal/10run/AccuracyTable.csv', header=False, index=False)
print("Global_Accuracy_E_N: ",Global_Accuracy_E_N)

# clf2 = RandomForestClassifier(n_estimators=100)
# clf2.fit(Data[:,0:DSlength_D2-1], y_train)
# y_pred2 = clf2.predict(TestData[:,0:TestDSlength_D2-1])
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred2))
#
# tp = 0
# tn = 0
# ttp = 0
# ttn = 0
#
# for i in range(0, TestDSlength_D1):
#     if ((y_test[i] == 1) & (y_pred2[i] == 1)):
#         tp = tp + 1
#     if ((y_test[i] == 2) & (y_pred2[i] == 2)):
#         tn = tn + 1
#     if (y_test[i] == 1):
#         ttp = ttp + 1
#     if (y_test[i] == 2):
#         ttn = ttn + 1
#
# tpr = tp / ttp
# tnr = tn / ttn
# gmean = math.sqrt(tpr * tnr)
# print(gmean)


print("end")