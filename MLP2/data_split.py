################################
#       Data set splitting     #
################################

#the inital input data
X_train = Data
y_train = Targets

#setting the empty lists
Data_class_0 = []
Data_class_1 = []
Targets_class_0 = []
Targets_class_1 = []

#seperates the data sets into class_0 and class_1
#also saves the index for these data sets in Targets_class_0 and Targets_class_1
for i in range(0, len(Targets)):
    if Targets[i] == 0:
        #stores the data belonging to class 0
        Data_class_0.append(Data[i])
        #saves the index for the class 0 data sets
        Targets_class_0.append(Targets[i])
    if Targets[i] == 1:
        # stores the data belonging to class 1
        Data_class_1.append(Data[i])
        # saves the index for the class 1 data sets
        Targets_class_1.append(Targets[i])

#the percent of data sets in the training set
percent_train = 0.33

#saves the first (percent_train) indices for use in the training set for both class 0 and 1 data sets
train_set_0 = range(0, int(percent_train*len(Targets_class_0)))
train_set_1 = range(0, int(percent_train*len(Targets_class_1)))

#saves the remaining (percent_train) indices for use in the testing set for both class 0 and 1 data sets
test_set_0 = set(range(0,len(Targets_class_0))).difference(set(train_set_0))
test_set_1 = set(range(0,len(Targets_class_1))).difference(set(train_set_1))

X_train = []
y_train = []

#train set

for i in train_set_0:
    X_train.append(Data_class_0[i][0])
    y_train.append(Targets_class_0[i])
for i in train_set_1:
    X_train.append(Data_class_1[i][0])
    y_train.append(Targets_class_1[i])

#test set

X_test = []
y_test = []

for i in test_set_0:
    X_test.append(Data_class_0[i][0])
    y_test.append(Targets_class_0[i])
for i in test_set_1:
    X_test.append(Data_class_1[i][0])
    y_test.append(Targets_class_1[i])

