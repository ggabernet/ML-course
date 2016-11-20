################################
#       Data set splitting     #
################################

X_train = Data
y_train = Targets

#X_train, X_test, y_train, y_test = \
#     train_test_split(Data, Targets, test_size=0.33, random_state=42)

Data_class_0 = []
Data_class_1 = []
Targets_class_0 = []
Targets_class_1 = []

for i in range(0, len(Targets)):
    if Targets[i] == 0:
        Data_class_0.append(Data[i])
        Targets_class_0.append(Targets[i])
    if Targets[i] == 1:
        Data_class_1.append(Data[i])
        Targets_class_1.append(Targets[i])

percent_train = 0.33

train_set_0 = range(0, int(percent_train*len(Targets_class_0)))
train_set_1 = range(0, int(percent_train*len(Targets_class_0)))

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

