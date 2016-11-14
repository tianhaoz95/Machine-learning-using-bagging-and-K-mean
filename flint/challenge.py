import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.datasets import load_iris, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn import metrics

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_to_int(arr):
    temp = np.zeros(len(arr))
    for i in range(0, len(arr)):
        if isfloat(arr[i]):
            temp[i] = float(arr[i])
        else:
            temp[i] = 0
    return temp

def majority_vote(pred_list):
    y_pred = []
    for i in range(len(pred_list[0])):
        lst = [row[i] for row in pred_list]
        y_pred.append(max(set(lst), key = lst.count))
    return y_pred


def bagging_ensemble(X_train, y_train, X_test, m = None, n_clf = 10):
    bagging_ratio = 0.6
    total_size = y_train.shape[0]
    bagging_size = int(bagging_ratio * total_size)
    bagsX = []
    bagsY = []
    for i in range(0, n_clf):
        tempx = []
        tempy = []
        for j in range(0, bagging_size):
            rand = np.random.randint(0, total_size)
            tempx.append(X_train[rand, :])
            tempy.append(y_train[rand])
        bagsX.append(tempx)
        bagsY.append(tempy)
    clfs = []
    accuracy = 0
    for i in range(0, n_clf):
        clfs.append(DecisionTreeClassifier(max_features=m));
    for i in range(0, n_clf):
        clfs[i].fit(bagsX[i], bagsY[i])
    preds = []
    for i in range(0, n_clf):
        temp = clfs[i].predict(X_test)
        preds.append(temp)
    y_pred = majority_vote(preds)
    return y_pred

print("==============")
print("=====START====")
print("==============")

raw_train_data = pd.read_csv("flint_train.csv")
raw_test_data = pd.read_csv("flint_test.csv")

dictionary = ["sample_id",
"Lead_(ppb)",
"parcel_id",
"Date_Submitted",
"google_add",
"Latitude",
"Longitude",
"Owner_Type",
"Land_Value",
"Land_Improvements_Value",
"Residential_Building_Value",
"Residential_Building_Style",
"Commercial_Building_Value",
"Building_Storeys",
"Parcel_Acres",
"Rental",
"Use_Type",
"Prop_Class",
"Year_Built",
"USPS_Vacancy",
"Zoning",
"Future_Landuse",
"DRAFT_Zone",
"Housing_Condition_2012",
"Housing_Condition_2014",
"Commercial_Condition_2013",
"Hydrant_Type",
"Ward",
"PRECINCT",
"CENTRACT",
"CENBLOCK",
"SL_Type",
"SL_Type2",
"SL_Lead",
"Homestead",
"Homestead_Percent",
"HomeSEV"]

train_sample_id = raw_train_data[dictionary[0]]
test_sample_id = raw_test_data[dictionary[0]]

train_y = raw_train_data[dictionary[1]]
train_y_converted = np.zeros([len(train_y)])

print("converting lead value, limit = 15")

for i in range(0, len(train_y)):
    if int(train_y[i]) < 15:
        train_y_converted[i] = 0
    else:
        train_y_converted[i] = 1
print("done!")

print("converting homestead to binary data")

homestead_raw_train = raw_train_data["Homestead"]
homestead_raw_test = raw_test_data["Homestead"]
homestead_train = np.zeros([len(homestead_raw_train)])
homestead_test = np.zeros([len(homestead_raw_test)])
for i in range(0, len(homestead_raw_train)):
    if homestead_raw_train[i] == "yes":
        homestead_train[i] = 1
    else:
        homestead_train[i] = 0

for i in range(0, len(homestead_raw_test)):
    if homestead_raw_test[i] == "yes":
        homestead_test[i] = 1
    else:
        homestead_test[i] = 0

print("done!")

print("converting Commercial_Building_Value to binary data")

raw_Commercial_Building_Value_train = raw_train_data["Commercial_Building_Value"]
commercial_value_train = np.zeros([len(raw_Commercial_Building_Value_train)])
for i in range(0, len(raw_Commercial_Building_Value_train)):
    if raw_Commercial_Building_Value_train[i] != '0':
        commercial_value_train[i] = 1
    else:
        commercial_value_train[i] = 0

raw_Commercial_Building_Value_test = raw_test_data["Commercial_Building_Value"]
commercial_value_test = np.zeros([len(raw_Commercial_Building_Value_test)])
for i in range(0, len(raw_Commercial_Building_Value_test)):
    if raw_Commercial_Building_Value_test[i] != '0':
        commercial_value_test[i] = 1
    else:
        commercial_value_test[i] = 0

print("done!")

print("converting Use_Type to binary data")

raw_use_type_train = raw_train_data["Use_Type"]
use_type_train = np.zeros([len(raw_use_type_train)])
for i in range(0, len(raw_use_type_train)):
    if raw_use_type_train[i] == 'Residential':
        use_type_train[i] = 1
    else:
        use_type_train[i] = 0

raw_use_type_test = raw_test_data["Use_Type"]
use_type_test = np.zeros([len(raw_use_type_test)])
for i in range(0, len(raw_use_type_test)):
    if raw_use_type_test[i] == 'Residential':
        use_type_test[i] = 1
    else:
        use_type_test[i] = 0

print("done!")

print("converting prop_class to binary data")

raw_prop_class_train = raw_train_data["Prop_Class"]
prop_class_train = np.zeros([len(raw_prop_class_train)])
for i in range(0, len(raw_prop_class_train)):
    if raw_prop_class_train[i] == 'RI':
        prop_class_train[i] = 1
    else:
        prop_class_train[i] = 0

raw_prop_class_test = raw_test_data["Prop_Class"]
prop_class_test = np.zeros([len(raw_prop_class_test)])
for i in range(0, len(raw_prop_class_test)):
    if raw_prop_class_test[i] == 'RI':
        prop_class_test[i] = 1
    else:
        prop_class_test[i] = 0

print("done!")

home_sev_train = convert_to_int(raw_train_data["HomeSEV"])
home_sev_test = convert_to_int(raw_test_data["HomeSEV"])

ward_train = convert_to_int(raw_train_data["Ward"])
precinct_train = convert_to_int(raw_train_data["PRECINCT"])
centract_train = convert_to_int(raw_train_data["CENTRACT"])
cenblock_train = convert_to_int(raw_train_data["CENBLOCK"])
sl_train = convert_to_int(raw_train_data["SL_Type"])
sl2_train = convert_to_int(raw_train_data["SL_Type2"])
sll_train = convert_to_int(raw_train_data["SL_Lead"])
latitude_train = convert_to_int(raw_train_data["Latitude"])
longitude_train = convert_to_int(raw_train_data["Longitude"])


ward_test = convert_to_int(raw_test_data["Ward"])
precinct_test = convert_to_int(raw_test_data["PRECINCT"])
centract_test = convert_to_int(raw_test_data["CENTRACT"])
cenblock_test = convert_to_int(raw_test_data["CENBLOCK"])
sl_test = convert_to_int(raw_test_data["SL_Type"])
sl2_test = convert_to_int(raw_test_data["SL_Type2"])
sll_test = convert_to_int(raw_test_data["SL_Lead"])
latitude_test = convert_to_int(raw_test_data["Latitude"])
longitude_test = convert_to_int(raw_test_data["Longitude"])

print("other data converted")

print("assemble training set")

train_x_converted = np.hstack((homestead_train, home_sev_train, ward_train, precinct_train, centract_train, cenblock_train, sl_train, sl2_train, sll_train, latitude_train,longitude_train, use_type_train)).reshape(len(homestead_train), 12)

test_x_converted = np.hstack((homestead_test, home_sev_test, ward_test, precinct_test, centract_test, cenblock_test, sl_test, sl2_test, sll_test, latitude_test, longitude_test, use_type_test)).reshape(len(homestead_test), 12)

print("homestead_train: ", len(homestead_train))
print("train_x_converted", train_x_converted.shape)
print("train_y_converted:", len(train_y_converted))

print("create log reg object")

logreg = linear_model.LogisticRegression()
clf = DecisionTreeClassifier()

print("fit data")

logreg.fit(train_x_converted, train_y_converted)
clf.fit(train_x_converted, train_y_converted)

print("fit finished")

print("predict")

result = logreg.predict(test_x_converted)
result_tree = clf.predict(test_x_converted)
result_bagging = bagging_ensemble(train_x_converted, train_y_converted, test_x_converted, 8, n_clf = 15)

count_true_train = 0
for i in range(0, len(train_y_converted)):
    if train_y_converted[i] == 1:
        count_true_train = count_true_train + 1

count_true = 0
for i in range(0, len(result)):
    if result[i] == 1:
        count_true = count_true + 1

count_true_tree = 0
for i in range(0, len(result_tree)):
    if result_tree[i] == 1:
        count_true_tree = count_true_tree + 1

count_true_bagging = 0
for i in range(0, len(result_bagging)):
    if result_bagging[i] == 1:
        count_true_bagging = count_true_bagging + 1

total_sample = len(test_sample_id)
print("in given sample", count_true_train / len(train_y_converted) * 100, "percent predicted to be true")
print("there are ", count_true / total_sample * 100, "percent predicted to be true in log reg")
print("there are ", count_true_tree / total_sample * 100, "percent predicted to be true in tree")
print("there are ", count_true_bagging / total_sample * 100, "percent predicted to be true in bagging")

print("predict finished, output DataFrame")

output = {'sample_id': test_sample_id,
            'Lead_gt_15': result_bagging}

df = pd.DataFrame(output, columns = ['sample_id', 'Lead_gt_15'])

df.to_csv('output.csv', index=False)
