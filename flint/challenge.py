import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

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

home_sev_train = convert_to_int(raw_train_data["HomeSEV"])
home_sev_test = convert_to_int(raw_test_data["HomeSEV"])

ward_train = convert_to_int(raw_train_data["Ward"])
precinct_train = convert_to_int(raw_train_data["PRECINCT"])
centract_train = convert_to_int(raw_train_data["CENTRACT"])
cenblock_train = convert_to_int(raw_train_data["CENBLOCK"])
sl_train = convert_to_int(raw_train_data["SL_Type"])
sl2_train = convert_to_int(raw_train_data["SL_Type2"])
sll_train = convert_to_int(raw_train_data["SL_Lead"])

ward_test = convert_to_int(raw_test_data["Ward"])
precinct_test = convert_to_int(raw_test_data["PRECINCT"])
centract_test = convert_to_int(raw_test_data["CENTRACT"])
cenblock_test = convert_to_int(raw_test_data["CENBLOCK"])
sl_test = convert_to_int(raw_test_data["SL_Type"])
sl2_test = convert_to_int(raw_test_data["SL_Type2"])
sll_test = convert_to_int(raw_test_data["SL_Lead"])

print("other data converted")

print("assemble training set")

train_x_converted = np.hstack((homestead_train, home_sev_train, ward_train, precinct_train, centract_train, cenblock_train, sl_train, sl2_train, sll_train))

test_x_converted = np.hstack((homestead_test, home_sev_test, ward_test, precinct_test, centract_test, cenblock_test, sl_test, sl2_test, sll_test))

print("create log reg object")

logreg = linear_model.LogisticRegression()

print("fit data")

logreg.fit(train_x_converted, train_y_converted)

print("fit finished")

print("predict")

result = logreg.predict(test_x_converted)

print("predict finished, output DataFrame")

output = {'sample_id': test_sample_id,
            'Lead_gt_15': result}

df = pd.DataFrame(output, columns = ['sample_id', 'Lead_gt_15'])

df.to_csv('output.csv')
