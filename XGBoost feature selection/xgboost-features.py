""""
~XGBoost implementation with pandas~


"""

import pandas as pd
import numpy as np
from numpy import nan
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# If the csv has named columns I would use the column name list [pd.DataFrame.columns]
# also make sure predictor is on the end and named 'pred'

datasetdf=pd.DataFrame(pd.read_csv('indexes_cut.csv', delimiter=","))
feat_names = list(datasetdf.columns)
del feat_names[0]
dataset = datasetdf.as_matrix() #XGBoost Requires pandas DataFrames as numpy matrixes

X = dataset[:,1:36] #predictors
Y = dataset[:,36] #response



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=14)
#i wonder if you could simulate random seed and see if it converged

model = XGBClassifier(n_estimators=10, missing=nan)
model = model.fit(X_train, y_train)


def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    weights =  []
    total = sum(total)
    for v, w in imp_dict.items():
        if v != "pred" and w!= 0.0: #drop pred label and 0.0 weighted features
            weights.append(float(w))
            n = w / total * 100
            print(v, "~", "f-score: ", w, ", total weight:" "{0:.2f}".format(round(n, 2)))
        else:
            pass
    names = [v for v, w in imp_dict.items()]
    return zip(names, weights)




#make predictions from test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
extracted_features = get_xgb_imp(model, feat_names) #run feature extraction and populate this list

print(model)

#ValueError: Can't handle mix of unknown and binary
#print(type(y_test))



yt_list = y_test.tolist()

n=0
for i in yt_list:
    if i == predictions[i]:
        n +=1
    else:
        pass


percento = n/len(y_test)*100
print(percento, "% Accuracy")



