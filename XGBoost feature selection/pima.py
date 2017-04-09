""""
~XGBoost implementation with pandas~
install XGBoost using:
pip install git+https://github.com/dmlc/xgboost.git
you MAY need to use a compiler on windows... Goodluck with that
---------------------------
Originally from http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn
--Using pima indians dataset
--Fixed code to use sklearn interface for XGBClassifier
--Added Feature Weights function printing f-test and weight on total

"""

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feat_names = ['var1','var2','var3','var4','var5', 'var6', 'var7', 'var8', 'pred']
# If the csv has named columns I would use the column name list [pd.DataFrame.columns]
# also make sure predictor is on the end and named 'pred'

datasetdf=pd.DataFrame(pd.read_csv('pima-indians-diabetes.csv', delimiter=",", names=feat_names))
dataset = datasetdf.as_matrix() #XGBoost Requires pandas DataFrames as numpy matrixes

X = dataset[:,0:8] #predictors
Y = dataset[:,8] #response



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=14)
#i wonder if you could simulate random seed and see if it converged

model = XGBClassifier(n_estimators=10)
model = model.fit(X_train, y_train)
print(model)

def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    weights =  []
    total = sum(total)
    for v, w in imp_dict.items():
        if v != "pred": #lazy hack to stop from printing predictor and 0 (it will)
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

#evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%"% (accuracy*100.0))