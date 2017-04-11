""""
~XGBoost implementation with pandas~
========================================================
TO RUN ON LOCAL MACHINE
- ENSURE PYTHON 3.5 IS INSTALLED WITH THE FOLLOWING DEPENDANCIES: PANDAS SKLEARN NUMPY GIT XGBOOST
- SET FILE: CSV FILE WITH Y AS BINARY VARIABLE [TO PREDICT] X AS NUMERIC LAST COLUMN AND NO INDEX COLUMN 
- SET CLASSIFICATION VARIABLE Y
- RUN IN CMD FROM SAME DIRECTORY AS THE FILE
____________________________________________________________________________________________________________
************************************************************************************************************

XGBoost WINDOWS INSTALLATION GUIDE 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To install XGboost on Windows you must use a compiler
First, go to the directory where you want to save XGBoost code by typing the cd command in the bash terminal.  
I used the following.

     $ cd /c/Users/IBM_ADMIN/code/

Then download XGBoost by typing the following commands. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    $ git clone --recursive https://github.com/dmlc/xgboost
    $ cd xgboost
    $ git submodule init
    $ git submodule update

Next step is to build XGBoost on your machine, i.e. compile the code we just downloaded.  
For this we need a full fledged 64 bits compiler provided with MinGW-W64.
I installed it from here:
``````````````````````````
http://iweb.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe

One installed add the file to the Path environment variables (See My Computer: Advanced Settings)
On my machine I found it here:
```````````````````````````````
C:\Program Files\mingw-w64\x86_64-6.3.0-posix-seh-rt_v5-rev1\mingw64\bin

We can now build XGBoost.  We first go back to the directory where we downloaded it:
Then we add a shortcut to make our lives easier

     $ cd /c/Users/IBM_ADMIN/code/xgboost
     $ set-alias make='mingw32-make'
     
Run this then to compile (:

    $ cd dmlc-core
    $ make -j4
    $ cd ../rabit
    $ make lib/librabit_empty.a -j4
    $ cd ..
    $ cp make/mingw64.mk config.mk
    $ make -j4

Now from its directory install with python
[Anaconda3] C:\Users\IBM_ADMIN\code\xgboost\python-package>python setup.py install

now open python from console to check it works
>>> import xgboost

Original Source:
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

"""

import pandas as pd
import numpy as np
from numpy import nan
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
===============================================
SET FILE AND CLASSIFICATION VARIABLE Y BELOW!!!
===============================================
"""

file='indexes_cut.csv'
Y_col = 37 
unused_cols=[0]


# If the csv has named columns I would use the column name list [pd.DataFrame.columns]
# also make sure predictor is on the end and named 'pred'

datasetdf=pd.DataFrame(pd.read_csv(file, delimiter=",")) #1 Change this file!
feat_names = list(datasetdf.columns)
del feat_names[unused_cols]
dataset = datasetdf.as_matrix() #XGBoost Requires pandas DataFrames as numpy matrixes

X = dataset[:,1:y_col-1] #predictors
Y = dataset[:,y_col-1] #response



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=14)
#i wonder if you could simulate random seed and see if it converged

model = XGBClassifier(n_estimators=10, missing=nan)
model = model.fit(X_train, y_train)


def get_xgb_imp(xgb, feat_names): #This function extracts the features
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



yt_list = y_test.tolist()

n=0
for i in yt_list:
    if i == predictions[i]:
        n +=1
    else:
        pass


percento = n/len(y_test)*100
print(percento, "% Accuracy")



