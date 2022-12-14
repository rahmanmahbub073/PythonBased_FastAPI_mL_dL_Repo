# -*- coding: utf-8 -*-
"""credit-card-fraud-detection-final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oyItWr5OuINDDVms6e7VuxiwppfujnUZ

## Credit card Fraud Detection (Handling Imbalanced dataset using ML)

# Loading Dataset
"""

#pip install pandas

import pandas as pd
df=pd.read_csv('/content/sample_data/creditcard.csv')
df.head()

df.shape

df['Class'].value_counts()

#statistical info
df.describe()

#datatype info
df.info()

"""# Preprocessing the Dataset"""

#check the null values
df.isnull().sum()

#df.dropna()

import matplotlib.pyplot as plt

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()  
    plt.show()
    
def draw_roc_curve_RF(model, x_test_roc, y_test_roc, label = None):
    y_scores = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test_roc, y_scores[:,1])
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)    
    plt.grid(True)  
    


draw_histograms(df,df.columns,8,4)

#### Independent and Dependent Features
X=df.drop("Class",axis=1)
y=df.Class

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import seaborn as sns
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(df.corr())
ax.set_title('Correlation')

"""## Sklearn Library installing"""

#pip install scikit-learn



"""# Logistic Regression"""



"""## Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.model_selection import GridSearchCV

def draw_roc_curve(model, x_train_roc, y_train_roc, label = None):
    y_scores = cross_val_predict(model, x_train_roc, y_train_roc, cv=3,method="decision_function")
    fpr, tpr, thresholds = roc_curve(y_train_roc, y_scores)
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)    
    plt.grid(True)

log_class=LogisticRegression()
grid={'C':10.0 **np.arange(-2,3),'penalty':['l1','l2']}
cv=KFold(n_splits=5,random_state=None,shuffle=False)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75)
X_train = X_train.dropna()
y_train = y_train.dropna()
X_test = X_test.dropna()
y_test = y_test.dropna()

clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

import seaborn as sns

# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve(clf, X_train, y_train, label = None)

2*0.69*0.72/(0.72+0.69)

"""# Logistic Regression with Under Sampling"""

from imblearn.under_sampling import RandomUnderSampler

u_sample_lr = RandomUnderSampler(sampling_strategy=0.7)
X_train_under_lr,y_train_under_lr = u_sample_lr.fit_resample(X_train , y_train)

clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train_under_lr,y_train_under_lr)

y_pred=clf.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve(clf, X_train_under_lr, y_train_under_lr, label = None)

"""# Logistic Regression with Over Sampling"""

from imblearn.over_sampling import RandomOverSampler

o_sample_lr = RandomOverSampler(sampling_strategy=0.8)
X_train_over_lr,y_train_over_lr = o_sample_lr.fit_resample(X_train , y_train)

clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train_over_lr,y_train_over_lr)

y_pred=clf.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve(clf, X_train_over_lr, y_train_over_lr, label = None)





"""# Random Forest Classifier :"""



"""## Random Forest Classifier :"""

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=1)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=1)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# Random Forest with Under Sampling

Let's find out the best strategy for sampling
"""

import time
t0=time.time()
k=0.5
while k<=1:
    u_sample = RandomUnderSampler(sampling_strategy=k)
    X_train_under,y_train_under = u_sample.fit_resample(X_train,y_train)
    c0=X_train_under[y_train_under==0]
    c1=X_train_under[y_train_under==1]
    print(len(c0),len(c1))
    classifier=RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=1)
    classifier.fit(X_train_under,y_train_under)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

"""So k = 0.7 is the best strategy because we want to keep the false negative as minimum as possible."""

u_sample_rf = RandomUnderSampler(sampling_strategy=0.7)
X_train_under_rf,y_train_under_rf = u_sample_rf.fit_resample(X_train,y_train)
    
classifier=RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=1)
classifier.fit(X_train_under_rf,y_train_under_rf)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)



"""# Random Forest with SMOTE"""

from imblearn.over_sampling import SMOTE
import time

t0=time.time()
k=0.5
while k<=1:
    o_sample = SMOTE(sampling_strategy=k)
    X_train_over,y_train_over = o_sample.fit_resample(X_train,y_train)
    c0=X_train_over[y_train_over==0]
    c1=X_train_over[y_train_over==1]
    print(len(c0),len(c1))
    classifier=RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train_over,y_train_over)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

"""Here the best strategy is k = 0.79"""

o_sample_rf = SMOTE(sampling_strategy=0.79, random_state=4002)
X_train_over_rf,y_train_over_rf = o_sample_rf.fit_resample(X_train,y_train)
    
classifier=RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train_over_rf,y_train_over_rf)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# XGBoost"""

from xgboost import XGBClassifier
classifier=XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

from sklearn import metrics
def buildROC(model, y_test,y_pred, label = None):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.figure(figsize=(8, 6)) 
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)    
    plt.grid(True)

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# XGBoost with Undersampling"""

#pip install xgboost

from xgboost import XGBClassifier

t0=time.time()
k=0.5
while k<=1:
    u_sample = RandomUnderSampler(sampling_strategy=k)
    X_train_under,y_train_under = u_sample.fit_resample(X_train,y_train)
    c0=X_train_under[y_train_under==0]
    c1=X_train_under[y_train_under==1]
    print(len(c0),len(c1))
    classifier=XGBClassifier()
    classifier.fit(X_train_under,y_train_under)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

u_sample_xg = RandomUnderSampler(sampling_strategy=0.7)
X_train_under_xg,y_train_under_xg = u_sample_xg.fit_resample(X_train,y_train)
    
classifier=XGBClassifier()
classifier.fit(X_train_under_xg,y_train_under_xg)

y_pred=classifier.predict(X_test.values)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

X_test = X_test.values

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# XGBoost with SMOTE"""

t0=time.time()
k=0.5
while k<=1:
    o_sample = SMOTE(sampling_strategy=k)
    X_train_over,y_train_over = o_sample.fit_resample(X_train,y_train)
    c0=X_train_over[y_train_over==0]
    c1=X_train_over[y_train_over==1]
    print(len(c0),len(c1))
    classifier=XGBClassifier()
    classifier.fit(X_train_over,y_train_over)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

o_sample_xg = SMOTE(sampling_strategy=0.5, random_state=4003)
X_train_over_xg,y_train_over_xg = o_sample_xg.fit_resample(X_train,y_train)
    
classifier=XGBClassifier()
classifier.fit(X_train_over_xg,y_train_over_xg)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""#

# Support Vector Machine(SVM)
"""

from sklearn import svm

#Create a svm Classifier
classifier = svm.SVC(kernel='linear',probability=True) # Linear Kernel
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

# confusion Matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# SVM with Under Sampling"""

from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
import time
t0=time.time()
k=0.5
while k<=1:
    u_sample = RandomUnderSampler(sampling_strategy=k)
    X_train_under,y_train_under = u_sample.fit_resample(X_train,y_train)
    c0=X_train_under[y_train_under==0]
    c1=X_train_under[y_train_under==1]
    print(len(c0),len(c1))
    classifier = svm.SVC(kernel='linear',probability=True) # Linear Kernel
    classifier.fit(X_train_under,y_train_under)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

from imblearn.under_sampling import RandomUnderSampler
u_sample_rf = RandomUnderSampler(sampling_strategy=0.7)
X_train_under_svm,y_train_under_svm = u_sample_rf.fit_resample(X_train,y_train)
    
classifier = svm.SVC(kernel='linear',probability=True) # Linear Kernel
classifier.fit(X_train_under_svm,y_train_under_svm)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)

"""# SVM with Over Sampling"""

import time
from sklearn import svm
from imblearn.over_sampling import RandomOverSampler

t0=time.time()
k=0.5
while k<=1:
    o_sample = RandomOverSampler(sampling_strategy=k)
    X_train_over,y_train_over = o_sample.fit_resample(X_train,y_train)
    c0=X_train_over[y_train_over==0]
    c1=X_train_over[y_train_over==1]
    print(len(c0),len(c1))
    classifier = svm.SVC(kernel='linear',probability=True) # Linear Kernel
    classifier.fit(X_train_over,y_train_over)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

from sklearn import svm
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train) 
X_train_svm = scaling.transform(X_train) 
X_test_svm = scaling.transform(X_test)

o_sample_svm = RandomOverSampler(sampling_strategy=0.8)
X_train_over_svm,y_train_over_svm = o_sample_svm.fit_resample(X_train_svm , y_train)
    
classifier = svm.SVC(kernel='linear',probability=True) # Linear Kernel
classifier.fit(X_train_over_svm,y_train_over_svm)

y_pred=classifier.predict(X_test_svm)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)



"""# Using ROC & AUC Curve to select the best model"""

clf_lr=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')

model_lr = clf_lr.fit(X_train,y_train)
probs_lr = model_lr.predict_proba(X_test)[:, 1]

model_rf = RandomForestClassifier(n_estimators = 10).fit(X_train,y_train)
probs_rf = model_rf.predict_proba(X_test)[:, 1]

model_xg = XGBClassifier().fit(X_train,y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

model_svm = svm.SVC(kernel='linear',probability=True).fit(X_train,y_train)
probs_svm = model_svm.predict_proba(X_test)[:, 1]

model_lr_under = clf_lr.fit(X_train_under_lr,y_train_under_lr)
probs_lr_under = model_lr_under.predict_proba(X_test)[:, 1]

model_rf_under = RandomForestClassifier(n_estimators = 10).fit(X_train_under_rf,y_train_under_rf)
probs_rf_under = model_rf_under.predict_proba(X_test)[:, 1]

model_xg_under = XGBClassifier().fit(X_train_under_xg,y_train_under_xg)
probs_xg_under = model_xg_under.predict_proba(X_test)[:, 1]

model_svm_under = svm.SVC(kernel='linear',probability=True).fit(X_train_under_svm,y_train_under_svm)
probs_svm_under = model_svm_under.predict_proba(X_test)[:, 1]

model_lr_over = clf_lr.fit(X_train_over_lr,y_train_over_lr)
probs_lr_over = model_lr_over.predict_proba(X_test)[:, 1]

model_rf_over = RandomForestClassifier(n_estimators = 10).fit(X_train_over_rf,y_train_over_rf)
probs_rf_over = model_rf_over.predict_proba(X_test)[:, 1]

model_xg_over = XGBClassifier().fit(X_train_over_xg,y_train_over_xg)
probs_xg_over = model_xg_over.predict_proba(X_test)[:, 1]

model_svm_over = svm.SVC(kernel='linear',probability=True).fit(X_train_over_svm,y_train_over_svm)
probs_svm_over = model_svm_over.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, probs_lr)

auc_rf = roc_auc_score(y_test, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_svm = roc_auc_score(y_test, probs_xg)
fpr_svm, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_lr_under = roc_auc_score(y_test, probs_lr_under)
fpr_lr_under, tpr_lr_under, thresholds_lr_under = roc_curve(y_test, probs_lr_under)

auc_rf_under = roc_auc_score(y_test, probs_rf_under)
fpr_rf_under, tpr_rf_under, thresholds_rf_under = roc_curve(y_test, probs_rf_under)

auc_xg_under = roc_auc_score(y_test, probs_xg_under)
fpr_xg_under, tpr_xg_under, thresholds_xg_under = roc_curve(y_test, probs_xg_under)

auc_svm_under = roc_auc_score(y_test, probs_svm_under)
fpr_svm_under, tpr_svm_under, thresholds_svm_under = roc_curve(y_test, probs_svm_under)

auc_lr_over = roc_auc_score(y_test, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, probs_lr)

auc_rf_over = roc_auc_score(y_test, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

auc_xg_over = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_svm_over = roc_auc_score(y_test, probs_svm_over)
fpr_svm_over, tpr_svm_over, thresholds_svm_over = roc_curve(y_test, probs_svm_over)

plt.figure(figsize=(12, 7))
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression) = {auc_lr:.2f}')
plt.plot(fpr_rf, tpr_rf, label=f'AUC (Random Forests) = {auc_rf:.2f}')
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
plt.plot(fpr_svm_under, tpr_svm_under, label=f'AUC (SVM) = {auc_svm:.2f}')
plt.plot(fpr_lr_under, tpr_lr_under, label=f'AUC (Logistic Regression-under) = {auc_lr_under:.2f}')
plt.plot(fpr_rf_under, tpr_rf_under, label=f'AUC (Random Forests-under) = {auc_rf_under:.2f}')
plt.plot(fpr_xg_under, tpr_xg_under, label=f'AUC (XGBoost-under) = {auc_xg_under:.2f}')
plt.plot(fpr_svm_under, tpr_svm_under, label=f'AUC (SVM-under) = {auc_svm_under:.2f}')
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression-over) = {auc_lr_over:.2f}')
plt.plot(fpr_rf, tpr_rf, label=f'AUC (Random Forests-over) = {auc_rf_over:.2f}')
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost-over) = {auc_xg_over:.2f}')
plt.plot(fpr_svm_over, tpr_svm_over, label=f'AUC (SVM-over) = {auc_svm_over:.2f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();

"""# From the above AUC Curve and ROC value, we can say that the XgBoost Algorithm model is the best trained model for this dataset!"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, roc_curve, roc_auc_score

"""# Corelation Matrix"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import seaborn as sns
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(df.corr())
ax.set_title('Correlation')

New_column=[ f'V{i}'for i in range(1,29)]
y_column="Class"
x_df=df[New_column]
y_df=df[y_column]

train_x,val_x,train_y,val_y=train_test_split(x_df,y_df,test_size=0.3)
val_x,test_x,val_y,test_y=train_test_split(val_x,val_y,test_size=0.2)
val_x.shape

"""# General Model"""

def model():
    inputs = tf.keras.Input(shape=(28),name='Input')
    x = Dense(32, activation=tf.nn.relu,name='hidden_layer_1')(inputs)
    
    
    x=tf.keras.layers.Dropout(0.45)(x)
    output= Dense(1,activation='sigmoid',name='Output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(
    optimizer='SGD', loss='binary_crossentropy', metrics='acc'
                                                        )
    return model

model_1=model()

Epoch=8
history=model_1.fit(train_x,train_y,batch_size=64,epochs=Epoch,validation_data=(val_x,val_y))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss , 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc=0)
plt.figure()

plt.figure()

prediction=model_1.predict(test_x)>=0.5
cm=tf.math.confusion_matrix(
    test_y, prediction
)

TP = tf.linalg.diag_part(cm)
precision = TP / tf.reduce_sum(cm, axis=0)

TP = tf.linalg.diag_part(cm)
recall = TP / tf.reduce_sum(cm, axis=1)
print(f'Precision is {precision[0]} and recall is {recall[0]}')

def Evaluation(x,y):
    his=model_1.evaluate(test_x, test_y, batch_size=16, verbose=1)
    print('Model Loss is {} and prediction Accuracy is {}'.format(his[0],his[1]))

Evaluation(test_x,test_y)





from imblearn.under_sampling import RandomUnderSampler

u_sample_lr = RandomUnderSampler(sampling_strategy=0.7)
X_train_under_lr,y_train_under_lr = u_sample_lr.fit_resample(train_x , train_y)
u_sample_lr = RandomUnderSampler(sampling_strategy=0.6)
x_val_under,y_val_under=u_sample_lr.fit_resample(val_x,val_y)

y_val_under.value_counts()

inputs = tf.keras.Input(shape=(28),name='Input_o')
x = Dense(64, activation=tf.nn.relu,name='hidden_layer_1_0',kernel_regularizer='l2')(inputs)

x=tf.keras.layers.Dropout(0.35)(x)
output= Dense(1,activation='sigmoid',name='Output')(x)

model_2= tf.keras.Model(inputs=inputs, outputs=output)
model_2.summary()
model_2.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics='acc')

Epoch=1000
history2=model_2.fit(X_train_under_lr,y_train_under_lr,batch_size=16,epochs=Epoch,validation_data=(x_val_under,y_val_under))

acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss , 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc=0)
plt.figure()

plt.figure()




y_pred=model_2.predict(test_x)>=0.5
cm=confusion_matrix(test_y,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(test_y,y_pred))

print(classification_report(test_y,y_pred))



from imblearn.over_sampling import RandomOverSampler

o_sample_lr = RandomOverSampler(sampling_strategy=0.8)
X_train_over_lr,y_train_over_lr = u_sample_lr.fit_resample(train_x , train_y)
o_sample_lr = RandomOverSampler(sampling_strategy=0.7)
x_val_over,y_val_over=o_sample_lr.fit_resample(val_x,val_y)

y_train_over_lr.value_counts()

inputs = tf.keras.Input(shape=(28),name='Input_o')
x = Dense(64, activation=tf.nn.relu,name='hidden_layer_1_0',kernel_regularizer='l2')(inputs)

x=tf.keras.layers.Dropout(0.35)(x)
output= Dense(1,activation='sigmoid',name='Output')(x)

model_3= tf.keras.Model(inputs=inputs, outputs=output)
model_3.summary()
model_3.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics='acc')

Epoch=500
history3=model_3.fit(X_train_over_lr,y_train_over_lr,batch_size=64,epochs=Epoch,validation_data=(x_val_over,y_val_over))

acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss , 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc=0)
plt.figure()

plt.figure()




y_pred=model_3.predict(test_x)>=0.5
cm=confusion_matrix(test_y,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(test_y,y_pred))

print(classification_report(test_y,y_pred))

"""# Please add ANN ROC curve with below this code"""

clf_lr=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')

model_lr = clf_lr.fit(X_train,y_train)
probs_lr = model_lr.predict_proba(X_test)[:, 1]

model_rf = RandomForestClassifier(n_estimators = 10).fit(X_train,y_train)
probs_rf = model_rf.predict_proba(X_test)[:, 1]

model_xg = XGBClassifier().fit(X_train,y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

model_svm = svm.SVC(kernel='linear',probability=True).fit(X_train,y_train)
probs_svm = model_svm.predict_proba(X_test)[:, 1]

model_lr_under = clf_lr.fit(X_train_under_lr,y_train_under_lr)
probs_lr_under = model_lr_under.predict_proba(X_test)[:, 1]

model_rf_under = RandomForestClassifier(n_estimators = 10).fit(X_train_under_rf,y_train_under_rf)
probs_rf_under = model_rf_under.predict_proba(X_test)[:, 1]

model_xg_under = XGBClassifier().fit(X_train_under_xg,y_train_under_xg)
probs_xg_under = model_xg_under.predict_proba(X_test)[:, 1]

model_svm_under = svm.SVC(kernel='linear',probability=True).fit(X_train_under_svm,y_train_under_svm)
probs_svm_under = model_svm_under.predict_proba(X_test)[:, 1]

model_lr_over = clf_lr.fit(X_train_over_lr,y_train_over_lr)
probs_lr_over = model_lr_over.predict_proba(X_test)[:, 1]

model_rf_over = RandomForestClassifier(n_estimators = 10).fit(X_train_over_rf,y_train_over_rf)
probs_rf_over = model_rf_over.predict_proba(X_test)[:, 1]

model_xg_over = XGBClassifier().fit(X_train_over_xg,y_train_over_xg)
probs_xg_over = model_xg_over.predict_proba(X_test)[:, 1]

probs_ann_u=model_2.predict(X_test)

probs_ann_o=model_3.predict(X_test)

auc_lr = roc_auc_score(y_test, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, probs_lr)

auc_rf = roc_auc_score(y_test, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_svm = roc_auc_score(y_test, probs_xg)
fpr_svm, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_lr_under = roc_auc_score(y_test, probs_lr_under)
fpr_lr_under, tpr_lr_under, thresholds_lr_under = roc_curve(y_test, probs_lr_under)

auc_rf_under = roc_auc_score(y_test, probs_rf_under)
fpr_rf_under, tpr_rf_under, thresholds_rf_under = roc_curve(y_test, probs_rf_under)

auc_xg_under = roc_auc_score(y_test, probs_xg_under)
fpr_xg_under, tpr_xg_under, thresholds_xg_under = roc_curve(y_test, probs_xg_under)

auc_svm_under = roc_auc_score(y_test, probs_svm_under)
fpr_svm_under, tpr_svm_under, thresholds_svm_under = roc_curve(y_test, probs_svm_under)

auc_lr_over = roc_auc_score(y_test, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, probs_lr)

auc_rf_over = roc_auc_score(y_test, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

auc_xg_over = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

auc_ann_u = roc_auc_score(y_test, probs_ann_u)
fpr_ann_u, tpr_aan_u, thresholds_ann_u = roc_curve(y_test, probs_ann_u)

auc_ann_o = roc_auc_score(y_test, probs_ann_o)
fpr_ann_o, tpr_aan_o, thresholds_ann_o = roc_curve(y_test, probs_ann_o)

plt.figure(figsize=(12, 7))
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression) = {auc_lr:.2f}')
plt.plot(fpr_rf, tpr_rf, label=f'AUC (Random Forests) = {auc_rf:.2f}')
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
plt.plot(fpr_svm_under, tpr_svm_under, label=f'AUC (SVM) = {auc_svm:.2f}')
plt.plot(fpr_lr_under, tpr_lr_under, label=f'AUC (Logistic Regression-under) = {auc_lr_under:.2f}')
plt.plot(fpr_rf_under, tpr_rf_under, label=f'AUC (Random Forests-under) = {auc_rf_under:.2f}')
plt.plot(fpr_xg_under, tpr_xg_under, label=f'AUC (XGBoost-under) = {auc_xg_under:.2f}')
plt.plot(fpr_svm_under, tpr_svm_under, label=f'AUC (SVM-under) = {auc_svm_under:.2f}')
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression-over) = {auc_lr_over:.2f}')
plt.plot(fpr_rf, tpr_rf, label=f'AUC (Random Forests-over) = {auc_rf_over:.2f}')
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost-over) = {auc_xg_over:.2f}')
plt.plot(fpr_ann_u, tpr_aan_u, label=f'AUC (ANN under) = {aauc_ann_u :.2f}')
plt.plot(fpr_ann_o,tpr_aan_o, label=f'AUC (ANN over) = {aauc_ann_o :.2f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();

"""# Extra """



clf_lr=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
model_lr = clf_lr.fit(X_train_over_lr,y_train_over_lr)
probs_lr = model_lr.predict_proba(X_test)[:, 1]

model_rf = RandomForestClassifier(n_estimators = 10).fit(X_train_over_rf,y_train_over_rf)
probs_rf = model_rf.predict_proba(X_test)[:, 1]

model_xg = XGBClassifier().fit(X_train_over_xg,y_train_over_xg)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, probs_lr)

auc_rf = roc_auc_score(y_test, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression) = {auc_lr:.2f}')
plt.plot(fpr_rf, tpr_rf, label=f'AUC (Random Forests) = {auc_rf:.2f}')
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();



"""# Random Forest with SMOTE"""

from imblearn.over_sampling import SMOTE
import time

t0=time.time()
k=0.5
while k<=1:
    o_sample = SMOTE(sampling_strategy=k)
    X_train_over,y_train_over = o_sample.fit_resample(X_train,y_train)
    c0=X_train_over[y_train_over==0]
    c1=X_train_over[y_train_over==1]
    print(len(c0),len(c1))
    classifier=RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train_over,y_train_over)
    #adap=RandomForestClassifier(n_estimators=10)
    #adap=AdaBoostClassifier(base_estimator=clfsm,n_estimators=10)
    #adap.fit(Xtrsm,ytrsm)
    pred=classifier.predict(X_test)
    print("K=",k)
    print("\nCONFUSION METRICS\n",confusion_matrix(y_test,pred))
    print("\nCLASSIFICATION REPORT\n",classification_report(y_test,pred))
    tn=time.time()-t0
    print(tn)
    print("-"*100)
    k=k+0.1

"""Here the best strategy is k = 0.79"""

o_sample_rf = SMOTE(sampling_strategy=0.79, random_state=4002)
X_train_over_rf,y_train_over_rf = o_sample_rf.fit_resample(X_train,y_train)
    
classifier=RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train_over_rf,y_train_over_rf)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

draw_roc_curve_RF(classifier, X_test, y_test, label = None)