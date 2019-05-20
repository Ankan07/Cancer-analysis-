# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:06:01 2019

@author: user
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn import utils
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from datetime import datetime
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_c=pd.read_csv("lungscancer.csv")
df_c.info()
#no null value

df_c.diagnosis.unique()
#considering "B" as 0,"M" as 1
d={"B":0,"M":1}
df_c.diagnosis.replace(d,inplace=True)

#all are continous columns

df_c.id.unique().shape[0] 
#dropping this 
df_c.drop("id",axis=1,inplace=True)

for col in df_c.columns.values:
    print("number of zero or negative values in column",str(col),(df_c[col]<=0).sum())

"""
number of zero or negative values in column radius_mean 0
number of zero or negative values in column texture_mean 0
number of zero or negative values in column perimeter_mean 0
number of zero or negative values in column area_mean 0
number of zero or negative values in column smoothness_mean 0
number of zero or negative values in column compactness_mean 0
number of zero or negative values in column concavity_mean 13
number of zero or negative values in column points_mean 13
number of zero or negative values in column symmetry_mean 0
number of zero or negative values in column dimension_mean 0
number of zero or negative values in column radius_se 0
number of zero or negative values in column texture_se 0
number of zero or negative values in column perimeter_se 0
number of zero or negative values in column area_se 0
number of zero or negative values in column smoothness_se 0
number of zero or negative values in column compactness_se 0
number of zero or negative values in column concavity_se 13
number of zero or negative values in column points_se 13
number of zero or negative values in column symmetry_se 0
number of zero or negative values in column dimension_se 0
number of zero or negative values in column radius_worst 0
number of zero or negative values in column texture_worst 0
number of zero or negative values in column perimeter_worst 0
number of zero or negative values in column area_worst 0
number of zero or negative values in column smoothness_worst 0
number of zero or negative values in column compactness_worst 0
number of zero or negative values in column concavity_worst 13
number of zero or negative values in column points_worst 13
number of zero or negative values in column symmetry_worst 0
number of zero or negative values in column dimension_worst 0  
"""  

for col in df_c.drop("diagnosis",axis=1).columns.values:
    if df_c[df_c[col]<=0].shape[0]>0:
        df_c.drop(df_c[df_c[col]<=0].index,inplace=True)



#outlier calculation
#IQR Calculation
def IQR(data):
    upper_quantile=data.quantile(0.75)
    lower_quantile=data.quantile(0.25)
    IQR=upper_quantile-lower_quantile
    outlier1=upper_quantile+1.5*IQR
    outlier2=lower_quantile-1.5*IQR
    return (IQR,outlier1,outlier2)

plt.figure(figsize=(20,10))
sns.heatmap(df_c.corr(),annot=True)

for col in df_c.drop("diagnosis",axis=1).columns.values:
    i,outlier1,outlier2=IQR(df_c[str(col)])
    print(df_c[df_c[str(col)]>outlier1].shape[0],"column name",str(col),"upper_outlier",outlier1,"max",df_c[str(col)].max())
    print(df_c[df_c[str(col)]<outlier2].shape[0],"column name",str(col),"lower_outlier",outlier2,"min",df_c[str(col)].min())

#bhetore boolean array return kore
"""
12 column name radius_mean upper_outlier 22.459999999999997 max 28.11
0 column name radius_mean lower_outlier 5.340000000000001 min 7.691
7 column name texture_mean upper_outlier 30.071249999999996 max 39.28
0 column name texture_mean lower_outlier 7.841250000000006 min 9.71
13 column name perimeter_mean upper_outlier 149.35749999999996 max 188.5
0 column name perimeter_mean lower_outlier 31.73750000000002 min 48.34
23 column name area_mean upper_outlier 1353.5 max 2501.0
0 column name area_mean lower_outlier -127.7000000000001 min 170.4
5 column name smoothness_mean upper_outlier 0.1335025 max 0.1634
0 column name smoothness_mean lower_outlier 0.05856249999999999 min 0.06251
18 column name compactness_mean upper_outlier 0.22658874999999998 max 0.3454
0 column name compactness_mean lower_outlier -0.029381249999999984 min 0.01938
18 column name concavity_mean upper_outlier 0.2844925 max 0.4268
0 column name concavity_mean lower_outlier -0.12128749999999998 min 0.000692
10 column name points_mean upper_outlier 0.15576374999999998 max 0.2012
0 column name points_mean lower_outlier -0.06002625 min 0.001852
14 column name symmetry_mean upper_outlier 0.24652500000000005 max 0.304
0 column name symmetry_mean lower_outlier 0.11112499999999996 min 0.1167
15 column name dimension_mean upper_outlier 0.0787125 max 0.09744
0 column name dimension_mean lower_outlier 0.04505249999999998 min 0.049960000000000004
37 column name radius_se upper_outlier 0.85825 max 2.873
0 column name radius_se lower_outlier -0.14315 min 0.1115
16 column name texture_se upper_outlier 2.4170750000000005 max 3.568
0 column name texture_se lower_outlier -0.12112500000000026 min 0.3602
37 column name perimeter_se upper_outlier 6.061249999999999 max 21.98
0 column name perimeter_se lower_outlier -1.0687499999999996 min 0.757
64 column name area_se upper_outlier 86.81375 max 542.2
0 column name area_se lower_outlier -23.516249999999992 min 6.8020000000000005
27 column name smoothness_se upper_outlier 0.01250375 max 0.03113
0 column name smoothness_se lower_outlier 0.0006957500000000002 min 0.002667
28 column name compactness_se upper_outlier 0.06093750000000002 max 0.1354
0 column name compactness_se lower_outlier -0.01466250000000001 min 0.002252
22 column name concavity_se upper_outlier 0.08297625 max 0.396
0 column name concavity_se lower_outlier -0.024793749999999996 min 0.000692
19 column name points_se upper_outlier 0.025336500000000005 max 0.05279
0 column name points_se lower_outlier -0.0024075000000000017 min 0.001852
28 column name symmetry_se upper_outlier 0.034820000000000004 max 0.07895
0 column name symmetry_se lower_outlier 0.0031199999999999978 min 0.007882
28 column name dimension_se upper_outlier 0.008022000000000001 max 0.02984
0 column name dimension_se lower_outlier -0.0012140000000000007 min 0.0008948000000000001
13 column name radius_worst upper_outlier 28.11625 max 36.04
0 column name radius_worst lower_outlier 4.066249999999998 min 8.677999999999999
5 column name texture_worst upper_outlier 42.12875 max 49.54
0 column name texture_worst lower_outlier 8.578750000000001 min 12.02
13 column name perimeter_worst upper_outlier 190.02375 max 251.2
0 column name perimeter_worst lower_outlier 21.29374999999999 min 54.49
31 column name area_worst upper_outlier 1984.6999999999998 max 4254.0
0 column name area_worst lower_outlier -356.4999999999999 min 223.6
6 column name smoothness_worst upper_outlier 0.19005000000000002 max 0.2226
0 column name smoothness_worst lower_outlier 0.07344999999999999 min 0.08125
16 column name compactness_worst upper_outlier 0.6272375 max 1.058
0 column name compactness_worst lower_outlier -0.13446249999999998 min 0.034319999999999996
12 column name concavity_worst upper_outlier 0.7828000000000002 max 1.252
0 column name concavity_worst lower_outlier -0.27480000000000016 min 0.0018449999999999999
0 column name points_worst upper_outlier 0.30930625 max 0.29100000000000004
0 column name points_worst lower_outlier -0.08044375000000001 min 0.008772
23 column name symmetry_worst upper_outlier 0.4207749999999999 max 0.6638
0 column name symmetry_worst lower_outlier 0.14897500000000008 min 0.1565
24 column name dimension_worst upper_outlier 0.12242874999999995 max 0.2075
0 column name dimension_worst lower_outlier 0.04153875000000004 min 0.05504
"""

for col in df_c.drop("diagnosis",axis=1).columns.values:
     p,q,r=IQR(df_c[str(col)])
     s=0
     l=[]
     for i in df_c.index.values:
        if df_c[str(col)][i]>q:
          s=s+1
          l.append(i)
          
     if s<7 and s!=0:
         df_c.drop(l,axis=0,inplace=True)
         print(str(col), ' ',s,l)
      
      




for col in df_c.drop("diagnosis",axis=1).columns.values:
     p,q,r=IQR(df_c[str(col)])
     s=0
     l=[]
     for i in df_c.index.values:
        if df_c[str(col)][i]>q:
            df_c.at[i,str(col)]=df_c[str(col)].mean()
            
            
            
df_c.corr()
        
      
"""
this is the function but i don't know how figsize works,it is not giving a good image
"""


     
     
mydict={}
for col in df_c.drop("diagnosis",axis=1).corr().columns.values:
    for index in df_c.drop("diagnosis",axis=1).corr().index.values:
        if df_c.corr()[str(col)][str(index)]>0.6 and df_c.corr()[str(col)][str(index)]<0.7:
            if str(col)+" "+str(index) not in mydict.keys():
              mydict[str(index)+" "+str(col)]=str(col)+" "+str(index)
print(mydict)

for keys in mydict.keys():
    x,y=keys.split(" ")
    sns.lmplot(x,y,data=df_c,fit_reg=False)
    
            
            
     
     
mydic={}
for col in df_c.drop("diagnosis",axis=1).corr().columns.values:
    c=0
    for index in df_c.drop("diagnosis",axis=1).corr().index.values:
        if df_c.corr()[str(col)][str(index)]>=0.7 and str(col)!=str(index):
              c+=1
    if c>0:
     mydic[str(col)]=c
print(mydic)

for keys in mydic.keys():
    x=mydic[keys]
    if x==1:
        df_c.drop(keys,inplace=True,axis=1)
df_c.drop("texture_mean",inplace=True,axis=1)
df_c.drop("smoothness_worst",inplace=True,axis=1)
            
            
sns.boxplot(y="radius_mean",x="diagnosis",data=df_c)
sns.boxplot(y="texture_mean",x="diagnosis",data=df_c)#further researh
sns.boxplot(y="perimeter_mean",x="diagnosis",data=df_c)
sns.boxplot(y="area_mean",x="diagnosis",data=df_c)
sns.boxplot(y="smoothness_mean",x="diagnosis",data=df_c)
sns.boxplot(y="compactness_mean",x="diagnosis",data=df_c)
sns.boxplot(y="concavity_mean",x="diagnosis",data=df_c)
sns.boxplot(y="points_mean",x="diagnosis",data=df_c)
sns.boxplot(y="symmetry_mean",x="diagnosis",data=df_c)
sns.boxplot(y="dimension_mean",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="radius_se",x="diagnosis",data=df_c)
sns.boxplot(y="texture_se",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="perimeter_se",x="diagnosis",data=df_c)
sns.boxplot(y="area_se",x="diagnosis",data=df_c)
sns.boxplot(y="smoothness_se",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="compactness_se",x="diagnosis",data=df_c)
sns.boxplot(y="concavity_se",x="diagnosis",data=df_c)
sns.boxplot(y="points_se",x="diagnosis",data=df_c)
sns.boxplot(y="symmetry_se",x="diagnosis",data=df_c)
sns.boxplot(y="dimension_se",x="diagnosis",data=df_c)
sns.boxplot(y="_se",x="diagnosis",data=df_c)


sns.violinplot(y="radius_mean",x="diagnosis",data=df_c)
sns.violinplot(y="texture_mean",x="diagnosis",data=df_c)#further researh
sns.boxplot(y="perimeter_mean",x="diagnosis",data=df_c)
sns.boxplot(y="area_mean",x="diagnosis",data=df_c)
sns.boxplot(y="smoothness_mean",x="diagnosis",data=df_c)
sns.boxplot(y="compactness_mean",x="diagnosis",data=df_c)
sns.boxplot(y="concavity_mean",x="diagnosis",data=df_c)
sns.boxplot(y="points_mean",x="diagnosis",data=df_c)
sns.boxplot(y="symmetry_mean",x="diagnosis",data=df_c)
sns.boxplot(y="dimension_mean",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="radius_se",x="diagnosis",data=df_c)
sns.boxplot(y="texture_se",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="perimeter_se",x="diagnosis",data=df_c)
sns.boxplot(y="area_se",x="diagnosis",data=df_c)
sns.boxplot(y="smoothness_se",x="diagnosis",data=df_c)#not varying
sns.boxplot(y="compactness_se",x="diagnosis",data=df_c)
sns.boxplot(y="concavity_se",x="diagnosis",data=df_c)
sns.boxplot(y="points_se",x="diagnosis",data=df_c)
sns.boxplot(y="symmetry_se",x="diagnosis",data=df_c)
sns.boxplot(y="dimension_se",x="diagnosis",data=df_c)
sns.boxplot(y="_se",x="diagnosis",data=df_c)



"""
ankan,if u modify something in the code please mention it through a comment somewhere
"""

x=df_c.drop("diagnosis",axis=1)
y=df_c.diagnosis
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.1,random_state=42)


trmodel=tree.DecisionTreeClassifier(max_depth=10)
trmodel.fit(xtrain,ytrain)
f_imp=trmodel.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":f_imp})
pdt=zzz.sort_values(by="fi",ascending=False)   
pdt
#feature selection using extratree classifier

etrmodel=ensemble.ExtraTreesClassifier()
etrmodel.fit(xtrain,ytrain)
fi=etrmodel.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":fi})
petr=zzz.sort_values(by="fi",ascending=False)
petr

#feature selection using random tree classifier
rf=ensemble.RandomForestClassifier(max_depth=10)
rf.fit(xtrain,ytrain)
fi=rf.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":fi})
prf=zzz.sort_values(by="fi",ascending=False)
prf

rfecv=feature_selection.RFECV(estimator=ensemble.ExtraTreesClassifier(),min_features_to_select=1,cv=5,scoring="recall")
rfecv.fit(xtrain,ytrain)
zzz=xtrain.columns[rfecv.get_support()]
print(zzz)




def modelstats2(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LogisticReg","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames) 



x=df_c[["points_se","area_se","concavity_se","symmetry_worst","symmetry_se","dimension_worst"]]
y=df_c["diagnosis"]

modelstats2(xtrain,xtest,ytrain,ytest)
