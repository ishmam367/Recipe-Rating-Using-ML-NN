import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.model_selection import cross_val_score, cross_val_predict,cross_validate
from imblearn.over_sampling import SMOTE
import pickle 

# "../data.csv" for original dataset
dataset = pd.read_csv('data2.csv')

dataset = dataset.astype({"title":'object',"preptime":'int64',"cooktime":'int64',"totaltime":'int64',"servings":'float64',"ingredients":'int64',"steps":'int64',"calories":'float64',"rating":'float64', "RatingClass":'category'})


#Split to train and test sets
str_split = StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=0)

for train_index, test_index in str_split.split(dataset[['preptime','cooktime','totaltime','servings','ingredients','steps','calories']],dataset[['RatingClass']].values.ravel()):
    train = dataset.loc[train_index]
    test = dataset.loc[test_index]

#skipped : 

X_train = train[['preptime','cooktime','totaltime','servings','ingredients','steps', 'calories']]
y_train = train[['RatingClass']].values.ravel()

X_test = test[['preptime','cooktime','totaltime','servings','ingredients','steps','calories']]
y_test = test[['RatingClass']].values.ravel()

# sm = SMOTE(random_state=0)
# X_train, y_train = sm.fit_resample(X_train1,y_train1)

# Chi2 test: Cooktime, Totaltime, Ingredients, Steps, calories 
# F test: totaltime,servings,ingredients,steps,calories
# Mutual info test: preptime,totaltime,servings,ingredients,steps
# Recursive Feature Elimination:
# LR: preptime, cooktime, servings, ingredients, steps
# RF: cooktime,totaltime,servings,ingredients, calories
# RF SelectFromModel: totaltime, ingredients, calories


# from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,f_classif,mutual_info_classif,RFE,SelectFromModel

# # model = LogisticRegression(solver='saga',multi_class='multinomial',max_iter=10000,penalty='l1')
# model = RandomForestClassifier(random_state=0,n_estimators=50)
# model.fit(X_train,y_train)
# # sel = SelectKBest(mutual_info_classif,k=5)
# # sel = RFE(estimator=model,n_features_to_select=5,step=1)
# sel = SelectFromModel(estimator=model,prefit=True,threshold='mean')
# X_train_rv = sel.transform(X_train)#,y_train)
# print(sel.get_support())



# #classifiers in increasing cross validated training accuracy
NB = GaussianNB()#MultinomialNB(alpha=0.01)
LR = LogisticRegression(multi_class='ovr', max_iter=1000,C=1.0,random_state=42)
KNN = KNeighborsClassifier(n_neighbors=9,p=3) #k=7,p=1 optimum
DT = DecisionTreeClassifier(criterion='entropy',splitter='best', max_depth=7,random_state=42)
SVM = SVC(C=1,gamma='scale', kernel='rbf') #low performing, should be discarded.
RF = RandomForestClassifier(n_estimators=1000,criterion='gini')#,bootstrap=True) #Best classifier. 70~% F1
ZeroR = DummyClassifier(strategy='most_frequent')



# scores = cross_validate(NB, X_train, y_train, cv=10, return_train_score=True)
# print('NB', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())

# # , return_estimator=True)
# scores = cross_validate(LR, X_train, y_train, cv=10, return_train_score=True)
# print('LR', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())

# # , return_estimator=True)
# scores = cross_validate(KNN, X_train, y_train, cv=10, return_train_score=True)
# print('KNN', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())

# # , return_estimator=True)
# scores = cross_validate(DT, X_train, y_train, cv=10, return_train_score=True)
# print('DT', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())

# # , return_estimator=True)
# scores = cross_validate(RF, X_train, y_train, cv=10, return_train_score=True)
# print('RF', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())

# # , return_estimator=True)
# scores = cross_validate(SVM, X_train, y_train, cv=10, return_train_score=True)
# print('SVM: ', scores['train_score'].mean(), scores['train_score'].std(
# ), scores['test_score'].mean(), scores['test_score'].std())





# NBscore=cross_val_score(NB, X_train, y_train, cv=10)#, scoring='f1_micro')
# LRscore=cross_val_score(LR, X_train, y_train, cv=10)#, scoring='f1_micro')
# KNNscore=cross_val_score(KNN, X_train, y_train, cv=10)#, scoring='f1_micro')
# DTscore=cross_val_score(DT, X_train, y_train, cv=10)#, scoring='f1_micro')
# RFscore=cross_val_score(RF, X_train, y_train, cv=10)#, scoring='f1_micro')
# SVMscore=cross_val_score(SVM, X_train, y_train, cv=10)#, scoring='f1_micro')

# print("Cross-Validation: \n")
# print("NB Accuracy: %0.2f (+/- %0.2f)" % (NBscore.mean(), NBscore.std() * 2))
# print("LR Accuracy: %0.2f (+/- %0.2f)" % (LRscore.mean(), LRscore.std() * 2))
# print("KNN Accuracy: %0.2f (+/- %0.2f)" % (KNNscore.mean(), KNNscore.std() * 2))
# print("DT Accuracy: %0.4f (+/- %0.4f)" % (DTscore.mean(), DTscore.std() * 2))
# print("RF Accuracy: %0.2f (+/- %0.2f)" % (RFscore.mean(), RFscore.std() * 2))
# print("SVM Accuracy: %0.2f (+/- %0.2f)" % (SVMscore.mean(), SVMscore.std() * 2))
print()

#TRAINING
NB.fit(X_train, y_train)
LR.fit(X_train, y_train)
KNN.fit(X_train, y_train)
DT.fit(X_train, y_train)
RF.fit(X_train, y_train)
SVM.fit(X_train, y_train)
ZeroR.fit(X_train, y_train)


# #Training Accuracy Prediction
# NBpred = NB.predict(X_train)
# LRpred = LR.predict(X_train)
# KNNpred = KNN.predict(X_train)
# DTpred = DT.predict(X_train)
# RFpred = RF.predict(X_train)
# SVMpred = SVM.predict(X_train)
# ZeroRpred = ZeroR.predict(X_train)



# print("Training Accuracy:\n")

# print("Naive Bayes:")
# print(metrics.classification_report(y_train,NBpred))
# print(metrics.confusion_matrix(y_train,NBpred,labels=["Average","Good","Excellent"]))

# print("Logistic Regression:")
# print(metrics.classification_report(y_train,LRpred))
# print(metrics.confusion_matrix(y_train,LRpred,labels=["Average","Good","Excellent"]))

# print("KNN:")
# print(metrics.classification_report(y_train,KNNpred))
# print(metrics.confusion_matrix(y_train,KNNpred,labels=["Average","Good","Excellent"]))

# print("Decision Tree:")
# print(metrics.classification_report(y_train,DTpred))
# print(metrics.confusion_matrix(y_train,DTpred,labels=["Average","Good","Excellent"]))

# print("Random Forest:")
# print(metrics.classification_report(y_train,RFpred))
# print(metrics.confusion_matrix(y_train,RFpred,labels=["Average","Good","Excellent"]))

# print("Support Vector Machines:")
# print(metrics.classification_report(y_train,SVMpred))
# print(metrics.confusion_matrix(y_train,SVMpred,labels=["Average","Good","Excellent"]))

# print("ZeroR:")
# print("Accuracy:",metrics.accuracy_score(y_train,ZeroRpred))
# print(metrics.confusion_matrix(y_train,ZeroRpred,labels=["Average","Good","Excellent"]))



# NBpred2 = NB.predict(X_test)
# LRpred2 = LR.predict(X_test)
# KNNpred2 = KNN.predict(X_test)
# DTpred2 = DT.predict(X_test)
# RFpred2 = RF.predict(X_test)
# SVMpred2 = SVM.predict(X_test)
# ZeroRpred2 = ZeroR.predict(X_test)


# # print(plot_tree(DT))


# print()

# print("Testing Accuracy:\n")

# print("Naive Bayes:")
# print(metrics.classification_report(y_test,NBpred2))
# print(metrics.confusion_matrix(y_test,NBpred2,labels=["Average","Good","Excellent"]))

# print("Logistic Regression:")
# print(metrics.classification_report(y_test,LRpred2))
# print(metrics.confusion_matrix(y_test,LRpred2,labels=["Average","Good","Excellent"]))

# print("KNN:")
# print(metrics.classification_report(y_test,KNNpred2))
# print(metrics.confusion_matrix(y_test,KNNpred2,labels=["Average","Good","Excellent"]))

# print("Decision Tree:")
# print(metrics.classification_report(y_test,DTpred2))
# print(metrics.confusion_matrix(y_test,DTpred2,labels=["Average","Good","Excellent"]))

# print("Random Forest:")
# print(metrics.classification_report(y_test,RFpred2))
# print(metrics.confusion_matrix(y_test,RFpred2,labels=["Average","Good","Excellent"]))

# print("Support Vector Machines:")
# print(metrics.classification_report(y_test,SVMpred2))
# print(metrics.confusion_matrix(y_test,SVMpred2,labels=["Average","Good","Excellent"]),end='\t')

# print("ZeroR:")
# print("Accuracy:",metrics.accuracy_score(y_test,ZeroRpred2),end='\t')

# print()

#ensemble:

# estimators = []
# estimators.append(("NB",NB))
# estimators.append(("LR",LR))
# estimators.append(("DT",DT))
# estimators.append(("RF",RF))
# estimators.append(("KNN", KNN))
# estimators.append(("SVM",SVM))
# #estimators.append(("ZeroR",ZeroR))

# ensemble = VotingClassifier(estimators,voting='hard')
# # ensemble = BaggingClassifier(base_estimator=SVM)
# ensemble.fit(X_train, y_train)

# print("Ensemble Learning:")

# print("Training:")
# ensemblepred = ensemble.predict(X_train)
# print(metrics.classification_report(y_train,ensemblepred))
# print(metrics.confusion_matrix(y_train,ensemblepred,labels=["Average","Good","Excellent"]))


# print("Testing:")
# ensemble2pred = ensemble.predict(X_test)
# print(metrics.classification_report(y_test,ensemble2pred))
# print(metrics.confusion_matrix(y_test,ensemble2pred,labels=["Average","Good","Excellent"]))


# def classify(noOfServings, noOfIngredients, noOfInstructions):
#     \""" Include integers for Servings yielded by the recipe,
#     its number of ingredients, and 
#     the number of preparation steps it needs. \"""
    
#     df = pd.DataFrame([[noOfServings, noOfIngredients, noOfInstructions]])

#     prediction = ensemble.predict(df)
#     return prediction



# print(classify(3,9,18))
# print(classify(1,1,1))
# print(classify(80,15,208))
# print(classify(12,14,15))





# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# export_graphviz(DT, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = X_train.columns ,class_names=['Excellent','Good', 'Average','Poor'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('recipe.png')
# Image(graph.create_png()) 



#PLOTS

import matplotlib.pyplot as plt


# # data to plot
n_groups = 7

#TESTING             NB, LR, KNN,DT, RF, SVM,ENSEMBLE
means_accuracy =  (51, 51, 48, 51, 50, 50, 51)
means_precision = (37, 34, 34, 41, 40, 34, 34)
means_recall =    (37, 37, 34, 36, 36, 35, 36)
means_f1 =        (36, 35, 34, 35, 36, 33, 35)

#TRAINING
# means_accuracy =  (50, 52, 61, 54, 100, 51, 62)
# means_precision = (38, 35, 52, 68, 100, 35, 71)
# means_recall =    (37, 37, 44, 39, 100, 36, 45)
# means_f1 =        (37, 36, 44, 38, 100, 34, 44)


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.5

rects1 = plt.bar(index, means_accuracy, bar_width,
alpha=opacity,
color='b',
label='Accuracy')

rects2 = plt.bar(index + bar_width, means_precision, bar_width,
alpha=opacity,
color='g',
label='Precision')

rects3 = plt.bar(index + bar_width + bar_width, means_recall, bar_width,
alpha=opacity,
color='y',
label='Recall')

rects4 = plt.bar(index + bar_width + bar_width + bar_width, means_f1, bar_width,
alpha=opacity-0.2,
color='r',
label='F1')

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Scores by models')
plt.xticks(index + bar_width, ('Naive Bayes', 'Logistic Regression', 'K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 'Ensemble'))
plt.legend()

# plt.tight_layout()
plt.show()
