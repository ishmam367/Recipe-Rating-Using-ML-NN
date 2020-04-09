import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle 

# "../data.csv" for original dataset
dataset = pd.read_csv('data.csv')

dataset= dataset.astype({'Servings':'int64', 'Ingredients':'int64','Instructions':'int64','RatingClass':'category'})


#Split to train and test sets
str_split = StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=0)

for train_index, test_index in str_split.split(dataset[['Servings','Ingredients','Instructions']],dataset[['RatingClass']].values.ravel()):
    train = dataset.loc[train_index]
    test = dataset.loc[test_index]



X_train = train[['Servings','Ingredients','Instructions']]
y_train = train[['RatingClass']].values.ravel()

X_test = test[['Servings','Ingredients','Instructions']]
y_test = test[['RatingClass']].values.ravel()

print(dataset.shape)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


#classifiers in increasing cross validated training accuracy
NB = MultinomialNB(alpha=1)
LR = LogisticRegressionCV(solver='lbfgs', multi_class='auto', max_iter=1000,Cs=10,cv=10)
KNN = KNeighborsClassifier(n_neighbors=7,p=1) #k=7,p=1 optimum
DT = DecisionTreeClassifier(criterion='entropy', max_depth=4)
SVM = SVC(C=0.7,gamma='auto', kernel='sigmoid',decision_function_shape='ovr') #low performing, should be discarded.
RF = RandomForestClassifier(n_estimators=10,criterion='entropy',bootstrap=True) #Best classifier. 70~% F1
ZeroR = DummyClassifier(strategy='most_frequent')


NBscore=cross_val_score(NB, X_train, y_train, cv=10)#, scoring='f1_micro')
LRscore=cross_val_score(LR, X_train, y_train, cv=10)#, scoring='f1_micro')
KNNscore=cross_val_score(KNN, X_train, y_train, cv=10)#, scoring='f1_micro')
DTscore=cross_val_score(DT, X_train, y_train, cv=10)#, scoring='f1_micro')
RFscore=cross_val_score(RF, X_train, y_train, cv=10)#, scoring='f1_micro')
SVMscore=cross_val_score(SVM, X_train, y_train, cv=10)#, scoring='f1_micro')

print("Cross-Validation: \n")
print("NB Accuracy: %0.2f (+/- %0.2f)" % (NBscore.mean(), NBscore.std() * 2))
print("LR Accuracy: %0.2f (+/- %0.2f)" % (LRscore.mean(), LRscore.std() * 2))
print("KNN Accuracy: %0.2f (+/- %0.2f)" % (KNNscore.mean(), KNNscore.std() * 2))
print("DT Accuracy: %0.4f (+/- %0.4f)" % (DTscore.mean(), DTscore.std() * 2))
print("RF Accuracy: %0.2f (+/- %0.2f)" % (RFscore.mean(), RFscore.std() * 2))
print("SVM Accuracy: %0.2f (+/- %0.2f)" % (SVMscore.mean(), SVMscore.std() * 2))
print()

#TRAINING
NB.fit(X_train, y_train)
LR.fit(X_train, y_train)
KNN.fit(X_train, y_train)
DT.fit(X_train, y_train)
RF.fit(X_train, y_train)
SVM.fit(X_train, y_train)
ZeroR.fit(X_train, y_train)




#Training Accuracy Prediction
NBpred = NB.predict(X_train)
LRpred = LR.predict(X_train)
KNNpred = KNN.predict(X_train)
DTpred = DT.predict(X_train)
RFpred = RF.predict(X_train)
SVMpred = SVM.predict(X_train)
ZeroRpred = ZeroR.predict(X_train)


print("Training Accuracy:\n")

print("Naive Bayes:")
print(metrics.classification_report(y_train,NBpred))
print(metrics.confusion_matrix(y_train,NBpred,labels=["Average","Good","Excellent"]))

print("Logistic Regression:")
print(metrics.classification_report(y_train,LRpred,zero_division=0))
print(metrics.confusion_matrix(y_train,LRpred,labels=["Average","Good","Excellent"]))

print("KNN:")
print(metrics.classification_report(y_train,KNNpred))
print(metrics.confusion_matrix(y_train,KNNpred,labels=["Average","Good","Excellent"]))

print("Decision Tree:")
print(metrics.classification_report(y_train,DTpred))
print(metrics.confusion_matrix(y_train,DTpred,labels=["Average","Good","Excellent"]))

print("Random Forest:")
print(metrics.classification_report(y_train,RFpred))
print(metrics.confusion_matrix(y_train,RFpred,labels=["Average","Good","Excellent"]))

print("Support Vector Machines:")
print(metrics.classification_report(y_train,SVMpred))
print(metrics.confusion_matrix(y_train,SVMpred,labels=["Average","Good","Excellent"]))

print("ZeroR:")
print("Accuracy:",metrics.accuracy_score(y_train,ZeroRpred))



NBpred2 = NB.predict(X_test)
LRpred2 = LR.predict(X_test)
KNNpred2 = KNN.predict(X_test)
DTpred2 = DT.predict(X_test)
RFpred2 = RF.predict(X_test)
SVMpred2 = SVM.predict(X_test)
ZeroRpred2 = ZeroR.predict(X_test)




print()

print("Testing Accuracy:\n")

print("Naive Bayes:")
print(metrics.classification_report(y_test,NBpred2))
print(metrics.confusion_matrix(y_test,NBpred2,labels=["Average","Good","Excellent"]))

print("Logistic Regression:")
print(metrics.classification_report(y_test,LRpred2))
print(metrics.confusion_matrix(y_test,LRpred2,labels=["Average","Good","Excellent"]))

print("KNN:")
print(metrics.classification_report(y_test,KNNpred2))
print(metrics.confusion_matrix(y_test,KNNpred2,labels=["Average","Good","Excellent"]))

print("Decision Tree:")
print(metrics.classification_report(y_test,DTpred2))
print(metrics.confusion_matrix(y_test,DTpred2,labels=["Average","Good","Excellent"]))

print("Random Forest:")
print(metrics.classification_report(y_test,RFpred2))
print(metrics.confusion_matrix(y_test,RFpred2,labels=["Average","Good","Excellent"]))

print("Support Vector Machines:")
print(metrics.classification_report(y_test,SVMpred2))
print(metrics.confusion_matrix(y_test,SVMpred2,labels=["Average","Good","Excellent"]),end='\t')

print("ZeroR:")
print("Accuracy:",metrics.accuracy_score(y_test,ZeroRpred2),end='\t')

print()

#ensemble:

estimators = []
# estimators.append(("NB",NB))
# estimators.append(("LR",LR))
estimators.append(("DT",DT))
estimators.append(("RF",RF))
estimators.append(("KNN", KNN))
# estimators.append(("SVM",SVM))
#estimators.append(("ZeroR",ZeroR))

ensemble = VotingClassifier(estimators,voting='hard')
# ensemble = BaggingClassifier(base_estimator=SVM)
ensemble.fit(X_train, y_train)

print("Ensemble Learning:")

print("Training:")
ensemblepred = ensemble.predict(X_train)
print(metrics.classification_report(y_train,ensemblepred))
print(metrics.confusion_matrix(y_train,ensemblepred,labels=["Average","Good","Excellent"]))


print("Testing:")
ensemble2pred = ensemble.predict(X_test)
print(metrics.classification_report(y_test,ensemble2pred))
print(metrics.confusion_matrix(y_test,ensemble2pred,labels=["Average","Good","Excellent"]))


def classify(noOfServings, noOfIngredients, noOfInstructions):
    """ Include integers for Servings yielded by the recipe,
    its number of ingredients, and 
    the number of preparation steps it needs. """
    
    df = pd.DataFrame([[noOfServings, noOfIngredients, noOfInstructions]])

    prediction = ensemble.predict(df)
    return prediction



print(classify(3,9,18))
print(classify(1,1,1))
print(classify(80,15,208))
print(classify(12,14,15))

"""


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = ['Servings','Ingredients','Instructions'],class_names=['Excellent','Good', 'Poor'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('recipe2.png')
Image(graph.create_png()) 
"""
