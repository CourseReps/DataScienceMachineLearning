import numpy as np
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm,metrics
import pandas as pd


training = pd.read_csv("train.csv")
final_test = pd.read_csv("test.csv")

y_train = training["Survived"]
X_train = training.drop(["Survived","Name","Ticket","PassengerId"],1)

def imputeThings(X):
    X_train = X
    #impute some things
    X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())
    X_train["Pclass"] = X_train["Pclass"].fillna(X_train["Pclass"].median())
    X_train["Family"] = X_train["SibSp"] + X_train["Parch"] + 1
    X_train = X_train.drop(["SibSp","Parch"],1)
    X_train["Fare"] = X_train["Fare"].fillna(X_train["Fare"].median())

    #impute Cabins
    df = X_train[~X_train["Cabin"].isnull()]
    df.ix[df["Cabin"].str.contains("A"),"Cabin"] = '1'
    df.ix[df["Cabin"].str.contains("B"),"Cabin"] = '2'
    df.ix[df["Cabin"].str.contains("C"),"Cabin"] = '3'
    df.ix[df["Cabin"].str.contains("D"),"Cabin"] = '4'
    df.ix[df["Cabin"].str.contains("E"),"Cabin"] = '5'
    df.ix[df["Cabin"].str.contains("F"),"Cabin"] = '6'
    df.ix[df["Cabin"].str.contains("G"),"Cabin"] = '7'
    df.ix[df["Cabin"].str.contains("T"),"Cabin"] = '0'
    df["Cabin"] = df["Cabin"].astype(int)
    X_train["Cabin"] = df["Cabin"]
    X_train["Cabin"] = X_train["Cabin"].fillna(X_train["Cabin"].median())

    #impute Sex
    X_train["Sex"][X_train["Sex"] == "male"] = 0
    X_train["Sex"][X_train["Sex"] == "female"] = 1
    X_train["Sex"] = X_train["Sex"].fillna(0)


    #impute Embarked
    X_train["Embarked"][X_train["Embarked"] == "S"] = 0
    X_train["Embarked"][X_train["Embarked"] == "C"] = 1
    X_train["Embarked"][X_train["Embarked"] == "Q"] = 2
    X_train["Embarked"] = X_train["Embarked"].fillna(0)
    return X_train

acc = []
X_train = imputeThings(X_train)
final_test = imputeThings(final_test)

for i in range (1,len(X_train.columns.values)+1):
    X_new =  SelectKBest(chi2,k=i).fit_transform(X_train,y_train)
    skf = StratifiedKFold(n_splits=10)
    scores = []
    for train, cv in skf.split(X_new, y_train):
        X, X_cv, y, y_cv = X_new[train], X_new[cv], y_train[train], y_train[cv]
        #clf = RandomForestClassifier(n_estimators=25,max_depth=5,min_samples_split=2,random_state=0)
        clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1,max_depth=1, random_state=0)
        #clf = MLPClassifier(solver='lbfgs',alpha=1,hidden_layer_sizes=(15,2),random_state=1)
        clf.fit(X,y)
        scores.append(clf.score(X_cv,y_cv))
    acc.append(np.mean(scores))

featureCount = acc.index(max(acc))+1
accuracy = max(acc)
print()
print()
print("Accuracy = {}", accuracy)
print("Number of features = {}", featureCount)

PassengerId = np.array(final_test["PassengerId"]).astype(int)
final_test = final_test.drop(["Name","Ticket","PassengerId"],1)

bestFeatures =  SelectKBest(chi2,k=featureCount)
X_new = bestFeatures.fit_transform(X_train,y_train)
indi = bestFeatures.get_support(indices=True)

clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1,max_depth=1, random_state=0)
clf.fit(X_new,y_train)

prediction = clf.predict(final_test[final_test.columns[indi]])

my_solution = pd.DataFrame(prediction,PassengerId,columns=["Survived"])
my_solution.to_csv("my_solution.csv",index_label=["PassengerId"])
