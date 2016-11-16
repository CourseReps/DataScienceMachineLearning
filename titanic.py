
# coding: utf-8

# In[682]:

cd Desktop


# In[683]:

import csv as csv
import pandas as pd
import numpy as np
df = pd.read_csv('train.csv', header=0)
df

    


# In[684]:

df['Age'].dropna()


# In[685]:

df['AgeFill'] = df['Age']
df.head()


# In[686]:

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)





# In[687]:

median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages


# In[688]:

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]


# In[689]:

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
df.head()


# In[690]:

df = df.drop(['PassengerId', 'Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df.head()


# In[691]:

test_df = pd.read_csv('test.csv', header=0)
test_df['Gender'] = test_df['sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['AgeFill'] = test_df['age']
median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = test_df[(test_df['Gender'] == i) &                               (test_df['pclass'] == j+1)]['age'].dropna().median()
 
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        test_df.loc[ (test_df.age.isnull()) & (test_df.Gender == i) & (test_df.pclass == j+1),                'AgeFill'] = median_ages[i,j]
test_df['AgeIsNull'] = pd.isnull(test_df.age).astype(int)
test_df['FamilySize'] = test_df['sibsp'] + test_df['parch']
test_df['Age*Class'] = test_df.AgeFill * test_df.pclass
test_df = test_df.drop(['name', 'age', 'sex', 'ticket', 'cabin', 'embarked'], axis=1)
test_df.head()




# In[692]:

train_data = df.values
test_data = test_df.values
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

