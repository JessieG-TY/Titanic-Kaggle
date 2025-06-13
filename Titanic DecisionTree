import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test  = pd.read_csv('/kaggle/input/titanic/test.csv')
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Title']

train['Sex'] = train['Sex'].map({'male':0, 'female' : 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train = train.dropna(subset=['Age'])

age_median = train['Age'].median()
test['Age'] = test['Age'].fillna(age_median)

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

fare_median = train['Fare'].median()
test['Fare'] = test['Fare'].fillna(fare_median)

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare',
    'Countess': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare', 'Sir': 'Rare',
    'Capt': 'Rare'
}
train['Title'] = train['Title'].map(title_map)
test['Title'] = test['Title'].map(title_map)

title_encoding = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
train['Title'] = train['Title'].map(title_encoding)
test['Title'] = test['Title'].map(title_encoding)

train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].fillna(0)

X = train[features]
y = train['Survived']

X_test = test[features]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X,y)

predictions = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived':     predictions
})
submission.to_csv('submission.csv', index=False)
