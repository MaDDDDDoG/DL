import urllib.request
import os

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)


import numpy
import pandas as pd
from sklearn import preprocessing

numpy.random.seed(10)

all_df = pd.read_excel(filepath)
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]

Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 30, 1, 0, 100.0000, 'S'])

JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=cols)
all_df = pd.concat([all_df, JR_df])


def PreprocessDate(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])
    ndarray = x_OneHot_df.values
    print(ndarray.shape)

    Label = ndarray[:, 0]
    Features = ndarray[:, 1:]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label


msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:', len(all_df), 'train:', len(train_df), 'test:', len(test_df))

train_Features, train_Label = PreprocessDate(train_df)
test_Features, test_Label = PreprocessDate(test_df)


from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=train_Features, y=train_Label,
                          validation_split=0.1, epochs=30, batch_size=30, verbose=2)

from plot import show_train_history

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x=test_Features, y=test_Label)
print(scores[1])

all_Features, Label = PreprocessDate(all_df)
all_probability = model.predict(all_Features)

pd = all_df
pd.insert(len(all_df.columns), 'probability', all_probability)
print(pd[-2:])

print(pd[(pd['survived'] == 0) & (pd['probability'] > 0.9)])
