# gender recognition with proper neural network model
#using csv file

import pandas as pd
#reading the file
voice=pd.read_csv("voice.csv")
#using all the features to train the neural network
X=voice.iloc[:,0:20].values
#noting down thw label
y=voice.iloc[:,-1].values

#used for label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
voice["label"] = le.fit_transform(voice["label"])
le.classes_

#to convert all the features in floating point
#voice[:]=preprocessing.MinMaxScaler().fit_transform(voice)

#to find the test and training set
from sklearn.model_selection import train_test_split
train, test = train_test_split(voice, test_size=0.3)

#splitting the features and labels into x and y labels
x_train= train.iloc[:,:-1].values
y_train= train["label"].values
x_test = test.iloc[:,:-1].values
y_test = test["label"].values

from keras.models import Sequential
from keras.callbacks import History
from keras.layers import Dense

model=Sequential()
history = History()

#number of input variables =20
#first layer 
#input_dim is only for the first layer
model.add(Dense(units=11,kernel_initializer="uniform",activation="relu",input_dim=20))
#first Hidden layer
model.add(Dense(units=11,kernel_initializer='uniform',activation='relu'))
#Second Hidden
model.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#output layer
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#Running the artificial neural network
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting
model.fit(x_train,y_train,batch_size=10,epochs=80,validation_split=0.1,shuffle=2)
#to print the summary of the model
model.summary()

#to predict the model
import sklearn.metrics as metrics
y_pred=model.predict_classes(x_test)
#a=model.predict_classes(x_test)
#y_pre= np.round(y_pred)

#
print('Accuracy we are able to achieve with our ANN is',metrics.accuracy_score(y_pred,y_test)*100,'%')
#
from sklearn.metrics import classification_report
target_names = ['female', 'male']
#printing the classification report
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


#from keras.model import load_model
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

