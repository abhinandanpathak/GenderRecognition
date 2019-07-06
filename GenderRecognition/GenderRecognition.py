#Audio Recording
import AudioRecording

#Audio Processing
import AudioProcessing




#GenderRecognition
import pandas as pd
from keras.models import model_from_json
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

voice=pd.read_csv("voice.csv")

json_file=open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

voice["label"] = le.fit_transform(voice["label"])

#Reading The sample Data
my_voice=pd.read_csv("my_voice.csv")

###Predicting the Gender
Gdr=loaded_model.predict_classes(my_voice)
Gdr=Gdr.ravel()
Gender= le.inverse_transform(Gdr)
print(Gender)