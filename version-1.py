# -*- coding: utf-8 -*-
"""
Spyder Editor

Author : Neeraj Sahani
"""


from sklearn.utils import shuffle
from tensorflow.keras import layers, models
import numpy as np, pandas as pd, os
import cv2


path = 'E:/Python/Notes/Data Science/My Notes/Facial keypoints'

os.chdir('D:/Datasets')

dataset = pd.read_csv('facialkeypoints/training.csv')

dataset.Image=dataset.Image.apply(lambda ar: np.fromstring(ar, sep=' '))

dataset = dataset.dropna()

X = np.vstack(dataset['Image'].values) 

X = X.astype(np.float32)/255.

X = X.reshape(-1, 96, 96, 1) 

y=dataset.iloc[:, :-1].values
y = y.astype(np.float32)

X, y = shuffle(X, y)

model = models.Sequential()
model.add(layers.Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.1))

model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Convolution2D(30, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(30))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history=model.fit(X, y, batch_size=200, epochs=1000, validation_split=0.2, verbose=1)

def save_model(path=''):
        if path == '':
            try:
                os.mkdir('Model')
                path='Model'
            except:
                pass
            
        json = model.to_json()
        with open(path+'/facial_keypoints.json', 'w') as file:
            file.write(json)
                
            model.save_weights(path+'/facial_keypoints.h5')
            print("Saved")

def load_model(path=''):
    json = open(path+'/facial_keypoints.json')
    model = models.model_from_json(json.read())
    json.close()
    model.load_weights(path+'/facial_keypoints.h5')
    return model
    
model = load_model(path)
#save_model('E:/Python/Notes/Data Science/My Notes/Facial keypoints')



def output():
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.25, 6)
        for (x, y, w, h) in faces:
        # Grab the face
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+h]
            original = color_face.shape[:2]
            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255
            color_normalized = color_face / 255
            color_normalized = cv2.resize(color_normalized, (96, 96))
            resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA).reshape((1, 96, 96, 1))
            # Resize it to 96x96 to match the input format of the model
            keypoints = model.predict(resized)
            resized = resized.reshape((96, 96))
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((int(co+10), int(keypoints[0][1::2][i]+15)))
            
            for keypoint in points:
                cv2.circle(color_normalized, keypoint, 1, (0,255,0), 1)
            
            resized = cv2.resize(color_normalized, original)*255
            frame[y:y+h, x:x+w] = resized
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('win', frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
output()
