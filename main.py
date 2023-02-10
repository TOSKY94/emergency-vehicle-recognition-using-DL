from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import json
import pickle

#load models
model = load_model('base_model.h5')
vgg_model = load_model('base_model.h5')

labels = ['emergency', 'not emergency']

#initialize fastapi class
app = FastAPI()

@app.post("/emergency_vehicle/")
async def emergency_vehicle(file: UploadFile):
    data = await file.read()
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resize_img = tf.image.resize(img, [150,150])
    original_dim = img.shape

    if original_dim!= resize_img.shape:
        img = resize_img

    prediction = model.predict(np.array([img])/255)
    vgg_prediction = vgg_model.predict(np.array([img])/255)

    index = np.argmax(prediction)
    vgg_index = np.argmax(vgg_prediction)

    return {"filename": file.filename,"original dimension": str(original_dim), "prediction":labels[index], "vgg prediction":labels[vgg_index], "resized image dim": str(resize_img.shape)}
