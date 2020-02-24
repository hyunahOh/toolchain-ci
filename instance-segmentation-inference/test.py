'''This code is for Inference
 author : Hyunah Oh
 data : 2020.02.11
 modified by : Sangbum Kim, for API-fying
'''

import numpy as np
import cv2
import os

from keras.models import load_model
from .model import create_model
from .align import AlignDlib

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('backend/models/landmarks.dat')

#### Detection & Alignment & Normalization ####
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('backend/weights/nn4.small2.v1.h5')

#### Embedding ####
def embedding_img(img):
    img = (img / 255.).astype(np.float32)
    return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

model = load_model('backend/models/best1.h5')

#### Detection ####
def detect_img(image):
    person=['현진', '아이린', '슬기', '태연']
    a_image = align_image(image)
    a_image = (a_image / 255.).astype(np.float32)
    embedded = nn4_small2_pretrained.predict(np.expand_dims(a_image, axis=0))
    predict = model.predict_classes(embedded)[0]
    return person[predict]

if __name__ == '__main__':
    detect_img()
