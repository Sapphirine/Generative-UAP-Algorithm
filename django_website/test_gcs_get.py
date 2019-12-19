# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:04:29 2019

@author: Kermit
"""
import numpy as np
import io
import os
import pickle
from google.cloud import storage
from imageio import imread
import matplotlib.pyplot as plt

import pandas_gbq
from google.oauth2 import service_account

# For GC Bucket Access
storage_client = storage.Client.from_service_account_json('Big Data Analytics-a54db9c973fd.json')
bucket = storage_client.get_bucket('jda2167-project')

# For GBQ
credentials = service_account.Credentials.from_service_account_file('Big Data Analytics-a54db9c973fd.json')
pandas_gbq.context.credentials = credentials
pandas_gbq.context.project = 'jda2167-project'

def get(path):
    blob = bucket.blob(path)
    
    return io.BytesIO(blob.download_as_string())

def get_image(path):
    img = imread(get(path), pilmode = 'RGB')
    img_size=(256,256)
    crop_size=(224,224)
    img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];

    return img

def get_v(path):
    raw_v = get(path)
    
    v = pickle.load(raw_v)
    
    return v[0]

def apply_v(image, v):
    
    clipped_v = np.clip(image+v, 0, 255) - np.clip(image, 0, 255)
    
    return (image + clipped_v).astype(np.uint8)


def get_image_classes(image_id, v_id):
    SQL = ""
    df = pandas_gbq.read_gbq(SQL)
    
    return df


def test():

    image = get_image('data/Images/ILSVRC2012_val_00000001.JPEG')
    v = get_v('data/GenAlgUap/rand_conv_224x224_iter45_id00814799.p')
    v_image = apply_v(image, v)
    
    plt.imshow(image)
    plt.show()
    
    plt.imshow(v.astype(np.uint8))
    plt.show()
    
    plt.imshow(v_image)
    plt.show()