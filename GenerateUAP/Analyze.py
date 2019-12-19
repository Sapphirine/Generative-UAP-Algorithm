"""
This code was used to generate the Google Cloud Big Query Data

Requirements:
    1) Set the three directories (lines 19, 20, and 21) accordingly
    2) ImageNet data should be from the ILSVRC2012 validation set
"""

import sys
import pickle
import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import MakeDeepFakeUap

def build_csv():
    imagenet_directory =     '...'
    deepfake_uap_diractory = '...'
    gen_alg_uap_directory = '...'
      
    f,  _ = MakeDeepFakeUap.getModelAndGradient()
    
    def predict(image):
        image = image.reshape((1, 224, 224, 3))
        return np.argmax(f(image), axis=1)[0]
    
    deepfake_uaps = {}
    
    for path, subdirs, files in os.walk(deepfake_uap_diractory):
        for name in files:
            deepfake_uaps[name] = pickle.load(open(os.path.join(path, name), 'rb'))[0]     
        break
    
    gen_alg_uaps = {}
    
    for path, subdirs, files in os.walk(gen_alg_uap_directory):
        for name in files:
            gen_alg_uaps[name] = pickle.load(open(os.path.join(path, name), 'rb'))[0]     
        break
            
           
    with open('Analysis.csv', 'w') as target:           
        
        target.write('image_id,perturbation_id,type,predicted_class,predicted_v_class\n')
            
        for i in range(50000):
            
            file_id = str(i + 1).zfill(8)
            path_img = os.path.join(imagenet_directory, 'ILSVRC2012_val_' + file_id + '.JPEG')
            img_size=(256,256)
            crop_size=(224,224)
            
            image = imread(path_img, pilmode='RGB')
            
            img = 255 * np.array(imresize(image,img_size))
            
            image = img.astype('float32')
    
            # We normalize the colors (in RGB space) with the empirical means on the training set
            image[:, :, 0] -= 123.68
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 103.939
    
            image = image[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];

            print(file_id)

            predicted_class = predict(image)
                
            for v_name, v in deepfake_uaps.items():
                predicted_v_class = predict(image + v)
                
                target.write(name + ',' + v_name + ',DF,' + str(predicted_class) + ',' + str(predicted_v_class) + '\n')

            for v_name, v in gen_alg_uaps.items():
                predicted_v_class = predict(image + v)
                
                target.write(name + ',' + v_name + ',GEN,' + str(predicted_class) + ',' + str(predicted_v_class) + '\n')

            
build_csv()        