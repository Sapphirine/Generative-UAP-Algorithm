"""
The functionality here can be used to evaluate the effect on fooling rates
after applying transforms to a UAP

Requirements:
    1) first call genFixedTestSet() to create a reusable set of images
    2) a sub folder 'gen_test' that contains pickled UAPs compatible 
       with the training images.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# Get everything needed to build a model
import MakeDeepFakeUap

def genFixedTestSet():
    """
    Generate a fixed set of 100 randomly selected images
    This supports direct comparisons between transforms
    """
    print('Loading images...')
    images = MakeDeepFakeUap.create_imagenet_npy(MakeDeepFakeUap.imagenet_directory, 100)
    
    pickle.dump(images, open('test_images.p', 'wb'))

def saturate(v):
    """
    Maps negative values to -10.0 and positive values to 10.0
    This is consistent with an L-infinity threshold of 10
    """
    shape = v.shape
    v = v.flatten()

    for i in range(v.shape[0]):
        if v[i] < 0:
            v[i] = -10.0
        elif v[i] > 0:
            v[i] = 10.0
    v = np.reshape(v, shape)
    return v
 
def asym_rotate(v):
    """
    Rotate color channels in thirds so that the relative positions are
    offset by roughly 1/3 of the height
    """
    # Keep red the same
    rotate_green = 224 // 3
    v[:,:,:,1] = np.roll(v[:,:,:,1], rotate_green, axis = 1)
    
    rotate_blue = (2 * 224) // 3
    v[:,:,:,2] = np.roll(v[:,:,:,2], rotate_blue, axis = 1)
    
    return v

def sym_rotate(v):
    """
    Rotate all color channels equally
    """
    rotate = 224 // 2
    v[:,:,:,0] = np.roll(v[:,:,:,0], rotate, axis = 1)
    v[:,:,:,1] = np.roll(v[:,:,:,1], rotate, axis = 1)
    v[:,:,:,2] = np.roll(v[:,:,:,2], rotate, axis = 1)

    return v

def color_rotate_1(v):
    """
    Swap color channels:
        red --> green
        green --> blue
        blue --> red
    """
    v = np.roll(v, 1, axis = 3)
    
    return v

def color_rotate_2(v):
    """
    Swap color channels:
        red --> blue
        green --> red
        blue --> green
    """
    v = np.roll(v, 2, axis = 3)
    
    return v

def test(transform_name = 'none', transform = lambda x : x):    
    """
    Loads pickled image data and tests fooling rate for all
    UAPs in the sub-folder 'gen_test'
    
    Saves an image of the transofrmed UAP and displays the fooling rate for 
    each individual UAP as well as their collectice average
    """
    print('Loading model...')
    f,  _ = MakeDeepFakeUap.getModelAndGradient()
    
    images = pickle.load(open('test_images.p', 'rb'))
    
    test_v_paths = []
    
    for path, subdirs, files in os.walk('gen_test'):
        for name in files:
            test_v_paths.append(path + '\\' + name)
    
    print('Testing...')
    
    fooling_rates = []
    
    for path in test_v_paths:
        v = transform(pickle.load(open(path, 'rb')))
        
        plt.imsave(transform_name + '.png', v[0].astype(np.uint8))
        
        print('Shape: {}'.format(v.shape))
        fooling_rate = MakeDeepFakeUap.foolingRate(images, f, v)
        
        fooling_rates.append(fooling_rate)
        
        print('... {}: {:.4f}'.format(path, fooling_rate))
        
    print('Mean: {}'.format(np.mean(fooling_rates)))

