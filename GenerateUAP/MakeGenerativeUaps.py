"""
Implements a generative UAP algorithm that is independent of training data
(images) and models.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle

from scipy import ndimage

def apply_conv(img, conv):
    """
    Applies convolution (conv) to image, renormalizes, then returns the result
    """
    out = ndimage.convolve(img, conv, mode = 'constant')
    out *= 10.0 /np.max(out)
    return out

def gen_conv(w = 4):
    """
    Generate a w x w convolution matrix
    """
    conv = np.random.uniform(0.0, 1.0, (w,w))
        
    half_size = w**2 // 2
    
    sign_mask = np.array([-1.0] * half_size + [1.0] * half_size)
    np.random.shuffle(sign_mask)
    sign_mask = sign_mask.reshape((w,w))
    
    conv = np.multiply(conv, sign_mask)
    
    return conv.reshape((w,w,1))
    
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

def gen(width, height, c, iters = 20):
    """
    Generates:
        UAP
        A GIF of the UAP for each iteration
        Image of the UAP (png)
    """
    padding = 100
    
    padded_width = width + 2 * padding
    padded_height = height + 2 * padding
    
    v_base = np.random.uniform(-10, 10, (padded_width, padded_height))
    
    v = np.zeros((1,padded_width,padded_height,3))
    
    v[:,:,:,0] = v_base
    v[:,:,:,1] = v_base
    v[:,:,:,2] = v_base
      
    history = []
        
    for i in range(iters):
        #v = mutate(v, 0.01)
        
        v = v.reshape((padded_width,padded_height,3))
        v = apply_conv(v, c)  
        v = v.reshape((1,padded_width,padded_height,3))
        
        v_saturated = saturate(v[:,padding:width + padding, padding:height + padding,:])
        history.append(v_saturated[0].astype(np.uint8))
        
    run_id = str(np.random.randint(0, 100000000, 1)[0]).zfill(8)   
    
    plt.title(run_id)
    plt.imshow(v_saturated[0].astype(np.uint8))
    plt.show()
    
    base_name = 'rand_conv_' + str(height) + 'x' + str(width) + '_iter' + str(iters) + '_id' + run_id
    pickle.dump(v_saturated, open(base_name + '.p'.format(i), 'wb'))
    plt.imsave(base_name + '.png', v_saturated[0].astype(np.uint8))
    
    imageio.mimsave(base_name + '.gif', history)
    
    return v_saturated

# Generate a few UAPs...
for i in range(50):
    conv_size = 10
    width = 244
    height = 244    
    
    c = gen_conv(conv_size)
    v = gen(height, width, c, 45)

