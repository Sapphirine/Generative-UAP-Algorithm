"""
Source code adapted from https://github.com/LTS4/universal

Requirements:
    1) set data_directory (line 24)
    2) set imagenet_directory (line 25))
    
Note, the data directory should container:
    1) tensorflow_inception_graph.pb - required to build the pretrained model
    2) labels.txt - class lables (indexed by row starting at 1)
    
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import sys
import pickle
from imageio import imread
from skimage.transform import resize as imresize
import datetime

data_directory =     '...'
imagenet_directory = '...'

# swap comments to support gpu if available
# device = '\gpu:0'
device = '/cpu:0'
num_classes = 2

# Quick and dirty - support printing progress to one line instead of 50k!

def printLine(line):
    print('>> ' + line)

def printInPlace(line):
    line = '>> ' + line
    print(('\b' * len(printInPlace.last)) + line, end = '')
    printInPlace.last = line
printInPlace.last = ''

def printInPlaceStart():
    printInPlace.last = ''

def printInPlaceStop():
    print()

# =============================================================================

def foolingRate(dataset, f, v, batch_size = 100):
    """
    Rerutns the fooling rate induced by the perturbation v (using the 
    model f)
    """
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    printInPlaceStart()

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
        
        dataset_perturbed = dataset[m:M, :, :, :] + v
        
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed), axis=1).flatten()
        printInPlace('Calculating fooling rate... batch {}/{}'.format(ii + 1, num_batches))

    printInPlaceStop()
    # Compute the fooling rate
    return float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
 
def getModelAndGradient():
    """
    Isolated for use by other python scripts
    Loads and returns both the model and the gradient required for the
    DeepFool algorithm
    """
    with tf.device(device):
        persisted_sess = tf.compat.v1.Session()
        inception_model_path = data_directory + 'tensorflow_inception_graph.pb'
   
        model = inception_model_path
        
        # Load the Inception model
        with gfile.GFile(model, 'rb') as f:
        #with pickle.load(open(data_directory + 'model.p', 'rb')) as f:
            pickle.dump( f, open(data_directory + "model.p", "wb" ) )
            graph_def = tf.compat.v1.GraphDef() 
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        printLine('Computing feedforward function...')
        
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})
             
        # TODO: Optimize this construction part!
        printLine('Compiling the gradient tensorflow functions. This might take some time...')
        y_flat = tf.reshape(persisted_output, (-1,))
        
        inds = tf.compat.v1.placeholder(tf.int32, shape=(num_classes,))  
        dydx = jacobian(y_flat,persisted_input,inds)
           
        printLine('Computing gradient function...')
        def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)    

    return (f, grad_fs)

def deepfool(image, f, grads, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i))

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        gradients = np.asarray(grads(pert_image,I))
        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image
        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # compute new label
        #if loop_i % 4 == 0:
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))

    r_tot = (1+overshoot)*r_tot

    return r_tot 



def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset, f, grads, run_id, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """

    v = 0 #np.random.uniform(-10, 10, (224, 224, 3))
    
    """
    v = np.array(imread(data_directory + 'v0.png', pilmode='RGB')).astype(np.float32)
    for i in range(224):
        for j in range(224):
            for c in range(3):
                if v[i][j][c] > 255.0 / 2:
                    v[i][j][c] -= 255.0
    """
                
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    itr = 0 
    
    while fooling_rate < 1-delta and itr < max_iter_uni:
        
        t0 = datetime.datetime.now()
        
        # Shuffle the dataset
        np.random.shuffle(dataset)

        printLine(' === Starting pass number {} ==='.format(itr))

        # Go through the data set and compute the perturbation increments sequentially

        printInPlaceStart()
        for k in range(0, num_images):            
            cur_img = dataset[k:(k+1), :, :, :]
            
            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):
                printInPlace('Image {}, pass {}, time {}'.format(k, itr, datetime.datetime.now() - t0))
                
                # Compute adversarial perturbation
                dr = deepfool(cur_img + v, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                             
                # Make sure it converged...       
                v = v + dr

                # Project on l_p ball
                v = proj_lp(v, xi, p)
        
        printInPlaceStop()
        itr = itr + 1

        # Perturb the dataset with computed perturbation
        
        fooling_rate = foolingRate(dataset, f, v)
        
        printLine('Fooling Rate = {}'.format(fooling_rate))

        t1 = datetime.datetime.now()
        
        pickle_name = 'v_id' + run_id + '_itr' + str(itr).zfill(3) + '_fr' + str(int(100 * fooling_rate)) + '.p'
        
        printLine('Saving UAP...')

        pickle.dump(v, open(data_directory + pickle_name, 'wb'))
        
        printLine('Iteration took {}'.format(t1 - t0))

    return v


def preprocess_image_batch(image_path, img_size=None, crop_size=None, color_mode="rgb"):
    img_list = []

    img = imread(image_path, pilmode='RGB')
    
    if img_size:
        img = 255 * np.array(imresize(img,img_size))

    img = img.astype('float32')
    
    # We normalize the colors (in RGB space) with the empirical means on the training set
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    
    if crop_size:
        img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :];

    img_list.append(img)

    img_batch = np.stack(img_list, axis=0)

    return img_batch


def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy


def create_imagenet_npy(path_train_imagenet, len_batch=1000):
    """
    Modified to support the naming format used for the ILSVRC2012 validation
    data
    """

    # path_train_imagenet = '/datasets2/ILSVRC2012/train';

    sz_img = [224, 224]
    num_channels = 3
    #num_classes = 1000

    im_array = np.zeros([len_batch] + sz_img + [num_channels], dtype=np.float32)
    
    ids = np.random.randint(0, 50000, len_batch)
    
    for index, iid in enumerate(ids):
        file_id = str(iid + 1).zfill(8)

        path_img = path_train_imagenet + 'ILSVRC2012_val_' + file_id + '.JPEG'
        image = preprocess_image_batch(path_img, img_size=(256,256), crop_size=(224,224), color_mode="rgb")
        im_array[index:(index+1), :, :, :] = image

            
    return im_array
            

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

  

def generate(train_size, run_id):
    """
    Generate a UAP!
    """
    printLine('run id = {}'.format(run_id))
    
    f, grad_fs = getModelAndGradient()

    X = create_imagenet_npy(imagenet_directory, train_size)

    # Running universal perturbation
    v = universal_perturbation(X, f, grad_fs, run_id, delta=0.2,num_classes=num_classes)
   
    return v

# Continuously generate UAPs
while(True):
    run_id = str(np.random.randint(0, 100000000, 1)[0]).zfill(8)
    v = generate(10000, run_id)
        
