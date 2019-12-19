"""
Requirements:
    1) User must set auth_path (to google JSON authorization file)
    2) Project Name
    3) Project ID
    4) Modify bucket paths as needed fro calls to get_image and get_v
"""

from django.shortcuts import render

import numpy as np
import io
import os
import pickle
from google.cloud import storage
from imageio import imread, imwrite

import pandas_gbq
from google.oauth2 import service_account

auth_path = '...'

if not os.path.exists(auth_path):
    raise Exception("The JSON authorization path does not exist: '{}'".format(auth_path))

# For GC Bucket Access
storage_client = storage.Client.from_service_account_json(auth_path)

# Replace with your project name
bucket = storage_client.get_bucket('jda2167-project')

# For GBQ
credentials = service_account.Credentials.from_service_account_file(auth_path)
pandas_gbq.context.credentials = credentials

# Replace with your project ID
pandas_gbq.context.project = 'optical-torch-252501'

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

def img_id_to_image_name(image_id):
    file_id = str(image_id).zfill(8)
    return 'ILSVRC2012_val_' + file_id + '.JPEG'

def get_labels():
    labels = {}
    with open('static/labels.txt', 'r') as source:
        for index, row in enumerate(source):
            labels[index + 1] = row.split(',')[0]
            
    return labels
        
def fooling_diversity(request):
    data = {}
    
    SQL = """
    SELECT
      Variety,
      COUNT(*) as Total
    From
    (
    SELECT
      image_id,
      COUNT(DISTINCT predicted_v_class) as Variety
    FROM Project.UapEval
    WHERE
      type = "GEN"
      AND predicted_class != predicted_v_class
    GROUP BY
      image_id
    ORDER BY
      Variety
    )
    GROUP BY
      Variety
    ORDER BY
      Variety    
    """ 

    df = pandas_gbq.read_gbq(SQL)
    
    bars1 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['Variety']
       barmap['Value'] = row['Total'] / 50000.0
       
       bars1.append(barmap)

    SQL = """
    SELECT
      Variety,
      COUNT(*) as Total
    From
    (
    SELECT
      image_id,
      COUNT(DISTINCT predicted_v_class) as Variety
    FROM Project.UapEval
    WHERE
      type = "DF"
      AND predicted_class != predicted_v_class
    GROUP BY
      image_id
    ORDER BY
      Variety
    )
    GROUP BY
      Variety
    ORDER BY
      Variety    
    """ 

    df = pandas_gbq.read_gbq(SQL)
    
    bars2 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['Variety']
       barmap['Value'] = row['Total'] / 50000.0
       
       bars2.append(barmap)

    data['data1'] = bars1
    data['data2'] = bars2
       
    return render(request, 'FoolingDiversity.html', data)

def class_foolability(request):    
    data = {}
    
    SQL = """
    SELECT 
      predicted_class,
      COUNT(CASE WHEN predicted_class != predicted_v_class THEN 1 END) as Fooled,
      COUNT(CASE WHEN predicted_class = predicted_v_class Then 1 END) as NotFooled
    From Project.UapEval 
    WHERE 
      type = "GEN"
    GROUP BY 
      predicted_class
    """
    df = pandas_gbq.read_gbq(SQL)
    
    bars1 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['predicted_class']
       
       fooled = row['Fooled']
       not_fooled = row['NotFooled']
        
       fooling_rate = fooled / (fooled + not_fooled)
       
       barmap['Value'] = fooling_rate
       
       bars1.append(barmap)
    
    bars1.sort(key = lambda x: x['Value'])
    
    SQL = """
    SELECT 
      predicted_class,
      COUNT(CASE WHEN predicted_class != predicted_v_class THEN 1 END) as Fooled,
      COUNT(CASE WHEN predicted_class = predicted_v_class Then 1 END) as NotFooled
    From Project.UapEval 
    WHERE 
      type = "DF"
    GROUP BY 
      predicted_class
    """
    df = pandas_gbq.read_gbq(SQL)
    
    bars2 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['predicted_class']
       
       fooled = row['Fooled']
       not_fooled = row['NotFooled']
        
       fooling_rate = fooled / (fooled + not_fooled)
       
       barmap['Value'] = fooling_rate
       
       bars2.append(barmap)
    
    bars2.sort(key = lambda x: x['Value']) 
               
    data['data1'] = bars1
    data['data2'] = bars2
    
    for i in range(len(bars1)):
        bars1[i]['Index'] = i
        bars2[i]['Index'] = i
        
    return render(request, 'ClassFoolability.html', data)


       
def image_foolability(request):
    
    data = {}
    
    SQL = """
    SELECT 
      Count,
      COUNT(*) as Total
    FROM 
    (
      SELECT 
        image_id,
        COUNT(CASE WHEN  predicted_class  != predicted_v_class THEN 1 END) as Count
      From Project.UapEval
      WHERE 
        type = "GEN"
      GROUP BY 
        image_id
    )
    Group BY
      Count
    ORDER BY
      Count 
    """
    df = pandas_gbq.read_gbq(SQL)
    
    bars1 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['Count']
       barmap['Value'] = row['Total'] / 50000.0
       
       bars1.append(barmap)
       
    SQL = """
    SELECT 
      Count,
      COUNT(*) as Total
    FROM 
    (
      SELECT 
        image_id,
        COUNT(CASE WHEN  predicted_class  != predicted_v_class THEN 1 END) as Count
      From Project.UapEval
      WHERE 
        type = "DF"
      GROUP BY 
        image_id
    )
    Group BY
      Count
    ORDER BY
      Count 
    """
    df = pandas_gbq.read_gbq(SQL)
    
    bars2 = []
    
    for index, row in df.iterrows():
       barmap = {}
        
       barmap['Label'] = row['Count']
       barmap['Value'] = row['Total'] / 50000.0
       
       bars2.append(barmap)   
    
    data['data1'] = bars1
    data['data2'] = bars2
       
    return render(request, 'ImageFoolability.html', data)
       
def uap_demo(request):    
    
    data = {}
    
    SQL = """
    SELECT * FROM Project.UapEval
    WHERE type = "GEN"
      AND predicted_class != predicted_v_class
    ORDER BY RAND()
    LIMIT 1
    """
    
    df = pandas_gbq.read_gbq(SQL)
    
    image_id = df['image_id'].iloc[0]
    perturbation_id = df['perturbation_id'].iloc[0]
    
    labels = get_labels()
    
    predicted_class = labels[df['predicted_class'].iloc[0]]
    predicted_v_class = labels[df['predicted_v_class'].iloc[0]]
    
    data['image_label'] = 'Image: ' + predicted_class
    data['purturbation_label'] = perturbation_id
    data['image_with_perturbation'] = 'Image + Perturbation: ' + predicted_v_class
    
    image = get_image('data/Images/' + img_id_to_image_name(image_id))
    v = get_v('data/GenAlgUap/' + perturbation_id)
    v_image = apply_v(image, v)
    
    imwrite('static/media/Image.JPEG', image)
    imwrite('static/media/Perturbation.JPEG', v)
    imwrite('static/media/ImageWithPerturbation.JPEG', v_image)
    
    data['selected_id'] = img_id_to_image_name(image_id)
    
    SQL = """
    SELECT
      perturbation_id,
      COUNT(CASE WHEN predicted_class != predicted_v_class THEN 1 END) as Fooled,
      COUNT(CASE WHEN predicted_class = predicted_v_class THEN 1 END) as NotFooled
    From Project.UapEval
    WHERE
      type = "GEN"
    GROUP BY
      perturbation_id
    ORDER BY
      Fooled
    """
    df = pandas_gbq.read_gbq(SQL)

    bars = []
        
    for index, row in df.iterrows():
        barmap = {}
        
        fooled = row['Fooled']
        not_fooled = row['NotFooled']
        i_perturbation_id = row['perturbation_id']
        
        fooling_rate = fooled / (fooled + not_fooled)
        
        barmap['Label'] = index
        barmap['Value'] = fooling_rate
        barmap['Id'] = i_perturbation_id

        bars.append(barmap)
        
        if (i_perturbation_id == perturbation_id):
            data['fooling_rate'] = fooling_rate

    data['data'] = bars
    
    return render(request, 'GenUapDemo.html', data)