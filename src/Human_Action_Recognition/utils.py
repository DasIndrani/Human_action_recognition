import os , sys
import pandas as pd
import numpy as np
import yaml
import dill
from PIL import Image
from src.Human_Action_Recognition.exception import RecognitionException



def read_yaml_file(filepath):
    try:
        with open(filepath,"r") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise RecognitionException(e,sys)



train_image_data=[]
train_image_label=[]
def prepare_train_data(filepath:str, df:pd.DataFrame):
   for i in range(len(df)):
        file = filepath + df['filename'][i]
        image = Image.open(file)
        train_image_data.append(np.asarray(image.resize((100,100))))
        train_image_label.append(df['label'][i])
   return train_image_data, train_image_label


test_image_data = []
def prepare_test_data(filepath:str, df:pd.DataFrame):
    for i in range(len(df)):
        file = filepath + df['filename'][i]
        image = Image.open(file)
        test_image_data.append(np.asarray(image.resize((100,100))))
    return test_image_data

def img_to_arr(image_data:list):
    image_arr = np.array(image_data)
    image_arr = image_arr/255.0
    return image_arr
    


def save_numpy_array_data(filepath: str, array:np.array):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise RecognitionException(e, sys)
    

def save_object(filepath:str, obj:object):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise RecognitionException(e, sys)
    
def load_numpy_array_data(filepath:str)->np.array:
    try:
        with open(filepath, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise RecognitionException(e, sys)
    
def load_object(filepath:str)->object:
    try:
        with open(filepath, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise RecognitionException(e,sys)
    



    
