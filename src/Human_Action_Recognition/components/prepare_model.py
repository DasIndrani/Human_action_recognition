import os,sys
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.logger import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16




class ModelBuild:
    def __init__(self,train_data,val_data,train_label,val_label):
        try:
            self.train_data = train_data
            self.val_data = val_data
            self.train_label = train_label
            self.val_label = val_label
        except Exception as e:
            raise RecognitionException(e,sys)
    
    def number_of_class(self,label):
        self.num_classes = len(np.unique(label))
       
    def create_model(self,dropout_rate):
         try:
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
            # Freeze the weights of the pre-trained layers
            for layer in base_model.layers:
                layer.trainable = False
            model = models.Sequential()
            model.add(base_model)
            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
               #kernel_regularizer=regularizers.l2(0.01)
    
             #Compile the model
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',metrics=['accuracy'])
         
            return model
         except Exception as e:
             raise RecognitionException(e,sys)