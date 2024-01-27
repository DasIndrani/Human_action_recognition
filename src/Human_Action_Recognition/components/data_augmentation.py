import os,sys
import numpy as np
from src.Human_Action_Recognition.entity.config_entity import DataAugmentationConfig
from src.Human_Action_Recognition.entity.artifact_entity import DataTransformationArtifact, DataAugmentationArtifact
from src.Human_Action_Recognition.logger import logging
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.utils import load_numpy_array_data,save_object,load_object,save_numpy_array_data


from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmentation:
    def __init__(self,data_augmentation_config:DataAugmentationConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f" ********** Data Augmentation ******** ")
            self.data_augmentation_config = data_augmentation_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise RecognitionException(e,sys)
        

    def initiate_data_augmentation(self)->DataAugmentationArtifact:
        try:
             logging.info(f"load data for augmentation")
             train_image_arr = load_numpy_array_data(filepath = self.data_transformation_artifact.transform_train_path)
             train_image_label = load_numpy_array_data(filepath=self.data_transformation_artifact.transform_train_img_label_path)
            

             logging.info(f" Augmented the data ")
            # Define the data augmentation parameters
             datagen = ImageDataGenerator(
                 rotation_range=30,        # Rotate images by 10 degrees
                 width_shift_range=0.2,    # Shift images horizontally by 10% of the width
                 height_shift_range=0.2,   # Shift images vertically by 10% of the height
                 zoom_range=0.3,           # Zoom in on images by up to 20%
                 horizontal_flip=True,     # Flip images horizontally
                 fill_mode='nearest'       # Fill newly created pixels using the nearest value
                 )
             # Create augmented images
             augmented_train_images = []
             augmented_train_labels = []
             for img, label in zip(train_image_arr, train_image_label):
                  img = np.reshape(img, (img.shape[0], img.shape[1], 3))
                  # Expand the dimensions to match the input shape of the generator
                  img = np.expand_dims(img, axis=0)
                  augmented = datagen.flow(img, batch_size=1)
                  # Retrieve the augmented image and label
                  augmented_img = next(augmented)[0].astype(np.float32)
                  augmented_label = label
                  # Add augmented image and label to the lists
                  augmented_train_images.append(augmented_img)
                  augmented_train_labels.append(augmented_label)
             #Concatenate the original and augmented data
             augmented_train_images_arr = np.concatenate((train_image_arr, augmented_train_images))
             augmented_train_labels= np.concatenate((train_image_label, augmented_train_labels))

             logging.info(f" save the augmentaed data ")
             save_numpy_array_data(filepath = self.data_augmentation_config.augmented_train_path, array=augmented_train_images_arr)
             save_numpy_array_data(filepath= self.data_augmentation_config.augmented_train_label_path, array=augmented_train_labels)

             data_augmentation_artifact = DataAugmentationArtifact(augmented_train_path=self.data_augmentation_config.augmented_train_path,
                                                                    augmented_train_label_path=self.data_augmentation_config.augmented_train_label_path)
             
             return data_augmentation_artifact
            
        except Exception as e:
            raise RecognitionException(e,sys)