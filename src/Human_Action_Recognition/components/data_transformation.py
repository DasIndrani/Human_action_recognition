import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.Human_Action_Recognition.entity.config_entity import DataTransformationConfig
from src.Human_Action_Recognition.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.Human_Action_Recognition.logger import logging
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.utils import prepare_train_data, prepare_test_data,save_numpy_array_data, save_object



class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f" ******  Data Transformation  ******")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise RecognitionException(e,sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f" Reading training and testing file ")
            train_df = pd.read_csv(self.data_ingestion_artifact.Training_set)

            logging.info(f"prepare data for train and test")
            train_image_data, train_image_label = prepare_train_data(filepath= self.data_ingestion_artifact.train, df= train_df)

            
            train_image_arr = np.array(train_image_data)
            train_image_label = np.array(train_image_label)

            #print(train_image_arr.dtype)

            logging.info(f"converting  cat feature into numerical feature using encoder")
            encoder = LabelEncoder()
            encoder.fit(train_image_label)

            logging.info(f" transform the target columns ")
            train_image_label = encoder.transform(train_image_label)

            logging.info(f"normalize the data")
            train_image_arr = train_image_arr/255.0

            #print(train_image_arr)

            
            if isinstance(train_image_arr, np.ndarray):
                 # Convert to float using astype
                 train_image_arr = train_image_arr.astype(float)
            else:
                print("The result is not a NumPy array.")

           
            logging.info(f"Save the array data")
            save_numpy_array_data(filepath = self.data_transformation_config.transform_train_path, array=train_image_arr)
            save_numpy_array_data(filepath= self.data_transformation_config.transform_train_img_label_path, array=train_image_label)

            logging.info(f" Save the object ")
            save_object(filepath = self.data_transformation_config.target_encoder_path, obj=encoder)


            data_transformation_artifact = DataTransformationArtifact(transform_train_path = self.data_transformation_config.transform_train_path,
                                                                      transform_train_img_label_path=self.data_transformation_config.transform_train_img_label_path,
                                                                      target_encoder_path = self.data_transformation_config.target_encoder_path)
            logging.info(f" Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise RecognitionException(e, sys)


            