import os, sys
import pandas as pd
import numpy as np
from src.Human_Action_Recognition.logger import logging
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.entity.config_entity import PredictionConfig
from src.Human_Action_Recognition.entity.artifact_entity import  DataIngestionArtifact,DataTransformationArtifact,ModelTrainingArtifact
from src.Human_Action_Recognition.utils import load_object, prepare_test_data, img_to_arr
from src.Human_Action_Recognition.constants.constants import Training_set,Testing_set,test,train

from sklearn.preprocessing import LabelEncoder


class ActionRecognitionPrediction:
    def __init__(self, prediction_config:PredictionConfig):
        try:
            logging.info(f" ****** prediction ********")
            self.prediction_config=prediction_config

        except Exception as e:
            raise RecognitionException(e,sys)
        
    def start_prediction(self):
        try:
            logging.info(f"load the test set")
            test_df = pd.read_csv(os.path.join("artifact","data_ingestion","Human Action Recognition","Testing_set.csv"))

            logging.info(f"transform the data for prediction")
            encoder= load_object(filepath=os.path.join("artifact","data_transformation","target_encoder","target_encoder.pkl"))
            test_image_data = prepare_test_data(filepath="artifact\\data_ingestion\\Human Action Recognition\\test\\",df=test_df)
            test_image_arr = img_to_arr(image_data=test_image_data)

            logging.info(f"load the model to make prediction")
            history=load_object(filepath=os.path.join("artifact","model_training","model","model.h5"))
            predictions = history.model.predict(test_image_arr)

            predicted_labels = np.argmax(predictions, axis=1)
            predict_label=encoder.inverse_transform(predicted_labels)
            test_df["predict_label"] = predict_label

            file_name = "Testing_set.csv"
            prediction_file_path = os.path.join(self.prediction_config.output_dir,file_name)

            logging.info(f"Saving prediction  file : {prediction_file_path}")
            test_df.to_csv(prediction_file_path,index=False,header=True)
        except Exception as e:
            raise RecognitionException(e,sys)
