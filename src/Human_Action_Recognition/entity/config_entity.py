import os, sys
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.Human_Action_Recognition.utils import read_yaml_file
from src.Human_Action_Recognition.constants.constants import Training_set,train

#data_ingestion_artifact = DataIngestionArtifact(Training_set=)
data = read_yaml_file(filepath="dataset/config.yaml")
source_path = data['path']['source_path']
Dataset = "data.zip"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_PATH = "model.h5"

class Artifacts:
    def __init__(self):
        os.makedirs("artifact", exist_ok=True)
        self.artifact_dir = os.path.join("artifact")

class DataIngestionConfig:
    def __init__(self,artifact_config:Artifacts):
        try:
            data_ingestion_dir = os.path.join(artifact_config.artifact_dir,"data_ingestion")
            self.source_path = source_path
            self.local_path = os.path.join(data_ingestion_dir, Dataset)
            self.unzip_file = os.path.join(data_ingestion_dir)
        except Exception as e:
            raise RecognitionException(e,sys)
        
class DataTransformationConfig:
    def __init__(self,artifact_config:Artifacts):
        try:
            data_transformation_dir = os.path.join(artifact_config.artifact_dir,"data_transformation")
            self.transform_data = os.path.join(data_transformation_dir,"transform_data")
            self.transform_train_path = os.path.join(self.transform_data, os.path.basename(Training_set).replace("csv","npz"))
            self.transform_train_img_label_path = os.path.join(self.transform_data,"image_label.pkl")
            self.target_encoder_path = os.path.join(data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise RecognitionException(e,sys)
        
class DataAugmentationConfig:
    def __init__(self,artifact_config:Artifacts,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            data_augmentation_dir = os.path.join(artifact_config.artifact_dir, "data_augmentation")
            self.augmented_data = os.path.join(data_augmentation_dir, "augmented_data")
            self.augmented_train_path = os.path.join(self.augmented_data, os.path.basename(data_transformation_artifact.transform_train_path))
            self.augmented_train_label_path = os.path.join(self.augmented_data,os.path.basename(data_transformation_artifact.transform_train_img_label_path))
        
        except Exception as e:
            raise RecognitionException(e,sys)
        
class ModelTrainingConfig:
    def __init__(self,artifact_config:Artifacts):
        try:
            model_training_dir = os.path.join(artifact_config.artifact_dir,"model_training")
            self.model_path = os.path.join(model_training_dir,"model", MODEL_FILE_PATH)
            self.expected_score = 0.5
            self.overfitting_threshold = 0.1
            self.test_size = 0.2
        except Exception as e:
            raise RecognitionException(e, sys)
        
class PredictionConfig:
    def __init__(self):
        try:
            os.makedirs("prediction", exist_ok=True)
            prediction_dir=os.path.join("prediction")
            self.output_dir = os.path.join(prediction_dir,"output")
            os.makedirs(self.output_dir, exist_ok=True)
            
        except Exception as e:
            raise RecognitionException(e,sys)
        
        