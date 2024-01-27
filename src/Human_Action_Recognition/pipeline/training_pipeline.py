import sys
from src.Human_Action_Recognition.entity.config_entity import Artifacts, DataIngestionConfig, DataTransformationConfig,DataAugmentationConfig,ModelTrainingConfig
from src.Human_Action_Recognition.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataAugmentationArtifact, ModelTrainingArtifact
from src.Human_Action_Recognition.components.data_ingestion import DataIngestion
from src.Human_Action_Recognition.components.data_transformation import DataTransformation
from src.Human_Action_Recognition.components.data_augmentation import DataAugmentation
from src.Human_Action_Recognition.components.model_training import ModelTraining
from src.Human_Action_Recognition.exception import RecognitionException


class TrainingPipeline:
    def __init__(self,artifact_config: Artifacts):
        try:
            self.artifact_config = artifact_config
        except Exception as e:
            raise RecognitionException(e,sys)
        
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion_config =  DataIngestionConfig(artifact_config=self.artifact_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.start_download()
            data_ingestion_artifact = data_ingestion.unzip()
            return data_ingestion_artifact
        
        except Exception as e:
            raise RecognitionException(e,sys)
        
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact)-> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(artifact_config=self.artifact_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                     data_ingestion_artifact = data_ingestion_artifact)
            return data_transformation.initiate_data_transformation()
        
        except Exception as e:
            raise RecognitionException(e, sys)
        
    def start_data_augmentation(self,data_transformation_artifact:DataTransformationArtifact)->DataAugmentationArtifact:
        try:
            data_augmentation_config = DataAugmentationConfig(artifact_config=self.artifact_config,
                                                              data_transformation_artifact=data_transformation_artifact)
            data_augmentation = DataAugmentation(data_augmentation_config=data_augmentation_config,
                                                 data_transformation_artifact=data_transformation_artifact)
            return data_augmentation.initiate_data_augmentation()
        
        except Exception as e:
            raise RecognitionException(e,sys)
        
    def start_model_training(self,data_augmentation_artifact:DataAugmentationArtifact)->ModelTrainingArtifact:
        try:
            model_training_config = ModelTrainingConfig(artifact_config=self.artifact_config)
            model_training = ModelTraining(model_training_config=model_training_config,
                                          data_augmentation_artifact=data_augmentation_artifact)
            return model_training.initiate_model_training()
        
        except Exception as e:
            raise RecognitionException(e, sys)
        
        
    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)

            data_augmentation_artifact = self.start_data_augmentation(data_transformation_artifact=data_transformation_artifact)

            model_training_artifact = self.start_model_training(data_augmentation_artifact=data_augmentation_artifact)

        except Exception as e:
            raise RecognitionException(e,sys)