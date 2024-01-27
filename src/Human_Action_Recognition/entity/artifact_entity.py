import os
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    Training_set: str
    train: str

@dataclass
class DataTransformationArtifact:
    transform_train_path:str
    transform_train_img_label_path:str
    target_encoder_path: str

@dataclass
class DataAugmentationArtifact:
    augmented_train_path:str
    augmented_train_label_path:str

@dataclass
class ModelTrainingArtifact:
    model_path:str
    accuracy_train_score:float
    accuracy_val_score:float

    