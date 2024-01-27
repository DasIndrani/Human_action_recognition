import os,sys
from src.Human_Action_Recognition.logger import logging
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.components.data_ingestion import DataIngestion
from src.Human_Action_Recognition.components.data_transformation import DataTransformation
from src.Human_Action_Recognition.entity.config_entity import ModelTrainingConfig
from src.Human_Action_Recognition.entity.artifact_entity import ModelTrainingArtifact, DataTransformationArtifact, DataAugmentationArtifact
from src.Human_Action_Recognition.utils import load_numpy_array_data, load_object, save_object
from src.Human_Action_Recognition.components.prepare_model import ModelBuild

from src.Human_Action_Recognition.entity.config_entity import Artifacts

from sklearn.model_selection import train_test_split 
import tensorflow as tf


class ModelTraining:
    def __init__(self,model_training_config:ModelTrainingConfig,
                 data_augmentation_artifact:DataAugmentationArtifact):
        
        try:
            logging.info(f" ******* Model Training *******")
            self.model_training_config = model_training_config
            self.data_augmentation_artifact = data_augmentation_artifact

        except Exception as e:
            raise RecognitionException(e, sys)
        
    def initiate_model_training(self)->ModelTrainingArtifact:
        try:
            logging.info(f" Load the numpy array data ")
            train_image_arr = load_numpy_array_data(filepath = self.data_augmentation_artifact.augmented_train_path)

            logging.info(f" load the save object ")
            train_image_label = load_numpy_array_data(filepath = self.data_augmentation_artifact.augmented_train_label_path)

            logging.info(f" split the data into train and val dataset")
            train_data, val_data, train_label, val_label = train_test_split(train_image_arr,train_image_label, 
                                                                            test_size=self.model_training_config.test_size,
                                                                            random_state= 33)
            
            early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001,
                                                           patience=8,verbose=1,mode="auto",baseline=None,
                                                             restore_best_weights=False,start_from_epoch=0)

            
            logging.info(f"build the model")
            model_build = ModelBuild(train_data=train_data, train_label=train_label, val_data= val_data, val_label= val_label)
            model_build.number_of_class(label=train_image_label)
            model = model_build.create_model(dropout_rate= 0.5)

            logging.info(f"Train the model and also store the progression of the model")
            history=model.fit(train_data, train_label, epochs=100, validation_data=(val_data, val_label), batch_size=20, callbacks=[early_stop])

            logging.info(f" prediction on train and val dataset ")
            train_pred = history.model.predict(train_data)
            val_pred = history.model.predict(val_data)

            logging.info(f"model evalution")
            train_loss,train_accuracy = model.evaluate(train_data,train_label)
            val_loss, val_accuracy = model.evaluate(val_data, val_label)

            logging.info(f"model_train_accuracy: {train_accuracy},  model_val_accuracy: {val_accuracy}")

            logging.info(f"checking if our model is underfitting or optimal")
            if(train_accuracy<self.model_training_config.expected_score):
                print(f"Model is not good as it's not give good accuracy:{train_accuracy}")
            
            logging.info(f" checking if our model is overfitting or optimal")
            if(abs(train_accuracy - val_accuracy)> self.model_training_config.overfitting_threshold):
                print(f"Difference between train and val accuracy :{abs(train_accuracy - val_accuracy)} is more than overfitting threshold")
            
            logging.info(f" save the model ")
            save_object(filepath = self.model_training_config.model_path, obj= history)


            logging.info("prepare artifact")
            model_training_artifact = ModelTrainingArtifact(model_path=self.model_training_config.model_path,
                                                          accuracy_train_score=train_accuracy,
                                                          accuracy_val_score=val_accuracy)
            logging .info(f"model trainer artifact:{model_training_artifact}")
            return model_training_artifact



        except Exception as e:
            raise RecognitionException(e, sys)
        


if __name__=="__main__":
    try:
        artifact_config = Artifacts()
        model_training_config = ModelTrainingConfig(artifact_config = artifact_config)
        data_augmentation_artifact = DataAugmentationArtifact(augmented_train_path="artifact/data_augmentation/augmented_data/Training_set.npz",
                                                              augmented_train_label_path= "artifact/data_augmentation/augmented_data/image_label.pkl")
        model_trainer = ModelTraining(model_training_config=model_training_config, data_augmentation_artifact=data_augmentation_artifact)
        model_trainer.initiate_model_training()
    except Exception as e:
        #logging.INFO(e)
        print(e)

        

