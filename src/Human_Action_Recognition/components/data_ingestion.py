import os, sys
import shutil
import zipfile
from src.Human_Action_Recognition.entity.config_entity import DataIngestionConfig
from src.Human_Action_Recognition.entity.artifact_entity import DataIngestionArtifact
from src.Human_Action_Recognition.logger import logging
from src.Human_Action_Recognition.exception import RecognitionException
from src.Human_Action_Recognition.constants.constants import Training_set,train

from src.Human_Action_Recognition.entity.config_entity import Artifacts


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise RecognitionException(e,sys)
        

    def start_download(self):
        try:
            os.makedirs(self.data_ingestion_config.local_path, exist_ok= True)
            shutil.copy2(self.data_ingestion_config.source_path, self.data_ingestion_config.local_path)
        except Exception as e:
            raise RecognitionException(e,sys)

    def unzip(self)->DataIngestionArtifact:
        try:
            os.makedirs(self.data_ingestion_config.unzip_file,exist_ok=True)
            if os.path.exists(self.data_ingestion_config.local_path):
                #print(self.data_ingestion_config.local_path)
                zip_file_name = os.path.basename(self.data_ingestion_config.source_path)
                copied_zip_path = os.path.join(self.data_ingestion_config.local_path, zip_file_name)
                with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_ingestion_config.unzip_file)
                    print(self.data_ingestion_config.unzip_file)
            logging.info(f"Unzip all file and folder successfully")


            logging.info("Preparing data ingestion artifact")
           
            data_ingestion_artifact = DataIngestionArtifact(Training_set= Training_set,
                                                            train = train)
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        

        except Exception as e:
            raise RecognitionException(e,sys)

        
if __name__=="__main__":
    try:
        artifact_config = Artifacts()
        data_ingestion_config = DataIngestionConfig(artifact_config = artifact_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.start_download()
        data_ingestion.unzip()
    except Exception as e:
        #logging.INFO(e)
        print(e)    