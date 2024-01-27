from src.Human_Action_Recognition.pipeline.training_pipeline import TrainingPipeline
from src.Human_Action_Recognition.entity.config_entity import Artifacts
from src.Human_Action_Recognition.logger import logging



if __name__=="__main__":
    try:
        artifact_config = Artifacts()
        training_pipeline = TrainingPipeline(artifact_config=artifact_config)
        training_pipeline.start()
    except Exception as e:
        print(e)