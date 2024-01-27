from src.Human_Action_Recognition.entity.config_entity import PredictionConfig
from src.Human_Action_Recognition.pipeline.prediction_pipeline import ActionRecognitionPrediction

if __name__=="__main__":
    prediction_config = PredictionConfig()
    action_recognition_prediction = ActionRecognitionPrediction(prediction_config=prediction_config)
    action_recognition_prediction.start_prediction()


