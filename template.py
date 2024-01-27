import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")


project_name = "Human_Action_Recognition"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/entity/config_entity",
    f"src/{project_name}/entity/artifact_entity",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    "Notebook/notebook.ipynb",
    "requirements.txt",
    "setup.py"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Create directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")

