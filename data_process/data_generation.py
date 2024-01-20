# Importing required libraries
import numpy as np
import pandas as pd
import logging
from sklearn.datasets import load_iris
import os
import sys
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating Iris dataset
@singleton
class IrisDatasetGenerator():
    def __init__(self):
        self.df_train = None
        self.df_inference = None

    # Method to load Iris dataset
    def load_iris_data(self):
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = iris['data']
        y = iris['target']
        self.df_train = pd.DataFrame(data=np.c_[X, y], columns=iris['feature_names'] + ['target'])
        return self.df_train

    # Method to create inference dataset from a portion of the loaded Iris dataset
    def create_inference_data(self, fraction: float, save_path: str):
        logger.info(f"Creating inference dataset with {fraction * 100}% of the data...")
        num_rows = int(len(self.df_train) * fraction)
        inference_data = self.df_train.sample(n=num_rows, random_state=42)
        self.df_inference = inference_data
        self.save(inference_data, save_path)
        return self.df_inference

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: str):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    
    # Create IrisDatasetGenerator instance
    iris_gen = IrisDatasetGenerator()
    
    # Load and save training dataset
    iris_gen.load_iris_data()
    iris_gen.save(iris_gen.df_train, save_path=TRAIN_PATH)
    
    # Create and save inference dataset with 20% of the data
    iris_gen.create_inference_data(fraction=0.2, save_path=INFERENCE_PATH)
    
    logger.info("Script completed successfully.")
