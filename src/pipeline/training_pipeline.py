import os 
import sys 
from log.looger import logging
from excep.exception import customexception
import pandas as pd 
import numpy as np 

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer 
from components.model_evaluation import ModelEvaluation


obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.init_data_transformation(train_data_path,test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)

model_eval_obj=ModelEvaluation()
model_eval_obj.initiate_model_evaluation(train_arr,test_arr)


