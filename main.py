import os,sys
import pandas as pd 
import numpy as np
from spamham.exception import CustomException
from spamham.logger import logging
from spamham.config.configuration import Configuration
from spamham.entity.config_entity import DataIngestionConfig
from spamham.constant import *

from spamham.pipeline.pipeline import Pipelineclass

def main():
    try:
        pipeline_class_var = Pipelineclass()
        pipeline_class_var.strart_pipeline()
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__=="__main__":
    main()