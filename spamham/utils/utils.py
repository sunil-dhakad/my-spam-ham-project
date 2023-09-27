import os,sys,yaml
from spamham.exception import CustomException
from spamham.logger import logging

def read_yalm_function(file_path:str)->dict:
    try:
        with open(file_path,'rb') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)

