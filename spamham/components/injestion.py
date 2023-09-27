import os,sys
import pandas as pd 
import numpy as np
from spamham.exception import CustomException
from spamham.logger import logging
from spamham.config.configuration import Configuration
from spamham.entity.config_entity import DataIngestionConfig
from spamham.constant import *
import urllib
import requests


class Ingestionclass:
    def __init__(self,ingestion_configuration_class_var =Configuration()):
        try:
            self.ingestion_configuration_fn_var = ingestion_configuration_class_var.get_ingestion_config()
        except Exception as e:
            raise CustomException(e,sys)


    def download_data(self):
        try:
            download_url_var = self.ingestion_configuration_fn_var.download_url
            download_in_rawdir_var = self.ingestion_configuration_fn_var.raw_data_dir
            os.makedirs(download_in_rawdir_var,exist_ok=True)
            
            download_file_name_var = os.path.basename(download_url_var)
            file_path_var =os.path.join(download_in_rawdir_var,download_file_name_var)

            urllib.request.urlretrieve(download_url_var,file_path_var)
            logging.info("downloading data")
            
            return file_path_var
        except Exception as e:
            raise CustomException(e,sys)
        
 #   def modifying_data(self):

#        raw_data_dir = self.ingestion_configuration_fn_var.raw_data_dir

#        file_name = os.listdir(raw_data_dir)[0]
#        file_path = os.path.join(raw_data_dir,file_name)

#        data = pd.read_csv()




    
    def initiate_data_ingestion(self):
        try:

            download_data = self.download_data()
            logging.info("downloa complete")

        except Exception as e:
            raise CustomException(e,sys)