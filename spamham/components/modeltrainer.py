import pandas as pd 
import numpy as np 
import pickle
from spamham.exception import CustomException
from spamham.logger import logging
from spamham.config.configuration import Configuration
import re
from spamham.utils.utils import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os,sys
 

class Modeltrainerclass:
    
    def __inti__(self,config_info=Configuration()):
        try:
            self.model_trainer_config=config_info.get_modeltrainer_config()
            self.transformation_config = config_info.get_tarnsform_config()

        except Exception as e:
            raise CustomException(e,sys) from e
        

    def inititat_model_traininf(self):

        




    


    