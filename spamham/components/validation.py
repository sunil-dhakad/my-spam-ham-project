import pandas as pd 
import numpy as np 
import pickle
from spamham.exception import CustomException
from spamham.logger import logging
from spamham.config.configuration import Configuration
import re
from spamham.utils.utils import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
nltk.download ('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import joblib
import os,sys


class validationclass:
    def __init__(self,config_info=Configuration()):
        try:

            self.validn_config_fn_var = config_info.get_validation_config()
            self.ingestion_config = config_info.get_ingestion_config()
        except Exception as e:
            raise CustomException(e,sys) from e
        


    def cleaning_data(self):
        try:

 #           raw_data_dir = self.ingestion_config.raw_data_dir
 #           file_name = os.listdir(raw_data_dir)[0]
 #           file_path = os.path.join(raw_data_dir,file_name)

            dff = pd.read_csv(self.validn_config_fn_var.spam_csv_file_dir, sep="," encoding = "ISO-8859-1", usecols=[0,1], skiprows=1,names=["target", "msg"])

            dff.drop_duplicates(inplace=True)

            dff['msg'] = dff['msg'].apply(lambda x:x.lower())

            dff['msg']  = dff['msg'].apply(lambda x: re.sub('[\d+]',' ',x))
        
            dff['msg']  = dff['msg'].apply(lambda x: re.sub('[\)&.(]',' ',x))

            dff['msg'] = dff['msg'].apply(lambda x: re.sub(re.compile('<.*?>'), '', x))


            dff['msg']= dff['msg'].apply(remove_stopwords)

            dff['target'] = dff['target'].map({'ham':1,'spam':0})


            toknzr = TweetTokenizer()
            dff['msg'] = dff['msg'].apply(lambda x: toknzr.tokenize(x))
            
            
            dff['msg']=dff['msg'].astype(str)

            bow = CountVectorizer(max_features=3000)
            tfidf = TfidfVectorizer()

            xfeatures_bow = bow.fit_transform(dff['msg'])
            xfeatures_tfidf = tfidf.fit_transform(dff['msg'])

            y_target =dff['target']


            os.makedirs(self.validn_config_fn_var.modified_data_folder,exist_ok=True)
            os.makedire(self.validn_config_fn_var.object_dir,exist_ok=True)
            xfeatures_bow.to_csv(self.validn_config_fn_var.modified_data_folder,xfeatures_bow.csv,index=False)
            xfeatures_tfidf.to_csv(self.validn_config_fn_var.modified_data_folder,xfeatures_tfidf.csv,index=False)

            y_target.to_csv(self.validn_config_fn_var.modified_data_folder,y_target.csv,index=False)


            obj_file_name = os.path.join(self.validn_config_fn_var.object_dir)
 #           joblib.dump(bow,self.validn_config_fn_var.object_dir,bow.pkl)

 # do i need to save above objects like ->>> bow,tfidf,toknzr 

            return 
        except Exception as e:
            raise CustomException(e,sys) from e

    def initiate_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*15} ")
            self.cleaning_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")


        except Exception as e:
            raise CustomException(e, sys) from e

