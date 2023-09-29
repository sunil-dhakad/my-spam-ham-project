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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


class Transformationclass:

        def __init__(self,config_info=Configuration()):       
            try:
                self.transform_config = config_info.get_tarnsform_config()
                self.validation_config =config_info.get_validation_config()

            except Exception as e:
                raise CustomException(e,sys)
            
        def smoting_split(self,yfile_path,xfile_bow_path,xfile_tfidf_path):
            try:


                target =pd.read_csv(yfile_path,index=False)
                xfeature_bow = pd.read_csv(xfile_bow_path,index=False)
                xfeature_tfidf = pd.read_csv(xfile_tfidf_path,index=False)




                sampling_strategy = 0.8  
                random_state = 42  
                k_neighbors = 4  

                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)

                xbow, ybow = smote.fit_resample(xfeature_bow,target)

                xtfidf, ybow = smote.fit_resample(xfeature_tfidf,target)

                
                
                 
                os.makedirs(self.transform_config.object_dir)
                object_path = os.path.join(self.transform_config.object_dir,'smote_obj.pkl')
                joblib.dump(smote,object_path)

                train_set,test_set = train_test_split(xbow,ybow,test_size = 0.2,random_state = 42)
                os.makdirs(os.path.dirname(self.transform_config.balanced_data_folder,'train_set_bow'),exist_ok=True)
                train_set.to_csv(self.transform_config.balanced_data_folder,'train_set_bow',index=False,header=True)

                os.makdirs(os.path.dirname(self.transform_config.balanced_data_folder,'test_set_bow'),exist_ok=True)
                test_set.to_csv(self.transform_config.balanced_data_folder,'test_set_bow',index=False,header=True)


            except Exception as e:
                 raise CustomException(e,sys) from e



        def initiate_transformation(self):
            try:
             
                yfile_path = self.transform_config.y_target_csv
                xfile_bow_path = self.transform_config.xfeatures_bow_csv
                xfile_tfidf_path = self.transform_config.xfeatures_tfidf_csv
                self.smoting_split(yfile_path,xfile_bow_path,xfile_tfidf_path)

            except Exception as e:
                 raise CustomException(e,sys) from e



