import os,sys,yaml
from spamham.exception import CustomException
from spamham.logger import logging
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def read_yalm_function(file_path:str)->dict:
    try:
        with open(file_path,'rb') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)



def remove_stopwords(text):
    stop_words_list = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words_list]
    return ' '.join(filtered_words)