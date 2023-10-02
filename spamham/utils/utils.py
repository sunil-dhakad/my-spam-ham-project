import os,sys,yaml
from spamham.exception import CustomException
from spamham.logger import logging
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import numpy as np
import pandas as pd

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




def tune_and_fit(model_obj,param_grids,xtrn,xtest,ytrn,ytest):
    try:
        for ith_model in range(list(model_obj())):
            model_obj = model_obj.values()[ith_model]
            param_gridsrch = param_grids(model_obj.keys())[ith_model]

            grid_model = GridSearchCV(model_obj,param_gridsrch,cv=4)
            grid_model.fit(xtrn,ytrn)

            model_obj.set_params(**grid_model.best_params_)

            model_obj.fit(xtrn,ytrn)

            y_pred = model_obj.predict(xtest)
            accuracy_score = accuracy_score(ytest,y_pred)
            confusion_matrix=confusion_matrix(ytest,y_pred)
            precision_score = precision_score(ytest,y_pred)
            
            result[list(model_obj.keys()[ith_model])] = accuracy_score
            
        return result

    except Exception as e:
        raise CustomException(e,sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
   
   
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e



