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
 import joblib
from spamham.utils.utils import tune_and_fit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mlflow





class Modeltrainerclass:
    
    def __inti__(self,config_info=Configuration()):
        try:
            self.model_trainer_config=config_info.get_modeltrainer_config()
            self.transformation_config = config_info.get_tarnsform_config()

        except Exception as e:
            raise CustomException(e,sys) from e
    mlflow.sklearn.autolog()
    with mlflow.start_run():
            
        def inititat_model_training(self,xtrain,xtest,ytrain,ytest):
            try:


                model_obj=  {'randomforestclassifier' : RandomForestClassifier(),
                            'svm_classifier' : SVC(),
                            'logistic_regression' : LogisticRegression(),
                            'multinomial_naive_bayse' : MultinomialNB()
                            }

                param_grids = {"randomforestclassifier":{"n_estimators": [50,100,150,200,250],
                                                            "max_depth": [1,3,5,7],
                                                            "criterion": ["gini","entropy"]
                                                            },

                                        "svm_classifier": {'C': [0.1,1,10],
                                                        'kernel':['linear','rbf'],
                                                        'gamma': ['scale','auto',0.1],
                                                        'degree': [2,3,4]},


                                    "logistic_regression": {'C': [0.001, 0.01, 0.1, 1, 10],           
                                                        'penalty': ['l1', 'l2'],                  
                                                        'max_iter': [100, 200, 300],              
                                                        'solver': ['liblinear', 'saga'],          
                                                        'multi_class': ['ovr', 'multinomial'],    
                                                        'class_weight': [None, 'balanced']},

                        
                                "multinomial_naive_bayse":{'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

                        }


                result_model_r2 = tune_and_fit(model_obj=model_obj,param_grids=param_grids,xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest)
                best_model_willbe = max(sorted(result_model_r2.values()))
                best_model = list(model_obj.keys()).index(best_model_willbe)

                obj_best_model = model_obj[best_model]

                
                os.makedirs(self.model_trainer_config.trained_models,exist_ok=True)
                model_path =os.path.join(self.model_trainer_config.trained_models,'model_file_name')
                joblib.dump(obj_best_model,model_path)
                
                r2_score = obj_best_model.predict(ytest,obj_best_model.pridect(xtest))
                return r2_score
            except Exception as e:
                raise CustomException(e,sys) from e



        def initiate_model_trainer(self):
            try:
                train_set = pd.read_csv(self.model_trainer_config.train_data)
                test_set =pd.read_csv(self.model_trainer_config.test_data)

                xtrain =train_set[:,:-1]
                ytrain = train_set[:-1]
                xtest = test_set[:,:-1]
                ytest = test_set[:-1]

# should i write here mlflow code?????

                self.inititat_model_training(xtrain,xtest,ytrain,ytest)
            except Exception as e:
                raise CustomException(e,sys) from e





    


    