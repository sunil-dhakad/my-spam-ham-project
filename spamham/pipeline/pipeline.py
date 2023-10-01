from spamham.components.injestion import Ingestionclass
from spamham.components.transformation import Transformationclass
from spamham.components.validation import validationclass
from spamham.components.modeltrainer import Modeltrainerclass
from spamham.exception import CustomException
from spamham.logger import logging
import os,sys



class Pipelineclass:
    def __init__(self):
        try:
            self.ingestion_class_var = Ingestionclass()
            self.transformation_class_var = Transformationclass()
            self.validation_class_var = validationclass()
            self.modeltrainer_class_var = Modeltrainerclass()

        except Exception as e:
            raise CustomException(e,sys) from e
    def strart_pipeline(self):
        try:
            self.ingestion_class_var.initiate_data_ingestion()
            self.validation_class_var.initiate_validation()
            self.transformation_class_var.initiate_transformation()
          
            self.modeltrainer_class_var.inititat_model_training()
        except Exception as e:
            raise CustomException(e,sys) from e

