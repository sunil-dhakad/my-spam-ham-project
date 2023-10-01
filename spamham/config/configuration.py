import os,sys
from spamham.logger import logging
from spamham.exception import CustomException
from spamham.constant import *
from spamham.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from spamham.utils.utils import read_yalm_function



try:
    pass
except Exception as e:
    raise CustomException(e,sys)



class Configuration:

    def __init__(self,config_file_path:str=CONFIG_FILE_PATH):
        try:
            self.config_info = read_yalm_function(file_path=config_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        


    def get_ingestion_config(self)->DataIngestionConfig:
        try:
             data_ingestion_config_var = self.config_info['ingestion_config']
             artifact_dir_var = self.config_info['artifacts_config']['artifact_dir']
             dataset_dir_var = data_ingestion_config_var['dataset_dir']

             raw_data_dir_var = data_ingestion_config_var['raw_data_dir']
             ingest_usable_data = data_ingestion_config_var['ingestion_df_modify_dir']
 
             download_url_var = data_ingestion_config_var['download_url']

             all_ingestion_config = DataIngestionConfig(
                                 download_url=download_url_var,
                                  raw_data_dir=raw_data_dir_var,
                                  ingestion_modify_dir=ingest_usable_data)
             

             return all_ingestion_config
        

        except Exception as e:
            raise CustomException(e,sys)



    def get_validation_config(self)->DataValidationConfig:
        try:

            validation_config_var =self.config_info['validation_config']
            ingestion_config_var =self.config_info['ingestion_config']
            dataset_dir_var = ingestion_config_var['dataset_dir']
            raw_data_dir = ingestion_config_var['raw_data_dir']

            artifact_dir = self.config_info['artifacts_config']['artifact_dir']


            modified_folder_var =os.path.join(validation_config_var['modified_data_folder'])
            object_dir_var = os.path.join(artifact_dir,validation_config_var['object_dir'])
            
            
            spam_csv_file_dir = validation_config_var['spam_csv_file_dir']
            

            validation_config_var = DataValidationConfig(modified_data_folder=modified_folder_var,
                                                            object_dir=object_dir_var,
                                                            spam_csv_file_dir=spam_csv_file_dir)
            
            return validation_config_var


        except Exception as e:
            raise CustomException(e,sys)




    def get_tarnsform_config(self)->DataTransformationConfig:
        try:
            transformation_config_var = self.config_info['transform_config']
            data_folder_var = transformation_config_var['balanced_data_folder']
            object_dir_var = transformation_config_var['object_dir']
            object_file_name_var = transformation_config_var['object_file_name']

            y_target_csv_filename = transformation_config_var['y_target_csv']
            xfeatures_bow_csv_filename = transformation_config_var['xfeatures_bow_csv']
            xfeatures_tfidf_csv_filename = transformation_config_var['xfeatures_tfidf_csv']

            artifact_dir = self.config_info['artifacts_config']['artifact_dir']

            y_target_csv_path = os.path.join(y_target_csv_filename)
            xfeatures_bow_csv_path = os.path.join(xfeatures_bow_csv_filename)
            xfeatures_tfidf_csv_path = os.path.join(xfeatures_tfidf_csv_filename)


            result_config = DataTransformationConfig(balanced_data_folder=data_folder_var,
                                                     object_dir=object_dir_var,
                                                     object_file_name=object_file_name_var,
                                                     y_target_csv=y_target_csv_path,
                                                     xfeatures_bow_csv=xfeatures_bow_csv_path,
                                                     xfeatures_tfidf_csv=xfeatures_tfidf_csv_path)
            return result_config
        except Exception as e:
            raise CustomException(e,sys) from e




    def get_modeltrainer_config(self)->ModelTrainerConfig:
        try:

            artifact_dir = self.config_info['artifacts_config']['artifact_dir']

            modeltrainer_config_var =self.config_info['model_trainer_config']



            train_file_name = self.config_info['train_data']
            test_file_name = self.config_info['test_data']

            test_file_path = os.path.join(artifact_dir,self.config_info['transform_config']['balanced_data_folder'],test_file_name)

            train_file_path = os.path.join(artifact_dir,self.config_info['transform_config']['balanced_data_folder'],train_file_name)


            trained_model_var = os.path.join(artifact_dir,modeltrainer_config_var['trained_model'])
            model_file_path = os.path.join(artifact_dir,modeltrainer_config_var['model_file_name'])



            model_trainer_config = ModelTrainerConfig(train_data=train_file_path,
                                                    test_data=test_file_path,
                                                    trained_models=trained_model_var,
                                                    model_file_name=model_file_path)
            return model_trainer_config


        except Exception as e:
            raise CustomException(e,sys) from e
