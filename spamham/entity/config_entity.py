from collections import namedtuple

DataIngestionConfig = namedtuple("IngestionConfig",
                                 ["download_url",
                                  "raw_data_dir",
                                  "ingestion_modify_dir"])

DataValidationConfig = namedtuple("validationConfig",
                                  ["modified_data_folder",
                                   "object_dir",
                                   "spam_csv_file_dir"])


DataTransformationConfig = namedtuple("DataTransformationConfig",["balanced_data_folder",
                                                                  "object_dir",
                                                                  "object_file_name",
                                                                  "y_target_csv",
                                                                  "xfeatures_bow_csv",
                                                                  "xfeatures_tfidf_csv"])



ModelTrainerConfig = namedtuple("ModelTrainerConfig",["train_data",
                                                      "test_data",
                                                      "trained_models",
                                                      "model_file_name"])


  