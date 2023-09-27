from spamham.components.injestion import Ingestionclass


class Pipelineclass:
    def __init__(self):
        self.ingestion_class_var = Ingestionclass()


    def strart_pipeline(self):
        self.ingestion_class_var.initiate_data_ingestion()


