
#constants for the names of the diseases and the models path as well.

class DiseaseIdentifiers:
    ALZHEIMER = "alzheimer"
    BRAIN_TUMOR = "brain_tumour"
    BREAST_CANCER = "breast_cancer"
    COVID = "covid"
    DIABETES = "diabetes"
    HEART_DISEASE = "heart_disease"
    PNEUMONIA = 'pneumonia'


class ModelPaths:
    ALZHEIMER_MODEL = "static/models/AlzheimerRfc.pkl"
    BRAIN_TUMOR_MODEL = "static/models/BrainTumorDetection.h5"
    BREAST_CANCER_MODEL = "static/models/breastCancerRfc.pkl"
    COVID_MODEL = "static/models/CovidDetection.h5"
    DIABETES_MODEL = "static/models/DiabetesClf.pkl"
    HEART_DISEASES = "static/models/HeartDiseaseSVM.pkl"
    PNEUMONIA_MODEL = "static/models/PneumoniaDetection.h5"

class ScalerPaths:
    ALZHEIMER_SCALER = "static/scalers/alzheimerScaler.sav"
    BREAST_CANCER_SCALER = "static/scalers/breastCancerScaler.sav"
    DIABETES_SCALER = "static/scalers/diabetesScaler.sav"
    HEART_DISEASE_SCALER = "static/scalers/heartDiseaseScaler.sav"