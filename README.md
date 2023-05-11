##### DESCRIPTION


The focus of the thesis project was to detect lung cancer type using CT scan images, with the main objective being to determine the suitable image processing method for medical images when only a limited amount of training data is available. In the study, PET-CT DICOM images from TCIA platform is used. The project utilized the prototypical network, a well-known few-shot learning model, which creates a prototype of all feature vectors from images. To achieve a better feature space, four different feature extraction methods were explored.

- VGG16
- ResNet50
- DenseNet
- CNN

VGG16 model provided the best feature performance.To further improve the accuracy of the model, Meta Learning was applied.The all implementation with VGG16 model is in FSL_model.py file.

