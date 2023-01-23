from EmotionRecognizerEyes import *
from Utils import *
from Manager import *
#
if __name__ == '__main__':

    import numpy as np
    import mediapipe as mp
    from matplotlib import pyplot as plt

    # ** arrange all your data in folders labled by there emotion **.
    # # temp arrange images, for first time use!
    # src = os.path.join(r"C:\Users\edena\OneDrive\Desktop", "train_set")
    # dst = os.path.join(r"C:\Users\edena\PycharmProjects\DESNET201_EDEN", "new_data_new")
    # (src, dst)

    # ** take all the images in the labled folders and make accordingly a croped images with folders **
    # path to full data, with all the images in their folders .labled.
    # src = r"C:\Users\edena\PycharmProjects\DESNET201_EDEN\new_data_new\train"
    # # # = path to empty directory (if not created it will create one)
    # dst = r"C:\Users\edena\PycharmProjects\DESNET201_EDEN\new_data_new\eyes_train" + '/'
    # cutImages(src, dst, eyesOuter)

    # ** fit the model and save **
    # MyModel = EmotionRecognizerEyes("new_data/", 0.2, 42)
    # MyModel.extract_images()
    # MyModel.fit_model()
    # MyModel.model_evaluation()
    # # MyModel.save_model()
    # MyModel.get_accuracy(True)
    # MyModel.getAccuracy(True)

    # ** run video on the model with classify to emotions **
    manager = Manager("saved_models/model-44.h5",
                      "saved_models/model-15 .h5")
    manager.prepare_frames("VE Project 2-1.mp4", 5)
    manager.make_predictions()
    manager.cross_predictions()
    print(manager.final_predictions)