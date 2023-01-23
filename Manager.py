from keras.models import load_model
from VideoManager import *
from Utils import *
from collections import Counter
from numpy import asarray

lips_weights = {1: 1.5, 5: 1.1}
eyes_weights = {0: 1.2, 4: 1.35}


class Manager:
    def __init__(self, lips_model_path, eyes_model_path):
        """
        Initialize the class
        Parameters:
        lips_model_path (str) : path to lips model
        eyes_model_path (str) : path to eyes model
        """
        self.eyes_model = load_model(eyes_model_path)
        self.lips_model = load_model(lips_model_path)
        self.video_manager = VideoManager()
        self.eyes_predictions = []
        self.lips_predictions = []
        self.final_predictions = []

    def prepare_frames(self, video_path, seconds_per_clip):
        """
        Prepare the video frames to use them in predictions.
        Parameters:
        video_path (str) : path to the video
        seconds_per_clip (float): length of each clip
        """
        self.video_manager.process_video(video_path, seconds_per_clip)

    def prepare_image(self, frame, organ):
        """
        Prepare the image and return it in a format that can be used by the model
        Parameters:
        frame (numpy array) : image frame
        organ (str): organ to predict, lips or eyes
        """
        import numpy as np
        import cv2

        outer = lipsOuter if organ == 'lips' else eyesOuter
        frame = cut_image(outer, frame)
        if frame is None:
            return None
        img = cv2.resize(frame, (48, 48))
        img = asarray(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def eyes_predict(self, frame):
        """
        Given an image of eyes, this function will predict the emotion of the eyes using a pre-trained model.

        Parameters:
            frame (numpy array): a numpy array representing an image of eyes
        Returns:
            int : the predicted class of the image, which is an index of the emotions_arr list
        """
        image = self.prepare_image(frame, 'eyes')
        if image is None:
            return None
        # Classify the image using the model's predict method
        predictions = self.eyes_model.predict(image)

        # Extract the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = predicted_class.item()

        return predicted_class

    def lips_predict(self, frame):
        """
        Given an image of lips, this function will predict the emotion of the lips using a pre-trained model.

        Parameters:
            frame (numpy array): a numpy array representing an image of lips
        Returns:
            int : the predicted class of the image, which is an index of the emotions_arr list
        """
        image = self.prepare_image(frame, 'lips')

        if image is None:
            return None
        # Classify the image using the model's predict method
        predictions = self.lips_model.predict(image)

        # Extract the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = predicted_class.item()

        return predicted_class

    def make_predictions(self):
        """
        This function makes predictions on the frames of the video that was processed by the process_video() function.
        It first checks if the video was processed, if not it returns an error message.
        Then, for each clip of frames, it calls the eyes_predict() and lips_predict() functions to get predictions for each frame.
        The predictions are saved in the eyes_predictions and lips_predictions lists.
        """
        if self.video_manager.frames is None:
            print("upload video first")
            return
        for clip in self.video_manager.frames:
            clip_lips_predictions = []
            clip_eyes_predictions = []
            for frame in clip:
                prediction_eyes = self.eyes_predict(frame)
                prediction_lips = self.lips_predict(frame)
                if prediction_eyes is None or prediction_lips is None:
                    continue
                clip_eyes_predictions.append(prediction_eyes)
                clip_lips_predictions.append(prediction_lips)
            self.lips_predictions.append(clip_lips_predictions)
            self.eyes_predictions.append(clip_eyes_predictions)

    def merge_maps(self, arr1, arr2):
        # Create an empty list to store the merged maps
        merged_maps = []
        # Iterate through both arrays
        for i in range(len(arr1)):
            # Create a new map to store the merged values
            merged_map = {}
            # Iterate through the keys and values of the first map
            for key, value in arr1[i].items():
                # If the key is not already in the merged map, add it
                if key not in merged_map:
                    merged_map[key] = value
                # If the key is already in the merged map, add the values together
                else:
                    merged_map[key] += value
            # Iterate through the keys and values of the second map
            for key, value in arr2[i].items():
                # If the key is not already in the merged map, add it
                if key not in merged_map:
                    merged_map[key] = value
                # If the key is already in the merged map, add the values together
                else:
                    merged_map[key] += value
            # Add the merged map to the list of merged maps
            merged_maps.append(merged_map)
        # Return the list of merged maps
        return merged_maps

    def make_histogram(self, clip, organ):
        counts = Counter(clip)
        counts = [(key, value) for key, value in counts.items()]
        weights = lips_weights if organ == 'lips' else eyes_weights
        histogram = {key: value * weights.get(key, 1) for key, value in counts}
        return histogram

    def cross_predictions(self):
        emotion_per_clip_lips = []
        emotion_per_clip_eyes = []
        for clip in self.lips_predictions:
            emotion_per_clip_lips.append(self.make_histogram(clip, 'lips'))
        for clip in self.eyes_predictions:
            emotion_per_clip_eyes.append(self.make_histogram(clip, 'eyes'))
        union_histogram = self.merge_maps(emotion_per_clip_eyes, emotion_per_clip_lips)
        for i in range(len(union_histogram)):
            highest_value_key = max(union_histogram[i], key=union_histogram[i].get)
            self.final_predictions.append(emotions_arr[highest_value_key])


def run(path_to_video, seconds):
    manager = Manager("saved_models/model-44.h5",
                      "saved_models/model-15 .h5")
    manager.prepare_frames(path_to_video, seconds)
    manager.make_predictions()
    manager.cross_predictions()
    return manager.final_predictions
