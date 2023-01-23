import cv2
import mediapipe as mp
import os
import warnings
import shutil
import numpy as np
import matplotlib.pyplot as plt  # draw accuracy graph
import seaborn as sns

warnings.filterwarnings("ignore")
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
# constant
RGB = 255.0
lipsOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
eyesOuter = [156, 70, 63, 105, 66, 107, 55, 193, 143, 111, 117, 118, 119, 120, 121, 128, 245, 383, 300, 293, 334,
             296, 336, 285, 417, 372, 340, 346, 347, 348, 349, 350, 357, 465]
emotions_arr = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def cut_images(input_dir, output_dir, outer):
    """
    Crop the images in the input directory based on the outer points specified in the outer parameter.
    Args:
    input_dir (str) : The directory where the images are located
    output_dir (str) : The directory where the cropped images will be saved
    outer (list) : A list of outer points that specifies which part of the image to crop

    Returns:
    None
    """
    create_directory(output_dir)
    for emotion in emotions_arr:
        out_path = os.path.join(output_dir, emotion)
        input_path = os.path.join(input_dir, emotion)
        create_directory(out_path)
        for file in os.listdir(input_path):
            sourceimage = input_path + '\\' + file
            destinationimage = out_path + '\\' + file
            crop_img = cut_image(outer, sourceimage)
            if crop_img is None or crop_img.size == 0:
                continue
            cv2.imwrite(destinationimage, crop_img)


def cut_image(outer, cv_image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True)
    if cv_image is None:
        return None
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_image)
    height, width, _ = rgb_image.shape
    print(height, width)
    organs = [0, 0, 0, 0]
    if not results.multi_face_landmarks:
        return None
    for facial_landmarks in results.multi_face_landmarks:
        pt1 = facial_landmarks.landmark[outer[0]]
        organs[0] = organs[1] = int(pt1.y * height)
        organs[2] = organs[3] = int(pt1.x * width)
        for i in outer:
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            if y < organs[0]:
                organs[0] = y
            if y > organs[1]:
                organs[1] = y
            if x < organs[2]:
                organs[2] = x
            if x > organs[3]:
                organs[3] = x
    return cv_image[organs[0]:organs[1], organs[2]:organs[3]]


def create_directory(path):
    """
    Create a directory if it does not already exist.
    Args:
    path (str) : The path of the directory to be created

    Returns:
    None
    """
    if not os.path.exists(path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(path)


def amount_of_files(path):
    """
    Get the number of files in a given directory
    Args:
    path (str) : The path of the directory to count the files in

    Returns:
    int : The number of files in the directory
    """
    _, dirnames, _ = next(os.walk(path))
    return len(dirnames)


def image_arrangement(src, dst, max_size=None):
    """
    Extracts and arranges images from a given directory for use in training or testing.
    Args:
    path (str) : The path of the directory containing the images
    SIZE (int) : The number of images to extract for each classification
    """
    emotion_size = [0, 0, 0, 0, 0, 0, 0]
    path = os.path.join(dst, "train")
    create_directory(path)
    for emotion in emotions_arr:
        emotion_path = os.path.join(path, emotion)
        create_directory(emotion_path)

    images_path = os.path.join(src, "images")
    labels_path = os.path.join(src, "annotations")
    directory = list(os.listdir(images_path))
    for img in directory:
        i = int(os.path.splitext(img)[0])
        numpy_arr_label = os.path.join(labels_path, str(i) + "_exp.npy")
        data = int(np.array(np.load(numpy_arr_label)).tolist())
        if data > 6:
            continue
        dst_path = os.path.join(path, emotions_arr[data])
        if max_size is not None and emotion_size[data] > max_size:
            continue
        cur_img = os.path.join(images_path, img)
        emotion_size[data] = emotion_size[data] + 1
        shutil.copy(cur_img, dst_path)


def extract_images(path_, size):
    """
    Extract images from the given path resizes them to 48x48 and normalizes the data,
    and store them in the data and labels attributes.
    """
    data = []
    labels = []
    if len(emotions_arr) < 6:
        exit(0)
    for emotion in emotions_arr:
        number = 0
        path = sorted(list(os.listdir(path_ + emotion)))
        for i in path:
            if number == size:
                break
            image = cv2.imread(path_ + emotion + '/' + i)
            if image is None or image.size == 0:
                continue
            image = cv2.resize(image, (size, size))
            image = np.asarray(image)
            data.append(image)
            label = emotion
            labels.append(label)
            number = number + 1
    data = np.array(data, dtype="float32") / RGB  # Normalize the data (scale 0-1)
    labels = np.array(labels)
    return data, labels


def plot_confusion_matrix(confusion_matrix, path):
    ax = sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(emotions_arr)
    ax.yaxis.set_ticklabels(emotions_arr)
    for label in ax.xaxis.get_ticklabels():
        label.set_size(9)
    for label in ax.yaxis.get_ticklabels():
        label.set_size(9)
    plt.savefig(path + '/' + 'Densnett201_confusion_matrix.jpg')