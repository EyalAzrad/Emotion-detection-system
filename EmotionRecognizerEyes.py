import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split  # Split arrays or matrices into random train and test subsets.
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization  # need_for_training
from keras.layers import Dropout
from keras.applications import DenseNet201  # The neural network
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from Utils import *
import os
import matplotlib.pyplot as plt  # draw accuracy graph
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint

SIZE = 15
NUMBER_OF_EPOCHS = 1
IMG_SIZE = 48


class EmotionRecognizerEyes:
    def __init__(self, path, test_size, seed):
        """
         Initialize an EmotionRecognizerEyes object.

         Parameters:
             path (str): The path to the directory containing the emotion images.
             test_size (float): The proportion of data to be used as test data.
             seed (int): The seed used for random number generation.
         """
        self.__test_size = test_size
        self.__seed = seed
        self.__path = path
        self.__data = []
        self.__labels = []
        random.seed(self.__seed)
        self.__lb = LabelBinarizer()
        self.__y_test = None  # test group for fit model and evaluation
        self.__x_test = None  # test group for predictions
        self.__model = None
        self.__ypred = None  # model predictor - Generate predictions (probabilities -- the output of the last layer)
        self.__history = None  # save the fit data, for example the max loss value during the fit
        self.accuracy_path = None

        create_directory("saved_models")
        n = amount_of_files("saved_models")
        new_model = os.path.join("saved_models",
                                 str(SIZE) + '_' + str(NUMBER_OF_EPOCHS) + '_' + str(IMG_SIZE) + '_' + str(n + 1))
        create_directory(new_model)
        self.accuracy_path = new_model

        self.checkpointer = ModelCheckpoint(filepath=self.accuracy_path + '/model-{epoch:02d}.h5', save_best_only=False,
                                            save_weights_only=False, monitor='val_acc')

    def extract_images(self):
        """
        Extract images from the given path resizes them to 48x48 and normalizes the data,
        and store them in the data and labels attributes.
        """
        self.__data, self.__labels = extract_images(self.__path, IMG_SIZE)

    def fit_model(self):
        """
        Fits the model to the data by training the model on the training set and evaluating it on the test set.
        """
        labels = self.__lb.fit_transform(self.__labels)  # creates binary label array
        print(labels[0])
        (x_train, x_test, y_train, y_test) = train_test_split(self.__data, labels, test_size=self.__test_size,
                                                              random_state=self.__seed)
        print(x_train.shape, x_test.shape)
        self.__x_test = x_test
        self.__y_test = y_test

        data_generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                            horizontal_flip=True, fill_mode="nearest")

        data_generator.fit(x_train)

        # comment model_d here
        model_d = DenseNet201(weights='imagenet', include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))

        output = model_d.output  # holds the input from the neural network
        output = GlobalAveragePooling2D()(output)  # Add global average pooling layer
        output = BatchNormalization()(output)  # batch normalization layer
        output = Dropout(0.5)(output)  # helps prevent overfitting
        # Fully Connected add layer
        output = Dense(1024, activation='relu')(output)
        output = Dense(512, activation='relu')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.5)(output)  # Dropout layer for overfitting reduction
        output = Dense(6, activation='softmax')(output)  # because we have to predict the AUC
        self.__model = Model(model_d.input, output)
        self.__model.compile(
            optimizer=Adam(lr=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.__model.summary()  # statistics summary
        self.__history = self.__model.fit(data_generator.flow(x_train, y_train, batch_size=32),  # takes long time..
                                          epochs=NUMBER_OF_EPOCHS, validation_data=(x_test, y_test),
                                          callbacks=[self.checkpointer])

    def model_evaluation(self):
        """
        Evaluates the model by comparing the predictions stored in the __ypred variable to the true labels stored in the __y_test variable.
        """
        print("Evaluate on test data")
        results = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=32)
        print("test loss, test acc:", results)

    def __plot_confusion_matrix(self):
        plot_confusion_matrix(self.__confusion_matrix, self.accuracy_path)

    def get_accuracy(self, to_draw):
        """
        Plots the accuracy of the model over the course of training.
        Parameters:
             to_draw (boolean): case of true: using plot to draw the graph, otherwise, just calculate the accuracy.
        """
        self.__ypred = self.__model.predict(self.__x_test)
        total = 0
        accurate = 0
        accurateindex = []
        wrongindex = []

        for i in range(len(self.__ypred)):
            if np.argmax(self.__ypred[i]) == np.argmax(self.__y_test[i]):
                accurate += 1
                accurateindex.append(i)
            else:
                wrongindex.append(i)
            total += 1

        print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ',
              total - accurate)
        print('Accuracy:', round(accurate / total * 100, 3), '%')
        if to_draw:
            plt.title("model accuracy")
            plt.plot(self.__history.history["accuracy"])
            plt.plot(self.__history.history["val_accuracy"])
            plt.axis([1, NUMBER_OF_EPOCHS, 0, 1])
            plt.ylabel("accuracy", fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.xlabel("epoch", fontsize=15)
            plt.legend(["train", "validation"], loc="lower right")
            # plt.show()
            plt.savefig(self.accuracy_path + '/' + 'Densnett201_class4.jpg')

            y_pred = self.__model.predict(self.__x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(self.__y_test, axis=1)
            self.__confusion_matrix = confusion_matrix(y_test, y_pred, labels=range(6))
            self.__plot_confusion_matrix()