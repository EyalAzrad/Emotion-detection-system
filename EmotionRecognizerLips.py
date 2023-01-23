import random
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from sklearn.model_selection import train_test_split   # for splitting the data into train and test samples
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from Utils import *
SIZE = 1000
NUMBER_OF_EPOCHS = 20
IMG_SIZE = 48


class EmotionRecognizerLips:
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


    def fitModel(self):
        """
        Fits the model to the data by training the model on the training set and evaluating it on the test set.
        """
        labels = self.__lb.fit_transform(self.__labels)  # creates binary label array
        print(labels[0])
        (x_train, x_test, y_train, y_test) = train_test_split(self.__data, labels, test_size=self.__test_size,random_state=self.__seed)
        print(x_train.shape, x_test.shape)
        self.__x_test = x_test
        self.__y_test = y_test

        data_generator = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True,
                                            shear_range=0.2)  # Image processing (augmentation to improve learning rate)
        data_generator.fit(x_train)
        inputs = tf.keras.Input(shape=(48, 48, 3))

        # Convolutional layer
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), data_format='channels_last', kernel_regularizer=l2(0.01))(inputs)
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        batchnormalize1 = tf.keras.layers.BatchNormalization()(conv2)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnormalize1)
        dropout1 = tf.keras.layers.Dropout(0.5)(pool1)
        conv3 = tf.keras.layers.Conv2D(2*64, kernel_size=(3, 3), activation='relu', padding='same')(dropout1)
        batchnormalize2 = tf.keras.layers.BatchNormalization()(conv3)
        conv4 = tf.keras.layers.Conv2D(2*64, kernel_size=(3, 3), activation='relu', padding='same')(batchnormalize2)
        batchnormalize3 = tf.keras.layers.BatchNormalization()(conv4)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnormalize3)
        dropout2 = tf.keras.layers.Dropout(0.5)(pool2)
        conv5 = tf.keras.layers.Conv2D(2*2*64, kernel_size=(3, 3), activation='relu', padding='same')(dropout2)
        batchnormalize4 = tf.keras.layers.BatchNormalization()(conv5)
        conv6 = tf.keras.layers.Conv2D(2 * 2 * 64, kernel_size=(3, 3), activation='relu', padding='same')(batchnormalize4)
        batchnormalize5 = tf.keras.layers.BatchNormalization()(conv6)
        pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnormalize5)
        dropout3 = tf.keras.layers.Dropout(0.5)(pool3)
        conv7 = tf.keras.layers.Conv2D(2 * 2*2 * 64, kernel_size=(3, 3), activation='relu', padding='same')(dropout3)
        batchnormalize6 = tf.keras.layers.BatchNormalization()(conv7)
        conv8 = tf.keras.layers.Conv2D(2 * 2 * 2 * 64, kernel_size=(3, 3), activation='relu', padding='same')(batchnormalize6)
        batchnormalize7 = tf.keras.layers.BatchNormalization()(conv8)
        pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(batchnormalize7)
        dropout4 = tf.keras.layers.Dropout(0.5)(pool4)
        flatten = tf.keras.layers.Flatten()(dropout4)
        fc1 = tf.keras.layers.Dense(2*2*2*64, activation='relu')(flatten)
        dropout5 = tf.keras.layers.Dropout(0.4)(fc1)
        fc2 = tf.keras.layers.Dense(2 * 2 * 64, activation='relu')(dropout5)
        dropout6 = tf.keras.layers.Dropout(0.4)(fc2)
        fc3 = tf.keras.layers.Dense(2 * 64, activation='relu')(dropout6)
        dropout7 = tf.keras.layers.Dropout(0.5)(fc3)
        outputs = tf.keras.layers.Dense(7, activation='softmax')(dropout7)

        # Create the model
        self.__model = tf.keras.Model(inputs=inputs, outputs=outputs)
        #Compile the model
        self.__model.compile(
            optimizer=Adam(lr=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        self.__model.summary()
        self.__history = self.__model.fit(x_train, y_train, batch_size=64,epochs=NUMBER_OF_EPOCHS, validation_data=(x_test, y_test),
                                          callbacks=[self.checkpointer])


    def modelEvaluation(self):
        """
        Evaluates the model by comparing the predictions stored in the __ypred variable to the true labels stored in the __y_test variable.
        """
        print("Evaluate on test data")
        results = self.__model.evaluate(self.__x_test, self.__y_test)
        print("test loss, test acc:", results)

    def __plot_confusion_matrix(self):
        plot_confusion_matrix(self.__confusion_matrix, self.accuracy_path)


    def getAccuracy(self, to_draw):
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
            # plt.figure(10,20)
            plt.plot(self.__history.history["accuracy"])
            plt.plot(self.__history.history["val_accuracy"])
            plt.axis([1, NUMBER_OF_EPOCHS, 0, 1])
            plt.ylabel("accuracy", fontsize=15)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.xlabel("epoch", fontsize=15)
            plt.legend(["train", "validation"], loc="lower right")
            plt.savefig(self.accuracy_path + '/' + 'cnn_class4.jpg')

            y_pred = self.__model.predict(self.__x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(self.__y_test, axis=1)
            self.__confusion_matrix = confusion_matrix(y_test, y_pred, labels=range(6))
            self.__plot_confusion_matrix()