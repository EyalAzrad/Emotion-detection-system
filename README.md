#Recognize human emotions from video
Emotion is a mental-physical state that has subjective and objective expressions. It may be expressed in different ways. Emotion recognition is a developing field, which may be useful in many fields. We believe that our project, which aims to identify emotions in a video where humans are present, may contribute to many fields such as: media, industry, various technologies, etc. Therefore, our challenge is to build a system that will be able to correctly classify human emotion from a video that he takes part in. In this paper we will describe our project, we are proposing a deep learning-based approach to recognizing the human mental state. We are working with six universal emotions (happiness, sadness, fear, anger, neutral, and surprise) and with a data set that we will create that contains 2 features: eyes and their surrounding areas and lips. The method that we are using in our project consists of feature extraction, feature selection, and classification. To extract the relevant features, we will use Media –pipe library, and then we will use 2 pre-trained neural network models in order to classify the emotions from the eyes and the lips. For the eyes, we will use the architecture of DenseNet201 and for the lips, we will use CNN. Our algorithm consists of 2 parts that work in parallel and in the end, we will combine the results of the two models and we will detect the emotion according to the emotion that received the highest prediction score for the frame section (the was executed from the video) we are testing. We built a method for classifying the result that takes into account the most common result classified by the 2 models.
Keywords—Human emotion recognition; convolutional neural network (CNN); Media-pipe; features; freams; detection; DenseNet-201.


example of our system :

![WhatsApp Image 2023-01-23 at 23 31 38](https://user-images.githubusercontent.com/76653366/214156149-0072b8d9-5aac-45d7-82e0-9fe2f155fb49.jpeg)


The accuracy of the models :
![image](https://user-images.githubusercontent.com/76653366/214155272-7bb44c48-9341-4436-82e9-aca3389ebd1a.png)

Confusion matrix:

![WhatsApp Image 2023-01-23 at 16 32 20 (1)](https://user-images.githubusercontent.com/76653366/214155865-faec1bd3-9adf-4425-8c9e-4b33e216b14e.jpeg)

![WhatsApp Image 2023-01-23 at 16 32 19](https://user-images.githubusercontent.com/76653366/214155910-8a044d27-1496-490c-93f4-50cc8d99c0f3.jpeg)

The accuracy of the combination of two models:
![image](https://user-images.githubusercontent.com/76653366/214155490-381d21a2-cc02-4ef2-8669-d49517c51ed2.png)



