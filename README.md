# Customer Feedback System Based On Facial Expressions
## Abstract
Customer feedback is the information from the customer about the experience of services offered by the retailer. It helps the store owner to improve the customer service. The changes in the customer feedback collection systems aim to reduce the time and complexity of submission. One such straightforward approach to a customer feedback system is the facial expression based emotion feedback collection system. The facial expression based emotion feedback system is the magnified thought of Emotional Artificial Intelligence (EAI) and can have applications, including telerehabilitation and Advanced Driver Assistance Systems (ADAS). This paper focuses on enhancing the performance of the customer feedback system. The authors used the FER-2013 dataset to train the algorithm. VGG16 is a CNN-based architecture designed 
explicitly for classification and localization, and the model is trained with 7 class classifications of emotions, including Happy, Neutral, Sad, Disgusted, Surprised, Fearful, and Angry. The real-time image is captured using the Caffe Model face detector and validated using the trained emotion detection model. The 
web interface displays the emotion detected with corresponding ratings. The collected feedback is stored as a log file with the 
date, time, rating, and emotion detected once the customer presses the submit button. This convenient consumer feedback system reduces the customer's response time and helps retailers improve their services.  
**Keywords** -  EAI, VGG16, customer feedback, emotion detection, facial expression, Machine learning.

## Publication
This paper was published in the proceedings of the IEEE 3rd International Conference on Innovations in Technology (INOCON) 2024.

## Repository Structure
### Emotion_Model
- **confusion.py**: Code for getting confusion matrix.
- **test.py**: Code for testing the trained Model.
- **train.py**: Code for training the Model.
- **data**: Consists of dataset zip file(FER-2013.zip).
- 

