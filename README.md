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
- **face**: Consists of caffe model face detector files.
- **model**: Consists of trained model .h5 files.
### Training_Results
Explore the performance metrics and insights gained during the training of our emotion detection model. This section includes detailed results such as the classification report, loss and accuracy graphs plotted against epochs, and the confusion matrix. These results provide a comprehensive view of the model's learning process and effectiveness in classifying emotions.
### Validate_Results
Witness the real-world application of our trained model through examples of testing with live video feed. These validation results showcase the model's ability to detect and classify emotions in real-time scenarios, highlighting its practical utility and performance in capturing emotion effectively.
### Videos
Explore the interactive features of our customer feedback system through these video demonstrations. Witness the real-time emotion detection and feedback submission process, showcasing the system's ease of use and effectiveness in capturing customer emotion and rating.
### Web_Interface
- **face**: Consists of caffe model face detector files.
- **feedback**: Consists of feedback text file created by web interface.
- **templates**: consists of html file for web interface.
- **app.py**: Code for iplementing web interface.
- **my_model.h5**: Trained model.

**Note:** To run the web interface, please ensure that the files are placed in the same hierarchy order and execute the app.py file.
### Web_Results 
Browse through a collection of images tested with different persons to observe the emotion detection capabilities of our system. These images showcase the system's performance in accurately identifying and classifying emotions across various individuals, highlighting its robustness and effectiveness in real-world scenarios.

**Note:** requirements.txt consists of python packages required for the project.

## Citation
If you find this repo useful in your project or research, please consider citing the publication.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
