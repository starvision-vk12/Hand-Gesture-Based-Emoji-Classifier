# Hand-Gesture-Based-Emoji-Classifier
This project is a simple yet powerful Emoji Classifier that detects the type of emoji based on hand gestures using a Convolutional Neural Network (CNN). It leverages PyTorch for deep learning model training and OpenCV for real-time image capture and processing. 
The project was undertaken for learning purposes, providing valuable insights into image processing and deep learning techniques.

-Project Overview
Key Features:
Real-time Hand Gesture Detection: Captures hand gestures using a webcam and processes them in real-time.
Deep Learning Model: Utilizes a Convolutional Neural Network to classify hand gestures into corresponding emojis.
Interactive and User-Friendly: Designed to be intuitive and engaging, with live feedback on gesture recognition.

--> Project Workflow

1. Data Collection
Hand gesture images are collected to create a training dataset.
These images are labeled according to the gesture they represent.
Preprocessing ensures the images are suitable for training the CNN model.

2. Data Preprocessing
Images are converted to grayscale, normalized, and resized to 50x50 pixels.
Continuous capture of multiple hand gesture images until a certain limit is reached.
Each image’s pixels are stored in a CSV file.
The dataset is split into training, test, and validation sets, and corresponding CSV files are created.

3. Model Design
A Convolutional Neural Network (CNN) is designed for gesture classification.
The network includes:
Convolutional Layers: Extract features with varying kernel sizes, strides, and padding.
Activation Functions: ReLU and Sigmoid introduce non-linearity.
Pooling Layers: Reduce spatial dimensions of feature maps.
Fully Connected Layers: Make predictions based on extracted features.
Dropout Layers: Prevent overfitting by randomly dropping units during training.

4. Model Training
The CNN model is trained using the preprocessed dataset, involving:
Splitting the dataset into training and validation sets.
Using Adam optimizer and CrossEntropyLoss for weight adjustment.
Iteratively feeding batches of training data into the model, calculating loss, performing backpropagation, and updating weights.
Evaluating model performance on the validation set and adjusting hyperparameters as needed.

5. Model Evaluation
Performance is evaluated on a separate test set to measure accuracy.
Ensures the model generalizes well to new, unseen data without overfitting.

6. Model Deployment
The trained model is saved and can be integrated into a real-time application.
- The application:
Captures live video using a webcam.
Processes video frames to isolate the hand region using HSV color space conversion, Gaussian blur, and morphological transformations.
Detects contours to find the hand in the frame.
Makes predictions on the extracted hand region using the trained CNN model.
Overlays the corresponding emoji on the video feed based on the prediction.

--> Key Findings
Model Performance
Kernel Sizes and Strides: Larger kernel sizes and strides with appropriate padding performed better, maintaining high accuracy.
--> Activation Functions:
-ReLU: Consistently high accuracy (98-99%), best overall performance.
-Sigmoid: Lower and more variable accuracy (96-97%), slower convergence due to vanishing gradient problem.
-Tanh: Better than Sigmoid but slightly worse than ReLU (97-98%).
--> Conclusion:-
Larger kernels and strides with same padding yield high accuracy.
Combining different activation functions can enhance model performance.
ReLU is the most effective activation function for this deep network.

--> Learning Experience:-
This project provided valuable hands-on experience with deep learning and real-time image processing. It helped deepen my understanding of data collection, preprocessing, model training, and deployment, culminating in a practical application that bridges theoretical knowledge with real-world utility.
