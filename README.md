# Fruit and Nutrient Classifier: Project Overview
- Created a tool that takes an image of a fruit and classifies it into one of 131 different categories while also outputting the nutrients that fruit provides
- Used over 90,000 images(obtained from Kaggle's fruit360 dataset) of different fruits to train and test two different models using Tensorflow, CNNs, and data augmentation
- Used rapidAPI to acquire information on the nutrients each fruit provides
- Deployed the model using Google Cloud Platform (and Google Cloud Functions)
- Tested both deployment methods using the Postman application

## Data Collection and Preprocessing
- Downloaded the dataset from [kaggle](https://www.kaggle.com/moltean/fruits)
- The dataset comes with a training and test folder for images, the test folder was split in half into a test and validation set
- For both models, the data was cached data for quicker subsequent iterations, resized and rescaled for data preprocessing
- The second model had the training set augmented by randomly flipping and rotating different images

## Model Building and Evaluation
- The CNN model was created using a combination of Conv2D, MaxPooling2D, Flatten, and Dense layers
- The accuracy score when evaluating the test set came back at 94%

## Model Deployment
The model was deployed using Google Cloud Services (GCP) and tested using the Postman Application. When the function is called it makes a request to the rapidAPI after predicting the fruit from an image. The function returns the predicted fruit, the confidence level, and the nutrients associated with the fruit

Below is an image of a lemon provided to Postman with the URL to the model in GCP:

![43_100](https://user-images.githubusercontent.com/74473048/156256544-953abe8e-9160-42e5-bd4c-6c998f65eba2.jpg)

Below is the response that the model returned using Postman:
![postman fruit reponse](https://user-images.githubusercontent.com/74473048/156256934-10f52ca0-a846-4214-8bb0-cae96120485e.JPG)
