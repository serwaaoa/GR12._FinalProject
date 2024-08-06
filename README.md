### Nigerian food Classification
### Group12_Intro_to_AI
## 2024-08-05
## Nigerian Foods and Snacks Image Classifier using InceptionV3
##link to youtube video: https://youtu.be/BcRLHNQAsFE

## Overview

This project aims to classify images of Nigerian foods and snacks using the InceptionV3 architecture. The classifier provides predictions along with additional nutritional and geographical information about the food. We used the InceptionV3 model in this project.

## Project Description 

The project leverages the InceptionV3 model to classify Nigerian foods and snacks into various categories. The model is fine-tuned to provide high accuracy and includes additional features like data augmentation and class weight adjustments to handle imbalanced data.

## Dataset

The dataset used in this project can be downloaded from Kaggle. It contains images of various Nigerian foods and snacks divided into training, validation, and test sets.

## Requirements 

- Python 3.12
- TensorFlow 2.17.0
- Streamlit
- Other dependencies listed in `requirements.txt`

## Installation 
1. Clone the repository:


git clone https://github.com/yourusername/nigerian-foods-and-snacks-classifier.git 

cd nigerian-foods-and-snacks-classifier 


2. Create a virtual environment: 


python -m venv venv 

source venv/bin/activate  # On Windows use `venv\Scripts\activate` 

Install dependencies:


pip install -r requirements.txt

Download the dataset from Kaggle and place it in the data/ directory as specified in the project structure.

## Usage 
# Training the Model 
1. Run the training script:

python model.py

This will preprocess the data, fine-tune the InceptionV3 model, and save the trained model.

# Running the Application Locally 
1. Start the Streamlit application:

streamlit run app.py

2. Open your browser and go to http://localhost:8501 to use the application.

# Deploying to the Cloud 
Using Streamlit Sharing\n

1. Push your code to a public GitHub repository.

2. Go to Streamlit Sharing and sign in with your GitHub account.

3. Click "New app" and select your GitHub repository, branch, and app.py file.

4. Click "Deploy" to deploy your application.

Link to Youtube: -- youtube link --

Link to trained models: https://drive.google.com/drive/folders/1--wpXlHY3UcPufKIRK1AsfJRpF1XN5If?usp=sharing

If you would like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!
