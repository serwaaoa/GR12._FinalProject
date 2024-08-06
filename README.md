### Nigerian food Classification\n
### Group12_Intro_to_AI\n
## 2024-08-05\n
## Nigerian Foods and Snacks Image Classifier using InceptionV3\n

## Overview\n

This project aims to classify images of Nigerian foods and snacks using the InceptionV3 architecture. The classifier provides predictions along with additional nutritional and geographical information about the food. We used the InceptionV3 model in this project.\n\n

## Project Description \n

The project leverages the InceptionV3 model to classify Nigerian foods and snacks into various categories. The model is fine-tuned to provide high accuracy and includes additional features like data augmentation and class weight adjustments to handle imbalanced data.\n

## Dataset\n\n

The dataset used in this project can be downloaded from Kaggle. It contains images of various Nigerian foods and snacks divided into training, validation, and test sets.\n

## Requirements \n\n

- Python 3.12\n
- TensorFlow 2.17.0\n
- Streamlit\n
- Other dependencies listed in `requirements.txt`\n

## Installation \n
1. Clone the repository:\n


git clone https://github.com/yourusername/nigerian-foods-and-snacks-classifier.git \n
cd nigerian-foods-and-snacks-classifier \n\n


2. Create a virtual environment: \n


python -m venv venv \n
source venv/bin/activate  # On Windows use `venv\Scripts\activate` \n
Install dependencies:\n


pip install -r requirements.txt\n
Download the dataset from Kaggle and place it in the data/ directory as specified in the project structure.\n\n

## Usage \n
# Training the Model \n
1. Run the training script:\n
python model.py\n
This will preprocess the data, fine-tune the InceptionV3 model, and save the trained model.\n\n

# Running the Application Locally \n
1. Start the Streamlit application:\n

streamlit run app.py\n
2. Open your browser and go to http://localhost:8501 to use the application.\n\n

# Deploying to the Cloud \n
Using Streamlit Sharing\n
1. Push your code to a public GitHub repository.\n

2. Go to Streamlit Sharing and sign in with your GitHub account.\n

3. Click "New app" and select your GitHub repository, branch, and app.py file.\n

4. Click "Deploy" to deploy your application.
If you would like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!
