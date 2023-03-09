# Neural Network Project: Predicting NFL Game Winners
This project aims to predict the winner of National Football League (NFL) games based on the home and away team and their previous wins and losses. We utilized a neural network model trained on data from September 4th, 2014 to the most recent Super Bowl when the Kansas City Chiefs won.

## Project Overview
We created a neural network model to predict the winner of NFL games based on the following features:

* Home team
* Away team
* Number of wins for the home team
* Number of wins for the away team
* The model was trained on historical data from NFL games and tested on a separate validation dataset to evaluate its performance.

## Installation and Usage
### Requirements
* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow

## Installation
1. Clone the repository to your local machine using the following command:
git clone https://github.com/your_username/nfl_predictions_running_colab.git
2. Navigate to the project directory:
cd nfl_predictions_running_colab
3. Install the required libraries using pip:
pip install -r requirements.txt

## Usage
1. Open the nfl_predictions_running_colab.ipynb file in Jupyter Notebook or Google Colab.
2. Upload the NFL_data.csv file to the notebook environment.
3. Run the notebook to train and evaluate the model.

## Packages and Libraries Used
* pandas: Data manipulation and analysis library
* numpy: Numerical computing library
* matplotlib: Data visualization library
* scikit-learn: Machine learning library
* tensorflow: Deep learning library

## Imports
The following Python libraries were used in this project:

* import pandas as pd
* import numpy as np
* from pathlib import Path
* import matplotlib.pyplot as plt
* from sklearn import svm
* import tensorflow as tp
* from tensorflow.keras.layers import Dense
* from tensorflow.keras.models import Sequential
* from sklearn.preprocessing import StandardScaler
* from pandas.tseries.offsets import DateOffset
* from sklearn.metrics import classification_report
* from sklearn.preprocessing import OneHotEncoder
* from sklearn.model_selection import train_test_split

## Analysis Summary and Results
After experimenting with two different models, we were able to achieve an accuracy of 0.6431 in our predictions by adjusting the epoch from 50 to 100.

This is a significant result, as it demonstrates the effectiveness of using neural network models in predicting the outcomes of sports games. Our model's accuracy can help provide valuable insights to fans, analysts, and even sports betting enthusiasts who are interested in predicting game results with greater accuracy.

It is worth noting that the accuracy achieved in this project is not perfect, and there may be other factors that could impact game outcomes that are not accounted for in our model. However, our results provide a strong foundation for future work in this area, and we are excited to see how this research can be further developed and applied in the future.

Overall, we are proud of the work we have done in creating and testing these models, and we believe that our findings have important implications for the world of sports analytics and beyond.


## Visualizations

## Notes
* The neural network models in this project use a basic architecture with a few dense layers and ReLU activation function. You can modify the models by adding or removing layers, changing the activation function, or adjusting the hyperparameters.
* The performance of the models may vary depending on the quality and quantity of the training data, as well as on the specific games and seasons being predicted. Therefore, we recommend using the models as a complementary tool for making informed decisions, rather than relying solely on their predictions.
* This project was developed in Google Colab, which provides free GPU and TPU resources for training deep learning models. You may need to adapt the code to run on your own hardware or cloud platform.

## Contributors
- Tyler Goering
- Terrence Mccoy
- Jacob Macpherson
