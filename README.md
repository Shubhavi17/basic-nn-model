# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
To build a neural network regression model for predicting a continuous target variable, we will follow a systematic approach. The process includes loading and pre-processing the dataset by addressing missing data, scaling the features, and ensuring proper input-output mapping. Then, a neural network model will be constructed, incorporating multiple layers designed to capture intricate patterns within the dataset. We will train this model, monitoring performance using metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE), to ensure accurate predictions. After training, the model will be validated and tested on unseen data to confirm its generalization ability. The final objective is to derive actionable insights from the data, helping to improve decision-making and better understand the dynamics of the target variable. Additionally, the model will be fine-tuned to enhance performance, and hyperparameter optimization will be carried out to further improve predictive accuracy. The resulting model is expected to provide a robust framework for making precise predictions and facilitating in-depth data analysis.

## Neural Network Model
![Screenshot 2024-09-01 234050](https://github.com/user-attachments/assets/a854f1c3-617e-40e8-8349-dc8fe10c6dfc)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SHUBHAVI M
### Register Number:212223040199
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab  import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds,_=default()
gc = gspread.authorize(creds)
worksheet =  gc.open('data').sheet1
data = worksheet.get_all_values()
df=pd.DataFrame(data[1:], columns=data[0])
df=df.astype({'X':'int'})
df=df.astype({'Y':'int'})
df.head()
X = df[['X']].values
Y = df[['Y']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer= 'rmsprop',loss = 'mse')
ai_brain.fit(X_train1,Y_train,epochs= 2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,Y_test)
X_n1 = [[6]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1 )
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,Y_test)
X_n1 = [[6]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1 )



```
## Dataset Information
![Screenshot 2024-09-01 232723](https://github.com/user-attachments/assets/5b023ccb-cbb3-493b-99b2-aefb6fa6362e)


## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-09-01 232753](https://github.com/user-attachments/assets/faabc1e6-ce35-4037-86e4-da10092b41a8)



### Test Data Root Mean Squared Error

![Screenshot 2024-09-01 232816](https://github.com/user-attachments/assets/6c42e285-ccc5-45a7-878f-a878b74964f0)


### New Sample Data Prediction
![Screenshot 2024-09-01 232902](https://github.com/user-attachments/assets/aa3507cc-296b-4aaa-9fb9-b3bd3ed4ed2e)

## RESULT
hus a Neural network for Regression model is Implemented.
