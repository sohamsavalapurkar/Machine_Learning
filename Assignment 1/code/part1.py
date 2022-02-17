import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class Linear_Regression:
    def __init__(self, learning_rate, epochs, theta):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = theta
    
    #Performs batch Gradient Descent
    def GradientDescent(self, x, y):
        m = x.shape[0]
        print(m)
        mse = []
        r2 = []
        for i in range(1, self.epochs+1):
            y_predicted = x.dot(self.theta)
            diff = y_predicted - y
            gradients = x.T.dot(diff)/m
            
            self.theta -= self.learning_rate * gradients
            if i % 100 == 0:
                print(f"epoch {i}")
                print(f"R2 Score : {r2_score(y, y_predicted)}")
                print(f"Cost : {mean_squared_error(y, y_predicted)}\n")
            mse.append(mean_squared_error(y, y_predicted))
            r2.append(r2_score(y, y_predicted))

    #Predicts Passed data using model
    def predict(self,x):
        x = scale_data_and_add_one_layer(x)
        
        return x.dot(self.theta)

#Scales data and adds a column of ones for intercept    
def scale_data_and_add_one_layer(x):
    scalar = StandardScaler()    
    
    x = scalar.fit_transform(x)
    x = np.hstack((np.ones((len(x),1)), x))

    return x


#Removes blank and invalid data
def preprocess_data(data):
    data.fillna(0)
    data.replace("", 0, inplace=True)
    return data

if __name__ == "__main__":
    #Read data from UCI website and preprocess it
    data = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat')
    data.columns = ["Frequency", "Angle of attack", "Chord length", "Free-stream velocity", "Suction side displacement thickness", "Scaled_sound_pressure_level"]
    data = preprocess_data(data)

    #Split Training and testing data
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.Scaled_sound_pressure_level, train_size = 0.8, random_state = 1)

    #Scale training data
    x_train = scale_data_and_add_one_layer(x_train)

    lr = Linear_Regression(learning_rate=0.006, epochs=1500, theta=np.zeros(x_train.shape[1]))

    #Perform Gradient descent on training data
    lr.GradientDescent(x_train, y_train)

    #Get prediction values for testing data
    predictions = lr.predict(x_test)

    print(f"Testing data R2 score : {r2_score(y_test, predictions)}")

    print(f"Testing data Mean Squared Error : {mean_squared_error(y_test, predictions)}")


