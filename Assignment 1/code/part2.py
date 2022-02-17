import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

class Linear_Regression:
    def __init__(self):
        self.regressor = linear_model.LinearRegression()
    
    #Performs Linear Regression using sklearn library
    def linear_regression(self, x_train, y_train):
        self.regressor.fit(x_train, y_train)

    #Predicts output values from given input using trained model
    def predict(self, x):
        return self.regressor.predict(x)

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

    #Create Linear_Regression class object
    lr = Linear_Regression()
    
    #Perform Linear Regression using sklearn
    lr.linear_regression(x_train, y_train)

    #Get prediction values for testing data
    predictions = lr.predict(x_test)

    print(f"Testing data R2 score : {r2_score(y_test, predictions)}")

    print(f"Testing data Mean Squared Error : {mean_squared_error(y_test, predictions)}")



