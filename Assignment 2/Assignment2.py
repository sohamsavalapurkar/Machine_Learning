#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################
#
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)
        self.raw_input.columns = ["variance_of_Wavelet", "skewness_of_Wavelet", "curtosis_of_Wavelet", "entropy_of_image", "output"]
        



    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.raw_input.fillna(0)
        self.raw_input.replace("", 0, inplace=True)
        self.processed_data = self.raw_input
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    #In this function we have defined the model and passing hyperparamters as parameters to the function to get models for different hyperparamters
    def model(self, x, y, x_test, activation, learning_rate, epochs, num_hidden_layers):
        model = Sequential()
        if num_hidden_layers == 2:
            model.add(Dense(2, input_dim=x.shape[1], activation=activation))
            model.add(Dense(1, activation=activation))
           
        else:
            model.add(Dense(4, input_dim=x.shape[1], activation=activation))
            model.add(Dense(2, activation=activation))
            model.add(Dense(1, activation=activation))
        
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        history = model.fit(x, y, epochs=epochs, verbose=0, batch_size=64)
        y_pred = model.predict(x_test)
        predictions = []
        for i in y_pred:
            if i > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return history, predictions

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size = 0.8, random_state = 1)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        histories = []
        labels = []
        training_accuracy = []
        testing_accuracy = []
        training_error = []
        testing_error = []
        activation1 = []
        learning_rate1 = []
        max_iterations1 = []
        num_hidden_layers1 = []

        #Training model for all different combinations of hyperparameters
        for activation in activations:
            for learn_rate in learning_rate:
                for max_iter in max_iterations:
                    for num_hidden in num_hidden_layers:
                        history, predictions = self.model(X_train, y_train, X_test, activation, learn_rate, max_iter, num_hidden)
                        histories.append(history)
                        
                        labels.append(f'{activation}, {learn_rate}, {max_iter}, {num_hidden}')
                        
                        training_accuracy.append(history.history['accuracy'][-1])
                        testing_accuracy.append(accuracy_score(y_test, predictions))
                        training_error.append(history.history['loss'][-1])
                        testing_error.append(mean_squared_error(y_test, predictions))

                        activation1.append(activation)
                        learning_rate1.append(learn_rate)
                        max_iterations1.append(max_iter)
                        num_hidden_layers1.append(num_hidden)

        #Creating dataframe to display all the required data
        df = pd.DataFrame({'activation':activation1, 'learning_rate':learning_rate1,'epochs':max_iterations1, 'num_hidden_layers':num_hidden_layers1,'training_accuracy':training_accuracy, 'testing_accuracy':testing_accuracy, 'training_error': training_error, 'testing_error': testing_error})                       
        print(df)
        plt.figure(figsize=(40, 25))
        for i in range(len(histories)):
            plt.plot(histories[i].history['accuracy'], label=labels[i])
        #
        #plt.set_yscale("log")
        plt.legend(labels, prop={'size': 20})
       
        plt.show()
        
        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0


if __name__ == "__main__":
    neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
