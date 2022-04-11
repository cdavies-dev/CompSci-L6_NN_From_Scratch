import sys
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import seaborn as sb

# a simple 30x15x2 neural network demonstrating backpropagation from scratch using the Wisconsin Breast Cancer Dataset
# this has been dynamic using variables to account for changing hidden layer neuron numbers

class NeuralNetwork:

    def __init__(self, file, hidden_neurons, epochs, learning_rate):

        self.file = file
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

    def preprocess(self, file):

        # preprocessing is somewhat hardcoded for this project but adaptability could be improved to account for more layers and varying input sizes

        # read data from file to DataFrame
        dataset = pd.read_csv(file, header=None)

        # drop ID column
        dataset = dataset.drop(dataset.columns[0], axis=1)

        # new matrix for class label encoding, -- set to int for 0 switch to -1?
        classifier = pd.get_dummies(dataset[1]).astype(int)  # .replace(0,-1)

        # min max norm of dataset, set to range 0-1, re-insert class values, reset index
        dataset_normed = self.normalise(dataset.iloc[:, 1:31])
        dataset_normed.insert(loc=0, column=32, value=classifier.iloc[:, 0])
        dataset_normed.insert(loc=0, column=33, value=classifier.iloc[:, 1])
        dataset = dataset_normed.T.reset_index(drop=True).T

        # rename columns for visual
        dataset.rename(columns={
            0: 'diagnosis_1',
            1: 'diagnosis_2',
            2: 'radius_mean',
            3: 'texture_mean',
            4: 'perimeter_mean',
            5: 'area_mean',
            6: 'smoothness_mean',
            7: 'compactness_mean',
            8: 'concavity_mean',
            9: 'concave_points_mean',
            10: 'symmetry_mean',
            11: 'fractal_dimension_mean',
            12: 'radius_sqerr',
            13: 'texture_sqerr',
            14: 'perimeter_sqerr',
            15: 'area_sqerr',
            16: 'smoothness_sqerr',
            17: 'compactness_sqerr',
            18: 'concavity_sqerr',
            19: 'concave_points_sqerr',
            20: 'symmetry_sqerr',
            21: 'fractal_dimension_sqerr',
            22: 'radius_worst',
            23: 'texture_worst',
            24: 'perimeter_worst',
            25: 'area_worst',
            26: 'smoothness_worst',
            27: 'compactness_worst',
            28: 'concavity_worst',
            29: 'concave_points_worst',
            30: 'symmetry_worst',
            31: 'fractal_dimension_worst',
        }, inplace=True)

        # define random training and testing sets, 70/30% split
        training_data = dataset.sample(
            frac=0.7, random_state=200)  # random_state = seed

        testing_data = dataset.drop(training_data.index)

        # divide training/testing inputs and class, split class from inputs, convert to np.array
        training_inputs = training_data.iloc[:, 2:32].to_numpy()
        training_class = training_data.iloc[:, 0:2].to_numpy()
        testing_inputs = testing_data.iloc[:, 2:32].to_numpy()
        testing_class = testing_data.iloc[:, 0:2].to_numpy()

        return training_inputs, training_class, testing_inputs, testing_class

    def normalise(self, dataset):

        data = dataset.copy()

        min_scaler = 0.8
        max_scaler = 1.2

        for i in data.columns:
            # min max using proportional population scalers 1.2 and 0.8
            data[i] = (data[i] - (data[i].min() * min_scaler) / (data[i].max() * max_scaler) - (data[i].min() * min_scaler))

            # normalise between 0 and 1
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

        return data

    def relu(self, data):
        return np.maximum(0, data)

    def relu_derivative(self, data):
        # needs adjusting to fit a matrix
        x = [1 if value > 0 else 0 for value in data]
        return x

    def softmax(self, data):
        exp_values = np.exp(data - np.max(data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return probabilities

    def softmax_derivative(self, data):
        None  # ??? -- to be amended using code in training()

    def training(self, train_X, train_Y, epochs, hidden_neurons, learning_rate):

        # hidden layer init
        hidden_layer = np.zeros((1, hidden_neurons))

        # weights matrices to be transposed
        input_hidden_weights = 2 * \
            np.random.random((hidden_neurons, len(train_X[0]))) - 1
        hidden_output_weights = 2 * \
            np.random.random((len(train_Y[0]), hidden_neurons)) - 1

        # column vectors for layer biases
        hidden_bias = np.zeros((1, hidden_neurons))
        output_bias = np.zeros((1, len(train_Y[0])))

        # mean squared error list init
        mse = []

        for i in range(epochs):

            accuracy = 0
            mean_squared_error_sum = 0
            mean_error_sum = 0
            learning_rate = self.learning_rate
            sum = 0

            # forward propagation
            for j in range(len(train_X)):

                input_layer = train_X[j].reshape(1, len(train_X[0]))
                target_output = train_Y[j].reshape(1, len(train_Y[0]))

                hidden_layer = np.dot(
                    input_layer, input_hidden_weights.T) + hidden_bias
                hidden_layer_activated = self.relu(hidden_layer)

                actual_output = np.dot(
                    hidden_layer_activated, hidden_output_weights.T) + output_bias
                actual_output_activated = self.softmax(actual_output)

                # categorical cross-entropy loss function
                target_output = target_output.flatten()
                actual_output_activated = actual_output_activated.flatten()
                actual_output = actual_output.flatten()

                # eliminating potential zeros causing infinity outputs
                actual_output_clipped = np.clip(
                    actual_output_activated, 1e-7, 1-1e-7)
                error = -(math.log(actual_output_clipped[0]) * target_output[0] + math.log(
                    actual_output_clipped[1]) * target_output[1])  # error calc coded out as not using one hot encoding
                #error = -(math.log(actual_output_activated[0]) * target_output[0] + math.log(actual_output_activated[1]) * target_output[1])

                # calculating accuracy using argmax()
                predictions = np.argmax(actual_output_activated)
                accuracy += np.argmax(predictions == target_output)

                # for tracking mean squared error for plot
                mean_squared_error_sum += (error ** 2) / len(train_X)
                # for tracking the mean of errors across each row of data in each epoch
                mean_error_sum += error

            mse.append(mean_squared_error_sum)
            # mean cost per epoch, used for calculating deltas and backpropagation
            cost_per_epoch = mean_error_sum / len(train_X)
            # tracking mean accuray per epoch
            accuracy_per_epoch = accuracy / len(train_X)

            print('Epoch: ' + str(i) + ', MSE: ' + str(mean_squared_error_sum) +
                  ', Accuracy: ' + str(accuracy_per_epoch))

            # backpropagation

            #sum = (actual_output[0] + actual_output[1]) / len(actual_output)
            #derivative_from_next_layer = 1.0
            #relu_derivative = derivative_from_next_layer * (1.0 if sum > 0 else 0.0)

            # jacobian matrix of partial derivatives for each softmax output with respect to each input
            #jacobian_matrix = np.diagflat(actual_output_clipped) - np.dot(actual_output_clipped, actual_output_clipped.T)
            #dinputs = np.empty_like[dvalues]

            # softmax derivative?? -- add to function above
            jacobian_matrix = np.diag(actual_output_clipped)
            for i in range(len(jacobian_matrix)):
                for j in range(len(jacobian_matrix)):
                    if i == j:
                        jacobian_matrix[i][j] = actual_output_clipped[i] * \
                            (1 - actual_output_clipped[i])
                    else:
                        jacobian_matrix[i][j] = - \
                            actual_output_clipped[i] * actual_output_clipped[j]
            derivative_output = jacobian_matrix.sum() / len(jacobian_matrix) ** 2
            delta_output = cost_per_epoch * derivative_output

            # deltas - matrices need adjusting to fit against eachother
            delta_output_weight = np.dot(delta_output, hidden_output_weights.T)
            delta_hidden_layer = delta_output_weight.T * \
                self.relu_derivative(
                    hidden_layer)  # code for relu der needs defining
            input_delta1 = np.dot(delta_output, hidden_layer.T)
            input_delta0 = np.dot(input_layer.T, delta_hidden_layer)

            hidden_output_weights = hidden_output_weights + \
                (input_delta1 * learning_rate)
            input_hidden_weights = input_hidden_weights + \
                (input_delta0 * learning_rate)

        return mse

    def confusion_matrix(self, actual, expected):

        x = actual
        y = expected

        false_pos = np.zeros((1, 2))
        false_neg = np.zeros((1, 2))
        true_pos = np.zeros((1, 2))
        true_neg = np.zeros((1, 2))

        if y == x:
            if y == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if y == 1:
                false_pos += 1
            else:
                false_neg += 1

        matrix = [[true_neg, true_pos], [false_neg, false_pos]]
        matrix = np.array(matrix)

        # number of values predicted correctly
        accuracy = (true_pos + true_neg) / 568  # number of samples (???)

        # percentage of times a guess was correct out of total predictions
        precision = true_pos / (true_pos + false_pos)

        return print('Accuracy: ' + accuracy), print('\nPrecision: ' + precision)

    def plot(self, mean_squared_error):
        mpl.xlabel('Number of Epochs')
        mpl.ylabel('Error')
        mpl.title('Error Over Eopchs')
        mpl.plot(mean_squared_error)
        mpl.show()

def main():
    NN = NeuralNetwork(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
    # NN = NeuralNetwork('breast_cancer.csv', '15', '2000', '0.1') #set epochs to 2000

    # defining training and testing sets
    train_X, train_Y, test_X, test_Y = NN.preprocess(str(NN.file))

    # returns mean squared error over epochs for plotting
    mean_squared_error = NN.training(train_X, train_Y, int(NN.epochs), int(NN.hidden_neurons), float(NN.learning_rate))

    # confusion_matrix()?? -- call this method to find tp_rate and fp_rate, and accuracy though this is already calculated as an average per epoch in training()

    # plot graph
    NN.plot(mean_squared_error)


if __name__ == '__main__':
    main()