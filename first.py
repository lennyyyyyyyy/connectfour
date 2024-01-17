import numpy as np 
# from matplotlib import pyplot as plt 
input_vector = [1.66, 1.56]
test_set = [input_vector, 1]
weights_1 = [1, 1]
bias = 0.25
learning_rate = 0.01

# sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
def prediction(input):
    # layer 1
    layer_1_result = np.dot(input_vector, weights_1)
    layer_1_result += bias

    # layer 2
    layer_2_result = sigmoid(layer_1_result)
    return layer_2_result
def update(prediction, target, layer1):
    derror_prediction = (prediction - target)
    dprediction_layer1 = sigmoid(layer1) * (1-sigmoid(layer1))
    dlayer1result_bias = 1
    derror_bias = derror_prediction*dprediction_layer1*dlayer1result_bias
    dlayer1_weights1 = input_vector
    derror_weights1 = derror_prediction*dprediction_layer1*dlayer1_weights1
    bias -= 0.01*derror_bias
    

print(prediction(input_vector))
mse = np.square(prediction(test_set[0]) - test_set[1])
print(mse)