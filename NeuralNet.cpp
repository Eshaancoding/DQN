//
//  NeuralNet.cpp
//  NeuralNet
//
//  Created by Eshaan Arnav on 9/7/20.
//  Copyright © 2020 Eshaan. All rights reserved.
//

#include "NeuralNet.hpp"

Layer :: Layer(int starting_node, int ending_node, double lr) {
    this->lr = lr;
    this->ending_node = ending_node;
    this->starting_node = starting_node;
	// random num    
	srand(time(NULL));
	// init weight
    this->weights = (double*)malloc(sizeof(double) * ending_node * starting_node);
    for (int i = 0; i < ending_node * starting_node; i++) {
        double random_num = (rand() % 1000000) / double(1000000) - 0.5;
        weights[i] = random_num;
    }
    // init bias
    this->bias = (double*)malloc(sizeof(double) * ending_node);
    for (int i = 0; i < ending_node; i++) {
        double random_num = (rand() % 1000000) / double(1000000) - 0.5;
        bias[i] = random_num;
    }
} 

double Layer :: sigmoid (double x) {
    return 1 / (1 + pow(this->e, -x));
}

vector<double> Layer :: predict (vector<double> input) {
    if (input.size() != this->starting_node) {
        throw invalid_argument("Input size not equal to input array. Expected " + to_string(this->starting_node) + " but got: " + to_string(input.size()));
    }
    
    vector<double> output;
    // fill output with zero
    for (int i = 0; i < this->ending_node; i++) {
        output.push_back(0);
    }
    // weight
    for (int x = 0; x < this->ending_node; x++) {
        for (int i = 0; i < this->starting_node; i++) {
            output[x] += input[i] * weights[(i * this->ending_node) + x];
        }
    }
    // bias & sigmoid
    for (int i = 0; i < this->ending_node; i++) {
        output[i] += bias[i]; // bias
        output[i] = this->sigmoid(output[i]);
    }
    return output;
}


vector<double> Layer :: backprop(vector<double> inputs, vector<double> outputs, vector<double> target) {
    vector<double> return_array;
    if (inputs.size() != this->starting_node) throw invalid_argument("input size not valid");
    if (outputs.size() != this->ending_node) throw invalid_argument("output size not valid");
    if (target.size() != this->ending_node) throw invalid_argument("target size not valid");

    for (int i = 0; i < starting_node; i++) {
        return_array.push_back(0);
    }
    for (int x = 0; x < this->ending_node; x++) {
        double output = outputs[x];
        for (int i = 0; i < this->starting_node; i++) {
            double input = inputs[i];
            
			// derivative of the weights 
            double dir_w = input * output * (1 - output) * target[x];
            // applying dir to weights
            weights[(i * this->ending_node) + x] -= this->lr * dir_w;
            
			// derivative of "first layer" (the derivative to be passed down to the next layer)
            double dir_a_l_1 = output * (1 - output) * target[x];
			// multiply by weights
            dir_a_l_1 *= weights[(i * this->ending_node) + x];
			// accumulate the derivative
            return_array[i] += dir_a_l_1;
        }

        // derivative of the bias
        double dir_b = output * (1 - output) * target[x]; 
        // apply der to bias 
		bias[x] -= this->lr * dir_b;
    }
    return return_array;
}

void Layer :: free_vars() {
    free(this->weights);
    free(this->bias);
}

NeuralNet :: NeuralNet () {
    
}

NeuralNet :: NeuralNet(vector<int> layout, double lr) {
    this->lr = lr;
    this->layout_size = int(layout.size());
    this->output_layout = layout[layout_size - 1];
    // malloc layers
    layers = (Layer*)malloc(sizeof(Layer) * layout_size);
    // init layers
    for (int i = 0; i < this->layout_size - 1; i++) {
        layers[i] = Layer(layout[i], layout[i+1], this->lr);
	}
    if (this->layout_size == 0) throw invalid_argument("layout size 0");
}

vector<double> NeuralNet :: predict (vector<double> input) {
    vector<double> return_array = input;
    for (int i = 0; i < this->layout_size - 1; i++) {
		return_array = this->layers[i].predict(return_array);
    }
    return return_array;
}

double NeuralNet:: mean (vector<double> array) {
    double return_num = 0;
    for (int i = 0; i < array.size(); i++) {
        return_num += array[i];
    }
    return return_num / array.size();
}

vector<double> NeuralNet :: subtract (vector<double> arrayOne, vector<double> arrayTwo) {
    vector<double> return_array;
    for (int i = 0; i < arrayOne.size(); i++) {
        return_array.push_back(arrayOne[i] - arrayTwo[i]);
    }
    return return_array;
}

vector<double> NeuralNet:: square (vector<double> arrayOne) {
    vector<double> return_array;
    for (int i = 0; i < arrayOne.size(); i++) {
        return_array.push_back(pow(arrayOne[i], 2));
    }
    return return_array;
}

vector<double> NeuralNet:: multiply (vector<double> arrayOne, double two) {
	vector<double> return_arr; 
	for (int i = 0; i < arrayOne.size(); i++) {
		return_arr.push_back(arrayOne[i] * two);
	}
	return return_arr;
}

double NeuralNet:: backprop (vector<double> input, vector<double> expected_output) {    
    vector<double> final_prediction = input;
    vector<double> prediction_array[this->layout_size];
    prediction_array[0] = input;
    for (int i = 0; i < this->layout_size - 1; i++) {
        final_prediction = this->layers[i].predict(final_prediction);
        prediction_array[i+1] = final_prediction; 
    }
	// MSE
	vector<double> error = multiply(subtract(final_prediction, expected_output),2);
    for (int i = this->layout_size - 2; i > -1; i--) {
        error = this->layers[i].backprop(prediction_array[i], prediction_array[i+1], error);
    }
    return mean(square(subtract(final_prediction, expected_output)));
}

// basically the same as NeuralNet::backprop except the error is the grad
void NeuralNet:: apply_grad (vector<double> input, vector<double> grad) {
	// forward propagate
	vector<double> output_array[this->layout_size];
	output_array[0] = input; 
	for (int i = 0; i < this->layout_size - 1; i++) {
		output_array[i+1] = this->layers[i].predict(output_array[i]);
	}	
	// backward propagate
	vector<double> error = grad;
	for (int i = this->layout_size - 2; i > -1; i--) {
		error = this->layers[i].backprop(output_array[i], output_array[i+1], error);
	}
} 