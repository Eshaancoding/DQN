//
//  NeuralNet.hpp
//  NeuralNet
//
//  Created by Eshaan Arnav on 9/7/20.
//  Copyright © 2020 Eshaan. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <iostream> 
#include <time.h>
#include <math.h>
#include <fstream>
#include <cstring>
using namespace std;

#endif /* NeuralNet_hpp */

class Layer {
public:
    double* weights;
    double* bias;
    int starting_node;
    int ending_node;
    double e = 2.7182818;
    double lr;

    Layer (int starting_node, int ending_node, double lr);
    vector<double> predict (vector<double> input);
    vector<double> backprop(vector<double> inputs, vector<double> outputs, vector<double> target); 
    
    void free_vars ();
    double sigmoid (double x);
};

class NeuralNet {
private:
    double mean (vector<double> array);
    vector<double> subtract (vector<double> arrayOne, vector<double> arrayTwo);
    vector<double> square (vector<double> arrayOne);
	vector<double> multiply (vector<double> arrayOne, double two);
public:
    Layer *layers;
    int layout_size = 0;
    int output_layout;
    double lr;
	// class / vars initialization
    NeuralNet (vector<int> layout, double lr);
    
    NeuralNet ();

	// Forward propagate
	vector<double> predict (vector<double> input);
	
	// Performs stochastic gradient descent and applies the gradient to the weights
    double backprop (vector<double> input, vector<double> expected_output);
	
	// If you can find the gradient of whatever function you have with respect to the output of the neural network, this function will figure out the derivative of the weights and bias and apply them  }
	void apply_grad (vector<double> input, vector<double> grad); 

    void save_params (string filename);
    
    void open_params (string filename);
}; 