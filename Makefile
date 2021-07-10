train: 
	@c++ -std=c++11 train.cpp NeuralNet.cpp -o main
	@./main

test:
	@c++ -std=c++11 test.cpp NeuralNet.cpp -o main
	@./main