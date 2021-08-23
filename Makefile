train: 
	@c++ -std=c++11 train.cpp NeuralNetwork/NeuralNet.cpp -o main
	@./main
	@python plot.py
	@rm main

test:
	@c++ -std=c++11 test.cpp NeuralNetwork/NeuralNet.cpp -o main
	@./main