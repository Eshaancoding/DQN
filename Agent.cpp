#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "NeuralNetwork/NeuralNet.hpp"
using namespace std;
class ReplayMemory {
private: 
    vector<vector<vector<double>>> mem;
    int capacity;
public:
    vector<double> current_state;
    int action;
    int reward;
    vector<double> next_state;
    bool is_done;
    ReplayMemory () {}
    ReplayMemory (int capacity) {
        this->capacity = capacity;
        srand(time(NULL));
    }
    void store (vector<double> current_state,
		    int action,
		    int reward,
		    vector<double> next_state, 
		    bool is_done) {
        if (this->mem.size() == capacity) {
            this->mem.erase(this->mem.begin()); 
        }
        
        vector<vector<double>> append_mem;
        append_mem.push_back(current_state);
        append_mem.push_back({(double)action});
        append_mem.push_back({(double)reward});
        append_mem.push_back(next_state);
        append_mem.push_back({(double)is_done});
        this->mem.push_back(append_mem); 
    }
    void random () {
        int random_num = rand() % this->mem.size();
        vector<vector<double>> mem_read = this->mem[random_num];
        this->current_state = mem_read[0];
        this->action = (int)mem_read[1][0];
        this->reward = (int)mem_read[2][0];
        this->next_state = mem_read[3]; 
        this->is_done = (bool)mem_read[4][0]; 
    }  
}; 

class Agent {
private:
    NeuralNet net = NeuralNet(); 
    NeuralNet target_net = NeuralNet();
    ReplayMemory mem = ReplayMemory();
    int frameReachProb;
    int batches;
    int targetFreqUpdate;
    bool is_testing = false;
public:
    vector<double> last_prediction;
    int frames = 0; 

    Agent (vector<int> layout, string filename) {
        this->net = NeuralNet(layout, -1); // the learning rate doesn't matter 
        this->net.open_params(filename);
        this->target_net = this->net;
        this->is_testing = true;
    }

    Agent (vector<int> layout, double lr, int mem_capacity, int frameReachProb, int targetFreqUpdate, int batches) {
        this->mem = ReplayMemory(mem_capacity);
        this->net = NeuralNet(layout, lr);
        this->target_net = net;
        this->frameReachProb = frameReachProb;
        this->targetFreqUpdate = targetFreqUpdate; 
        this->batches = batches; 
        srand(time(NULL));
    }
    int argmax (vector<double> array) {
        int index = 0;
        double max_value = array[0];
        for (int i = 1; i < array.size(); i++) {
            if (array[i] > max_value) {
                max_value = array[i];
                index = i;
            }
        }
        return index; 
    }    

    int action (vector<double> input) {
        if (!is_testing) {
            this->frames++;
            double probability;
            if (frames <= frameReachProb) {
                probability = (-0.95 / double(frameReachProb)) * frames + 1;
            } else {
                probability = 0.05; 
            }
            bool isRandom = (rand() % 100) < (probability * 100);
            int action;
            if (isRandom) { 
                action = rand() % 3;
                last_prediction = vector<double>({-1, -1, -1});
            } else {
                last_prediction = this->net.predict(input); 
                action = argmax(last_prediction);
            }
            return action;
        } else {
            last_prediction = this->net.predict(input); 
            return argmax(last_prediction);
        }
    }
    void store_mem (vector<double> current_state, int action, int reward, vector<double> next_state, bool is_done) {
        if (is_testing) throw invalid_argument("Cannot use this function while testing");
        mem.store(current_state, action, reward, next_state, is_done);
    } 
    double max (vector<double> array) {
        double max_val = array[0];
        for (int i = 1; i < array.size(); i++) {
            if (array[i] > max_val) {
                max_val = array[i];
            }
        }
        return max_val;
    }
    void train () {
        if (is_testing) throw invalid_argument("Cannot use this function while testing");
        // sample minibatch
        for (int i = 0; i < batches; i++) {
            mem.random(); 
            vector<double> current_state = mem.current_state;
            int action = mem.action;
            int reward = mem.reward;
            vector<double> next_state = mem.next_state; 
            bool is_done = mem.is_done;
            // train
            double y;
            if (is_done) {
                y = reward;
            } else {
                y = reward + (0.99 * max(this->target_net.predict(next_state)));
            }
            vector<double> target = this->net.predict(current_state);
            target[action] = y; 
            this->net.backprop(current_state, target);
        }
        if (frames % this->targetFreqUpdate == 0) {
            this->target_net = this->net; 
            this->net.save_params("model.txt"); // yes im using .txt file dont bully me
        }
    }
};
