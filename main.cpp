#include "Agent.cpp"
#include "Pong.cpp"
#include <iostream>
#include <stdio.h>
using namespace std;
// Parameters
int BATCHES = 32; // number of training data trained per frame 
double LR = 0.0005; // 0.0005 Learning rate for the neural networks
int MEM_CAP = 1000000; // 1 000 000 replay mem capacity
int FRAME_REACH = 5000; // 1000 frames till epsilon reaches 0.1
int TARGET_UPDATE = 1000; // 1 000 frames for target net to update with net
vector<int> layout ({8, 150, 130, 3}); // Neural Network layout
int gameWidth = 3;
int gameHeight = 7; // if you want to alter width or height try to play around with the reward system
// main
int main () {
    Pong game = Pong(gameWidth, gameHeight); // PASS
    Agent agent = Agent(layout, LR, MEM_CAP, FRAME_REACH, TARGET_UPDATE, BATCHES); 
    vector<double> current_state = game.return_state(); 
    int max_score = 0;
    while (true) {
        int action = agent.action(current_state);
        int reward = game.act(action);
        vector<double> next_state = game.return_state();
        // print
	system("clear");
	cout<<"Max score: "<<max_score<<" Score: "<<game.score<<endl;
	cout<<"Reward: "<<reward<<endl;
	cout<<"Episodes: "<<game.episodes<<" Frames: "<<agent.frames<<endl;
	if (game.score > max_score) {
		max_score = game.score; 
	}
	agent.store_mem(current_state, action, reward, next_state, game.is_done);
        agent.train();
        current_state = next_state; 
    }    
}
