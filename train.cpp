#include "Agent.cpp"
#include "Pong.cpp"
#include <iostream>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>
#include <fstream>
using namespace std;

void gotoxy(int x,int y)    
{
    printf("%c[%d;%df",0x1B,y,x);
}

// Parameters
int BATCHES = 32; // number of training data trained per frame 
double LR = 0.001; // 0.001 Learning rate for the neural networks
int MEM_CAP = 10000; // 1 000 000 replay mem capacity
int FRAME_REACH = 10000; // 5 000 frames till epsilon reaches 0.05
int TARGET_UPDATE = 5000; // 1000 frames for target net to update with net, and also saves neural network parameters every 5000 iter
vector<int> layout ({8, 50, 3}); // Neural Network layout
int gameWidth = 7;
int gameHeight = 12; // if you want to alter width or height try to play around with the reward system
int iteration = 0;
// main
int main () {
    Pong game = Pong(gameWidth, gameHeight); // PASS
    Agent agent = Agent(layout, LR, MEM_CAP, FRAME_REACH, TARGET_UPDATE, BATCHES); 
    vector<double> current_state = game.return_state(); 
    int max_score = 0;
    int avg_reward = 0;
    ofstream ofs;
    ofs.open("reward.txt", std::ofstream::out | std::ofstream::trunc); // clear txt file
    ofs.close();
    system("clear");
    while (iteration <= 100000) {
        iteration += 1;
        int action = agent.action(current_state);
        int reward = game.act(action);
        vector<double> next_state = game.return_state();
        avg_reward += reward; 
        if (iteration % 100 == 0) {
            // save reward to .txt
            avg_reward /= 100;
            ofstream myfile;
            myfile.open ("reward.txt", ios::app);
            myfile << avg_reward << endl;
            myfile.close();
            // show progress
            gotoxy(0,0);
            cout << "Iteration: " << iteration << " | Max Score: " << max_score << " | Avg Reward: " << avg_reward << endl;
            avg_reward = 0;
        }

        // train
        if (game.score > max_score) {
            max_score = game.score; 
        }
        agent.store_mem(current_state, action, reward, next_state, game.is_done);
        agent.train();
        current_state = next_state; 
    }    
}
