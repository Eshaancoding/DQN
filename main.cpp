#include "Agent.cpp"
#include "Pong.cpp"
#include <iostream>
#include <stdio.h>
using namespace std;

void gotoxy(int x,int y)    
{
    printf("%c[%d;%df",0x1B,y,x);
}

// Parameters
int BATCHES = 32; // number of training data trained per frame 
double LR = 0.001; // 0.001 Learning rate for the neural networks
int MEM_CAP = 10000; // 10 000 replay mem capacity
int FRAME_REACH = 10000; // 10 000 frames till epsilon reaches 0.05. Epsilon declines linearly
int TARGET_UPDATE = 5000; // 5000 frames for target net to update with net
vector<int> layout ({8, 50, 3}); // Neural Network layout
int gameWidth = 7;
int gameHeight = 12; // if you want to alter width or height try to play around with the reward system
// main
int main () {
    Pong game = Pong(gameWidth, gameHeight); // PASS
    Agent agent = Agent(layout, LR, MEM_CAP, FRAME_REACH, TARGET_UPDATE, BATCHES); 
    vector<double> current_state = game.return_state(); 
    int max_score = 0;
    system("cls"); // change to clear if your OS is Macosx or Linux
    while (true) {
        int action = agent.action(current_state);
        int reward = game.act(action);
        vector<double> next_state = game.return_state();
        // print
        gotoxy(0,0);
        cout<<"Max score: "<<max_score<<"         "<<endl;
        cout<<"Score: "<<game.score<<"         "<<endl;
        cout<<"Reward: "<<reward<<"         "<<endl;
        cout<<"Episodes: "<<game.episodes<<"         "<<endl;
        cout<<"Frames: "<<agent.frames<<"         "<<endl;
        if (game.score > max_score) {
            max_score = game.score; 
        }
        game.print_state();
        agent.store_mem(current_state, action, reward, next_state, game.is_done);
        agent.train();
        current_state = next_state; 
    }    
}
