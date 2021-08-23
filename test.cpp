#include "Agent.cpp"
#include "Pong.cpp"
#include <iostream>
#include <stdio.h>
using namespace std;

void gotoxy(int x,int y)    
{
    printf("%c[%d;%df",0x1B,y,x);
}

int gameWidth = 7;
int gameHeight = 12; // if you want to alter width or height try to play around with the reward system
vector<int> layout ({8, 50, 3}); // Neural Network layout
// main
int main () {
    Pong game = Pong(gameWidth, gameHeight); // PASS
    Agent agent = Agent(layout, "model.txt"); 
    vector<double> current_state = game.return_state(); 
    int max_score = 0;
    system("clear");
    while (true) {
        int action = agent.action(current_state);
        cout<<"Action: "<<action<<"         "<<endl;
        int reward = game.act(action);
        vector<double> next_state = game.return_state();
        // print
        gotoxy(0,0);
        game.print_state();
        current_state = next_state; 
    }    
}
