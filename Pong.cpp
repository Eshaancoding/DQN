#include <iostream> 
#include <stdlib.h>
#include <vector>
#include <string>
#include <cmath> 
#include <algorithm>
using namespace std;
class Pong {
private:
    int pongX;
    int pongY;
    int pongVelX;
    int pongVelY;
    int width;
    int height;
    int playerX = 0;
    int paddleWidth;
public:
    int score = 0;
    int episodes = 0;
    bool is_done = false;
    void reset () {
        this->pongX = 0;
        this->pongY = 0;  
	bool isPongVelX = rand() % 2; 
	bool isPongVelY = rand() % 2;
        if (isPongVelX) {	
		this->pongVelX = 1;
	} else {
		this->pongVelX = -1;
	}
	if (isPongVelY) {
		this->pongVelY = 1;
	} else {
		this->pongVelY = -1;
	}
        episodes++; 
        score = 0;
    }
    Pong (int width, int height, int paddleWidth = 1) {
        reset();
        this->width = width;
        this->height = height;
        this->paddleWidth = paddleWidth; 
    }
    // 0 : Left
    // 1 : Stay
    // 2 : Right
    int act (int action) {
        int reward = 0; 
        is_done = false;
        if (action > 2) {
            throw invalid_argument("invalid action");
        }
        if (action == 0 && this->playerX - this->paddleWidth != -width + 1) {
            this->playerX -= 1;    
        }
        if (action == 2 && this->playerX + this->paddleWidth != width - 1) {
            this->playerX += 1;
        }
	if (action == 0 && this->playerX - this->paddleWidth == -width + 1) {
	    reward = -1;
	}
	if (action == 2 && this->playerX + this->paddleWidth == width - 1) {
	    reward = -1; 
	}
        // update pong position
        this->pongX += this->pongVelX; 
        this->pongY += this->pongVelY;
        // boundry
        if (pongX == -width + 1 && pongVelX < 0) {
            pongVelX *= -1;
        }
        if (pongX == width - 1 && pongVelX > 0) {
            pongVelX *= -1; 
        }
        // hits player paddle
        if (abs(pongX - playerX) <= paddleWidth && pongY == -height + 2 && pongVelY < 0) {
            pongVelY *= -1; 
            reward = 3;
            score++;
        }
        if (pongY == -height + 1) {
            reward = -3;
            is_done = true; 
            this->reset(); 
        }
        // hit enemy paddle
        if (pongY == height - 2 && pongVelY > 0) {
            pongVelY *= -1; 
        }
        return reward; 
    }
    int coordX (int x) {
        int widthOffset = this->width;
        return x + widthOffset; 
    }
    int coordY (int y) {
        int heightOffset = this->height;
        return (y * -1) + heightOffset;
    }
    void print_state () {
        string a[(height * 2)+1][(width * 2)+1];
        a[coordY(pongY)][coordX(pongX)] = "O";
        a[coordY(height - 1)][coordX(pongX)] = "-";
        a[coordY(-height + 1)][coordX(playerX)] = "-"; 
        for (int i = 0; i <= this->paddleWidth; i++) {
            a[coordY(-height + 1)][coordX(playerX + i)] = "-";
            a[coordY(-height + 1)][coordX(playerX - i)] = "-";
        }
        for (int i = 0; i < (width * 2) + 3; i++) {
            cout<<"#";
        }
        cout<<endl; 
        for (int i = 0; i < (height * 2) + 1; i++) {
            cout<<"#";
            for (int x = 0; x < (width * 2) + 1; x++) {
                if (a[i][x] == "") {
                    cout<<" ";
                } else {
                    cout<<a[i][x];
                }
            }
            cout<<"#"<<endl; 
        }
        for (int i = 0; i < (width * 2) + 3; i++) {
            cout<<"#";
        }
        cout<<endl; 
    }
    vector<double> return_big_state () {
        double a[(height * 2)+1][(width * 2)+1];
        a[coordY(pongY)][coordX(pongX)] = 255; 
        a[coordY(height - 0)][coordX(pongX)] = 100; 
        a[coordY(-height + 1)][coordX(playerX)] = 100; 
        vector<double> state;
        for (int i = 0; i < (height * 2) + 1; i++) {
            for (int x = 0; x < (width * 2) + 1; x++) {
                state.push_back(a[i][x]);
            }
        }
        return state; 
    }
    vector<double> return_state () {
        vector<double> state; 
        state.push_back(pongX); 
        state.push_back(pongY);
        state.push_back(pongVelX);
        state.push_back(pongVelY);
        state.push_back(width);
        state.push_back(height);
        state.push_back(playerX); 
        state.push_back(paddleWidth);
        return state;
    }
};
