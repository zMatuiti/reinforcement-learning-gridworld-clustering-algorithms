//
//  tutorial.cpp
//  RLTutorial
//
//  Created by Julio Godoy on 11/25/18.
//  Copyright © 2018 Julio Godoy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2, current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward, Qvalues[100][100][4], reward[100][100], finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos, y_pos, prev_x_pos, prev_y_pos, blockedx, blockedy, i, j, k;
ofstream reward_output;

//////////////
// Setting value for learning parameters
int action_sel = 2;         // 1 is greedy, 2 is e-greedy
int environment = 2;        // 1 is small grid, 2 is Cliff walking
int algorithm = 2;          // 1 is Q-learning, 2 is Sarsa
int stochastic_actions = 0; // 0 is deterministic actions, 1 for stochastic actions
int num_episodes = 500;     // total learning episodes
float learn_rate = 0.1;     // how much the agent weights each new sample
float disc_factor = 0.99;   // how much the agent weights future rewards
float exp_rate = 0.05;      // how much the agent explores
///////////////

void Initialize_environment()
{
    if (environment == 1)
    {

        height_grid = 3;
        width_grid = 4;
        goalx = 3;
        goaly = 2;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    if (environment == 2)
    {

        height_grid = 4;
        width_grid = 12;
        goalx = 11;
        goaly = 0;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    for (i = 0; i < width_grid; i++)
    {
        for (j = 0; j < height_grid; j++)
        {

            if (environment == 1)
            {
                reward[i][j] = -0.04; //-1 if environment 2
                blocked[i][j] = 0;
            }

            if (environment == 2)
            {
                reward[i][j] = -1;
                blocked[i][j] = 0;
            }

            for (k = 0; k < 4; k++)
            {
                Qvalues[i][j][k] = rand() % 10;
                cout << "Initial Q value of cell [" << i << ", " << j << "] action " << k << " = " << Qvalues[i][j][k] << "\n";
            }
        }
    }

    if (environment == 1)
    {
        reward[goalx][goaly] = 100;
        reward[goalx][(goaly - 1)] = -100;
        blocked[1][1] = 1;
    }

    if (environment == 2)
    {
        reward[goalx][goaly] = 1;

        for (int h = 1; h < goalx; h++)
        {
            reward[h][0] = -100;
        }
    }
}

int action_selection()
{
    // Primero, encontrar la mejor acción posible (explotar)
    float best_q_value = -999999.0;
    int best_action = 0;
    for (int act = 0; act < 4; act++)
    {
        if (Qvalues[x_pos][y_pos][act] > best_q_value)
        {
            best_q_value = Qvalues[x_pos][y_pos][act];
            best_action = act;
        }
    }

    // Segundo, decidir si explorar o explotar
    float epsilon = 0.05; // Como lo pide el enunciado
    if ((float)rand() / RAND_MAX < epsilon)
    {
        return rand() % 4; // Explorar: Devolver una acción al azar
    }
    else
    {
        return best_action; // Explotar: Devolver la mejor acción
    }
}

void move(int action)
{
    prev_x_pos = x_pos; // Backup of the current position, which will become past position after this method
    prev_y_pos = y_pos;

    // Stochastic transition model (not known by the agent)
    // Assuming a .8 prob that the action will perform as intended, 0.1 prob. of moving instead to the right, 0.1 prob of moving instead to the left

    if (stochastic_actions)
    {
        // Code here should change the value of variable action, based on the stochasticity of the action outcome
        int random_chance = rand() % 10; // aleatorio entre 0 y 9
        if (random_chance == 8)
        {                              // 10% de probabilidad
            action = (action + 1) % 4; // DERECHA
        }
        else if (random_chance == 9)
        {                              // 10% de probabilidad
            action = (action + 3) % 4; // IZQUIERDA
        }
    }

    // After determining the real outcome of the chosen action, move the agent

    if (action == 0) // Up
    {

        if ((y_pos < (height_grid - 1)) && (blocked[x_pos][y_pos + 1] == 0)) // If there is no wall or obstacle Up from the agent
        {
            y_pos = y_pos + 1; // move up
        }
    }

    if (action == 1) // Right
    {

        if ((x_pos < (width_grid - 1)) && (blocked[x_pos + 1][y_pos] == 0)) // If there is no wall or obstacle Right from the agent
        {
            x_pos = x_pos + 1; // Move right
        }
    }

    if (action == 2) // Down
    {

        if ((y_pos > 0) && (blocked[x_pos][y_pos - 1] == 0)) // If there is no wall or obstacle Down from the agent
        {
            y_pos = y_pos - 1; // Move Down
        }
    }

    if (action == 3) // Left
    {

        if ((x_pos > 0) && (blocked[x_pos - 1][y_pos] == 0)) // If there is no wall or obstacle Left from the agent
        {
            x_pos = x_pos - 1; // Move Left
        }
    }
}

void update_q_prev_state() // Updates the Q value of the previous state
{
    // Determine the max_a(Qvalue[x_pos][y_pos])
    float max_q_s_prime = -999999.0;
    for (int act = 0; act < 4; act++)
    {
        if (Qvalues[x_pos][y_pos][act] > max_q_s_prime)
        {
            max_q_s_prime = Qvalues[x_pos][y_pos][act];
        }
    }
    if (((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == 3) && (y_pos == 1)) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0)))
    {
        max_q_s_prime = 0;
    }

    Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * (reward[x_pos][y_pos] + disc_factor * max_q_s_prime - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
}

void update_q_prev_state_sarsa()
{
    float q_s_prime_a_prime = Qvalues[x_pos][y_pos][action_taken2];
    if (((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == 3) && (y_pos == 1)) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0)))
    {
        q_s_prime_a_prime = 0;
    }

    Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * (reward[x_pos][y_pos] + disc_factor * q_s_prime_a_prime - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
}

void Qlearning()
{
    action_taken = action_selection();
    move(action_taken);
    cum_reward += reward[x_pos][y_pos];
    update_q_prev_state();
}

void Sarsa()
{
    move(action_taken);
    cum_reward += reward[x_pos][y_pos];

    action_taken2 = action_selection();
    update_q_prev_state_sarsa();
    action_taken = action_taken2;
}

void Multi_print_grid()
{
    int x, y;

    for (y = (height_grid - 1); y >= 0; --y)
    {
        for (x = 0; x < width_grid; ++x)
        {

            if (blocked[x][y] == 1)
            {
                cout << " \033[42m# \033[0m";
            }
            else
            {
                if ((x_pos == x) && (y_pos == y))
                {
                    cout << " \033[44m1 \033[0m";
                }
                else
                {
                    cout << " \033[31m0 \033[0m";
                }
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    reward_output.open("Rewards.txt", ios_base::app);
    Initialize_environment(); // Initialize the features of the chosen environment (goal and initial position, obstacles, rewards)

    for (i = 0; i < num_episodes; i++)
    {
        cout << "\n \n Episode " << i;
        current_episode = i;
        x_pos = init_x_pos;
        y_pos = init_y_pos;
        cum_reward = 0;

        // If Sarsa was chosen as the algorithm:
        if (algorithm == 2)
        {
            action_taken = action_selection();
        }

        // While the agent has not reached a terminal state:
        while (!(((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == goalx) && (y_pos == (goaly - 1))) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0))))
        {
            if (algorithm == 1)
            {

                Qlearning();
            }
            if (algorithm == 2)
            {
                Sarsa();
            }
        }

        finalrw[i] = cum_reward;
        cout << " Total reward obtained: " << finalrw[i] << "\n";
        reward_output << i << "," << finalrw[i] << "\n";
    }
    reward_output.close();

    return 0;
}
