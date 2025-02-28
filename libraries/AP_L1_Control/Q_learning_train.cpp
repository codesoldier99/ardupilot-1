// Q_learning_train.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "DDPG_C.H"

using namespace std;

// Define state and action spaces
#define STATE_NUM 3      // Number of trajectory error states
#define ACTION_NUM 3     // Number of k3 parameter options
#define MAX_EPISODES 1000
#define MAX_STEPS 200
#define ALPHA 0.8        // Learning rate
#define GAMMA 0.95       // Discount factor
#define EPSILON 0.1      // Exploration rate for ε-greedy policy

// Flight log data structure
struct FlightLog {
    float crosstrack_error;
    float k3_value;
    float reward;
};

// State mapping function
int map_to_state(float crosstrack_error) {
    if (crosstrack_error < -0.1f) {
        return 0;
    } else if (crosstrack_error >= -0.1f && crosstrack_error <= 0.1f) {
        return 1;
    } else {
        return 2;
    }
}

// Action mapping function
float map_to_k3(int action) {
    switch(action) {
        case 0: return 3.8f;
        case 1: return 4.0f;
        case 2: return 6.0f;
        default: return 4.0f;
    }
}

// Reward calculation function
float calculate_reward(float crosstrack_error, float prev_error) {
    float reward = 0;
    
    // Positive reward if error decreases
    if (fabs(crosstrack_error) < fabs(prev_error)) {
        reward += 10;
    }
    
    // Additional reward if error is within acceptable range
    if (fabs(crosstrack_error) < 0.1f) {
        reward += 100;
    }
    
    // Penalty if error is too large
    if (fabs(crosstrack_error) > 0.5f) {
        reward -= 50;
    }
    
    return reward;
}

// Load flight log data
vector<FlightLog> load_flight_logs(const string& filename) {
    vector<FlightLog> logs;
    ifstream file(filename, ios::binary);
    
    if (!file.is_open()) {
        cout << "Failed to open flight log file" << endl;
        return logs;
    }
    
    FlightLog log;
    while (file.read(reinterpret_cast<char*>(&log), sizeof(FlightLog))) {
        logs.push_back(log);
    }
    
    file.close();
    return logs;
}

// Function to find the best action for a given state
int inference_best_action(int state, double Q[STATE_NUM][ACTION_NUM]) {
    int best_action = 0;
    double max_q_value = Q[state][0];
    
    for (int a = 1; a < ACTION_NUM; a++) {
        if (Q[state][a] > max_q_value) {
            max_q_value = Q[state][a];
            best_action = a;
        }
    }
    
    return best_action;
}




// ε-greedy policy for action selection
int select_action(double Q[STATE_NUM][ACTION_NUM], int state, bool training = true) {
    if (training && ((double)rand() / RAND_MAX) < EPSILON) {
        // Exploration: randomly select an action
        return rand() % ACTION_NUM;
    } else {
        // Exploitation: select the action with highest Q-value
        return inference_best_action(state, Q);
    }
}

// Train Q-table
void train_q_table(vector<FlightLog>& logs, double Q[STATE_NUM][ACTION_NUM]) {
    // Initialize Q-table
    for (int i = 0; i < STATE_NUM; i++) {
        for (int j = 0; j < ACTION_NUM; j++) {
            Q[i][j] = 0.0;
        }
    }
    
    // Training loop
    auto start_time = chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        float total_reward = 0;
        float prev_error = 0;
        
        // Randomly select a starting point from logs
        int start_idx = rand() % (logs.size() - MAX_STEPS);
        
        for (int step = 0; step < MAX_STEPS; step++) {
            // Get current state
            int current_state = map_to_state(logs[start_idx + step].crosstrack_error);
            
            // Select action
            int action = select_action(Q, current_state, true);
            float k3 = map_to_k3(action);
            
            // Get next state and reward
            float next_error = logs[start_idx + step + 1].crosstrack_error;
            int next_state = map_to_state(next_error);
            float reward = calculate_reward(next_error, prev_error);
            
            // Find maximum Q-value for next state
            double max_next_q = Q[next_state][0];
            for (int a = 1; a < ACTION_NUM; a++) {
                if (Q[next_state][a] > max_next_q) {
                    max_next_q = Q[next_state][a];
                }
            }
            
            // Q-learning update formula
            Q[current_state][action] = Q[current_state][action] + 
                                     ALPHA * (reward + GAMMA * max_next_q - 
                                     Q[current_state][action]);
            
            total_reward += reward;
            prev_error = next_error;
            
            // End episode early if target state is reached
            if (fabs(next_error) < 0.1f) {
                break;
            }
        }
        
        // Output training progress every 100 episodes
        if ((episode + 1) % 100 == 0) {
            cout << "Episode " << episode + 1 << ", Total Reward: " << total_reward << endl;
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Training completed in " << duration.count() / 1000.0 << " seconds" << endl;
}

// Save trained Q-table
void save_q_table(double Q[STATE_NUM][ACTION_NUM], const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < STATE_NUM; i++) {
            for (int j = 0; j < ACTION_NUM; j++) {
                file << Q[i][j] << " ";
            }
            file << endl;
        }
        file.close();
        cout << "Q-table saved to " << filename << endl;
    } else {
        cout << "Failed to save Q-table to " << filename << endl;
    }
}

// Validate Q-table performance
void validate_q_table(double Q[STATE_NUM][ACTION_NUM], vector<FlightLog>& test_logs) {
    float total_error = 0;
    int test_steps = 0;
    
    for (size_t i = 0; i < test_logs.size() - 1; i++) {
        int state = map_to_state(test_logs[i].crosstrack_error);
        int action = inference_best_action(state, Q);
        float k3 = map_to_k3(action);
        
        total_error += fabs(test_logs[i].crosstrack_error);
        test_steps++;
    }
    
    cout << "Average tracking error: " << total_error / test_steps << endl;
}

int main() {
    // Set random seed
    srand(static_cast<unsigned int>(time(NULL)));
    
    // Initialize Q-table with proper dimensions
    double Q[STATE_NUM][ACTION_NUM] = {0};
    
    // Load training data
    string train_file = "train_flight_logs.dat";
    vector<FlightLog> train_logs = load_flight_logs(train_file);
    if (train_logs.empty()) {
        cout << "No training data available" << endl;
        return -1;
    }
    
    cout << "Loaded " << train_logs.size() << " training samples" << endl;
    
    // Train Q-table
    cout << "Starting training..." << endl;
    train_q_table(train_logs, Q);
    
    // Save training results
    string output_file = "trained_q_table.txt";
    save_q_table(Q, output_file);
    
    // Load test data and validate
    string test_file = "test_flight_logs.dat";
    vector<FlightLog> test_logs = load_flight_logs(test_file);
    if (!test_logs.empty()) {
        cout << "Validating training results..." << endl;
        validate_q_table(Q, test_logs);
    } else {
        cout << "No test data available for validation" << endl;
    }
    
    // Print final Q-table
    cout << "\nFinal Q-table:" << endl;
    for (int i = 0; i < STATE_NUM; i++) {
        for (int j = 0; j < ACTION_NUM; j++) {
            cout << Q[i][j] << "\t";
        }
        cout << endl;
    }
    
    return 0;
}