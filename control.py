import pandas as pd
from differential_equations_hypoxia_advanced import radioimmuno_response_model
import numpy as np
import matplotlib.pyplot as plt
from TMEClass import TME
import collections
import random

params = pd.read_csv('new_hypoxia_parameters.csv').values.tolist()
initial_cell_count = 100000
param = params[0]
param[0] = initial_cell_count
reward_type = 'killed'
action_type = 'RT'

def test_network(reward_type, action_type, env, sample_size=100):

    # Manually run the testing loop
    env.mode = 1  # Set mode to testing
    total_reward = 0
    doses = []
    final_rewards = []
    results = []
    #agent.run(n_epochs = 1, epoch_length = 26)
    for patient in range(100):
        env = TME(reward_type, 'DQN', action_type, params[0], range(10, 36), [10, 19], [10, 14], None, 1)
        env.reset(1, patient)
        schedule = []
        for _ in range(26):  # Run for 26 steps
            state = env.observe()
        #state_input = np.expand_dims(state, axis=0)  # Add batch dimension
            action_index = 10  #fix dose at 2Gy
            schedule.append(env.action_space[action_index])
            state, reward, done = env.act(action_index)
            total_reward += reward
            if done:
                final_rewards.append(reward)
                doses.append(schedule)
                plt.plot(schedule)
                plt.xlabel('RT Fraction Number')
                plt.ylabel('Dose (Gy)')
                plt.title('Optimal Dose per Fraction')
                #plt.savefig('optimal dose per fraction ' + action_type + reward_type + ' ' + str(counter) + '.png')
                plt.show()
                result = [np.exp(reward), schedule]
                results.append(result)
                #plt.close()
                break

    print(f'Total reward during testing: {total_reward}')
    print(doses)
    print(final_rewards)
    dataFrame = pd.DataFrame(results, columns=["TCP", "RT Dose Schedule"])
    print(dataFrame)
    file_name = 'control RT.csv'
    dataFrame.to_csv(file_name, index=False)
    #print(results)
    f = open('mean TCP control.txt', 'w')
    f.write("mean TCP " + str(np.mean(dataFrame['TCP'])))
    print("mean TCP " + str(np.mean(dataFrame['TCP'])))
    f.close()

reward_type='killed'
# Define your environment class (assuming TME is already defined)
env = TME(reward_type, 'DQN', action_type, params[40], range(10, 36), [10, 15], [10, 11], None, 1)

test_network(reward_type, action_type, env)
