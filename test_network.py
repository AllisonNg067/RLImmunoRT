import pandas as pd
from differential_equations_hypoxia_advanced import radioimmuno_response_model
import numpy as np
import matplotlib.pyplot as plt
from TMEClass import TME
import collections
import random
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import backend as K  # Correct import for Keras backend
import keras.backend as K
from sklearn.model_selection import KFold
import tensorflow as tf
tf.random.set_seed(42)  # extra code – ensures reproducibility on the CPU

reward_type = 'killed'
if reward_type == 'dose':
    input_shape = [3]  # == env.observation_space.shape
else:
    input_shape = [2]
n_outputs = 21  # == env.action_space.n

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])
params = pd.read_csv('new_hypoxia_parameters.csv').values.tolist()
initial_cell_count = 100000
param = params[0]
param[0] = initial_cell_count

action_type = 'RT'
double_Q = False
if double_Q:
    network_type = 'ddqn'
else:
    network_type = 'dqn'
model.load_weights(action_type + '_' + network_type + '_' + reward_type + '_more_exploration.weights.keras')
#model.load_weights('ddqn_dose.weights.keras')
def predict_one_episode(policy, env, n_max_steps=26, seed=42):
    '''
    This function predicts the entire episode using the given policy.
    Input arguments:
      policy - this can be any policy, i.e., a function that would
    '''
    final_rewards = []
    doses = []
    results = []
    obs = env.reset(1, seed=seed)
    tot_reward = 0
    schedule = []
    for step in range(n_max_steps):
        action, dose = policy(obs, testing=True)
        schedule.append(dose)
        obs, reward, done = env.step(action)
        tot_reward += reward
        if done:
                final_rewards.append(reward)
                #print(schedule)
                doses.append(schedule)
                # plt.plot(schedule)
                # plt.xlabel('RT Fraction Number')
                # plt.ylabel('Dose (Gy)')
                # plt.title('Optimal Dose per Fraction')
                #plt.savefig('optimal dose per fraction ' + action_type + reward_type + ' ' + str(counter) + '.png')
                # plt.show()
                result = [np.exp(reward), schedule]
                break
    return tot_reward, result, schedule


def epsilon_greedy_policy(state, epsilon=0, testing=False):
    doses = np.array(range(10, 31)) / 10
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # random action
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        print('state', state)
        print('Q', Q_values)
        if not testing:
          return Q_values.argmax()  # optimal action according to the DQN
        else:
          return Q_values.argmax(), doses[Q_values.argmax()]

rewards = []
Nepisodes = 100
seeds = range(100000, Nepisodes+100000)
results = []
schedules = []
for seed in seeds:
    env = TME(reward_type, 'DQN', action_type, params[0], range(10, 36), [10, 19], [10, 11], None, (-1,))
    env.reset(1, seed=seed)
    reward, result, doses = predict_one_episode(epsilon_greedy_policy, env, seed=seed)
    print('result', result)
    rewards += [reward]
    results.append(result)
    schedules.append(doses)

print(f'For the {Nepisodes} episodes:')
print(f'Minimum reward = {np.min(rewards)}')
print(f'Maximum reward = {np.max(rewards)}')
print(f'Average reward = {np.mean(rewards)}')
print(f'Standard deviation of rewards = {np.std(rewards)}')
print(f'Number of rewards above the average value =', np.sum(rewards > np.mean(rewards)))
print(results)
dataFrame = pd.DataFrame(results, columns=["TCP", "RT Dose Schedule"])
print(dataFrame)
dataFrame.to_csv('more exploration results ' + action_type + ' ' + reward_type + ' ' + network_type + '.csv', index=False)
max_length = max(len(lst) for lst in schedules)
padded_schedules = [lst + [np.nan] * (max_length - len(lst)) for lst in schedules]
padded_schedules = np.array(padded_schedules)
mean_dose = np.nanmean(padded_schedules, axis=0)
std_dose = np.nanstd(padded_schedules, axis=0)
# Plot with error bars
plt.figure()
plt.errorbar(range(len(mean_dose)), mean_dose, yerr=std_dose, fmt='o', ecolor='#0e4a41', capsize=5, color='#0e4a41')
plt.plot(range(len(mean_dose)), mean_dose, color='#0e4a41')  # Line connecting the points
plt.scatter(range(len(mean_dose)), mean_dose, color='#0e4a41')  # Mark points clearly
plt.xlabel('RT Fraction Number')
plt.ylabel('Dose (Gy)')
plt.title('Optimal ' + action_type + ' Dose per Fraction')
plt.savefig('more exploration optimal dose per fraction ' + action_type + ' ' + reward_type + ' ' + network_type + '.png')
#f = open('mean TCP ' + action_type + reward_type + '.txt', 'w')
#f.write("mean TCP " + str(np.mean(dataFrame['TCP'])))
print("mean TCP " + str(np.mean(dataFrame['TCP'])))
plt.figure()
plt.bar(np.arange(Nepisodes), np.array(rewards))
plt.plot([0, Nepisodes], np.mean(rewards)*np.ones(2,), 'r-', label='average reward')
plt.legend(loc='lower right')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
plt.savefig('more exploration barplot total rewards ' + action_type + ' ' + reward_type + ' ' + network_type + '.png')
