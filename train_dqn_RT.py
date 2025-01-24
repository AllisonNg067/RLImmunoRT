import tensorflow as tf
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
from collections import deque

reward_type = 'dose'
action_type = 'RT'
params = pd.read_csv('new_hypoxia_parameters.csv').values.tolist()
env = TME(reward_type, 'DQN', action_type, params[0], range(10, 36), [10, 19], [10, 11], None, (-1,))
tf.random.set_seed(42)  # extra code – ensures reproducibility on the CPU

input_shape = [3]  # == env.observation_space.shape
n_outputs = 21  # == env.action_space.n

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0, testing=False):
    doses = np.array(range(10, 31)) / 10
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # random action
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        if not testing:
          return Q_values.argmax()  # optimal action according to the DQN
        else:
          return Q_values.argmax(), doses[Q_values.argmax()]

replay_buffer = deque(maxlen=2000)
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]  # [states, actions, rewards, next_states, dones]

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)

    next_state, reward, done = env.step(action)
    #print(next_state)
    replay_buffer.append((state, action, reward, next_state, done))
    #print('buffer length',  len(replay_buffer))
    return next_state, reward, done

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # print(states)
    # print(actions)
    # print(rewards)
    # print(next_states)
    # print(dones)
    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - dones # episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

replay_buffer = deque(maxlen=2000)
np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = 0
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
final_rewards = []
loss_fn = tf.keras.losses.MeanSquaredError()
for episode in range(7000):
    obs = env.reset(-1, episode)
    #print('obs', obs)
    total_reward = 0
    for step in range(26):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done = play_one_step(env, obs, epsilon)
        total_reward += reward
        if done:
            break
        #print(replay_buffer)
    # extra code – displays debug info, stores data for the next figure, and
    #              keeps track of the best model weights so far
    final_rewards.append(reward)
    rewards.append(total_reward) # Note from Du 11/01/2025: as each step that we stay
                         # in the game yields 1 reward point, appending the
                         # last step value is the same as appending the total
                         # reward for the current episode
    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step

    if episode > 99:
        training_step(batch_size)

model.set_weights(best_weights)  # extra code – restores the best model weights

# we expect len(replay_buffer) to be 2000 afeter 600 episodes
print('The replay_buffer currently has', len(replay_buffer), 'items.')
# extra code – this cell generates and saves Figure 18–10
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.grid(True)
plt.show()

plt.savefig('sum of rewards dqn dose.png')
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.grid(True)
plt.show()
plt.savefig('final rewards dqn dose.png')
model.save('dqn_dose.weights.keras')
