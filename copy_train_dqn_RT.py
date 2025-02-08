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
import pickle
reward_type = 'killed'
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
model.load_weights(action_type + '_dqn_' + reward_type + '.weights.keras')
def epsilon_greedy_policy(state, episode=0, epsilon=0, testing=False):
    doses = np.array(range(10, 31)) / 10
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs), model.predict(state[np.newaxis], verbose=0)[0]  # random action
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        if not testing:
          return Q_values.argmax(), Q_values  # optimal action according to the DQN
        else:
          return Q_values.argmax(), doses[Q_values.argmax()]

replay_buffer = deque(maxlen=2000)
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]
    return (
        tf.convert_to_tensor(states, dtype=tf.float32),
        tf.convert_to_tensor(actions, dtype=tf.int32),
        tf.convert_to_tensor(rewards, dtype=tf.float32),
        tf.convert_to_tensor(next_states, dtype=tf.float32),
        tf.convert_to_tensor(dones, dtype=tf.float32)
    )

def play_one_step(env, state, epsilon, episode):
    action, _ = epsilon_greedy_policy(state, episode, epsilon, False)

    next_state, reward, done = env.step(action)
    #print(next_state)
    replay_buffer.append((state, action, reward, next_state, done))
    #print('buffer length',  len(replay_buffer))
    return tf.convert_to_tensor(next_state, dtype=tf.float32), reward, done

@tf.function(reduce_retracing=True)
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # print(states)
    # print(actions)
    # print(rewards)
    # print(next_states)
    # print(dones)
    next_Q_values = model(next_states, training=False)
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    runs = 1.0 - dones # episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

replay_buffer = deque(maxlen=2000)
with open('replay_buffer_' + action_type + '_dqn_' + reward_type + '.pkl', 'rb') as f:
    replay_buffer = pickle.load(f)
    #print(replay_buffer)
np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = 0
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-5)
with open('train_reward_' + action_type + '_dqn_' + reward_type + '.pkl', 'rb') as f:
    rewards = pickle.load(f)

with open('train_final_reward_' + action_type + '_dqn_' + reward_type + '.pkl', 'rb') as f:
    final_rewards = pickle.load(f)
loss_fn = tf.keras.losses.MeanSquaredError()
for episode in range(8000, 18000):
    obs = env.reset(-1, episode)
    #print('obs', obs)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    total_reward = 0
    for step in range(26):
        epsilon = max(1 - episode / 9000, 0.01)
        obs, reward, done = play_one_step(env, obs, epsilon, episode)
        if episode >= 17990:
            print('Q', epsilon_greedy_policy(obs))
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

    
    training_step(batch_size)

import pickle
with open('replay_buffer_' + action_type + '_dqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(replay_buffer, f)

with open('train_reward_' + action_type + '_dqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(rewards, f)

with open('train_final_reward_' + action_type + '_dqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(final_rewards, f)
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

plt.savefig('sum of rewards ' + action_type + ' dqn ' + reward_type + '.png')
plt.figure(figsize=(8, 4))
plt.plot(final_rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Final Reward", fontsize=14)
plt.grid(True)
plt.show()
plt.savefig('sum of rewards ' + action_type + ' dqn ' + reward_type + '.png')
model.compile(optimizer='adam', loss='mse')
model.save(action_type + '_dqn_' + reward_type + '.weights.keras')
print('model trained and saved successfully')
