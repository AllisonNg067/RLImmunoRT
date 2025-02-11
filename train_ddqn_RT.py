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

target = tf.keras.models.clone_model(model)  # clone the model's architecture
target.set_weights(model.get_weights())  # copy the weights
env.reset(-1, seed=42)
np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = 0

batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-5)
loss_fn =  tf.keras.losses.MeanSquaredError()

replay_buffer = deque(maxlen=2000)  # resets the replay buffer
if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. Neural nets can be very slow without a GPU.")
else:
    print("GPU detected")

def play_one_step(env, state, epsilon, episode):
    action, Q = epsilon_greedy_policy(state, 'RT', epsilon, False, episode)

    next_state, reward, done = env.step(action)
    #print(next_state)
    replay_buffer.append((state, action, reward, next_state, done))
    #print('buffer length',  len(replay_buffer))
    return tf.convert_to_tensor(next_state, dtype=tf.float32), reward, done

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

def epsilon_greedy_policy(state, treatment_to_optimise='RT', epsilon=0, testing=False, episode=0):
    if treatment_to_optimise == 'RT':
      doses = np.array(range(10, 31)) / 10
    elif treatment_to_optimise == 'anti-PD-1':
      doses = np.array([0.01*0.76*x for x in range(1, 25)])
    else:
      doses = np.array([0.01*0.04*x for x in range(1, 25)])
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs), model.predict(state[np.newaxis], verbose=0)[0]  # random action
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        if not testing:
          return Q_values.argmax(), Q_values  # optimal action according to the DQN
        else:
          return Q_values.argmax(), doses[Q_values.argmax()]

@tf.function(reduce_retracing=True)
def training_step(batch_size):
    global target
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target(next_states, training=False)  # <= CHANGED
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    runs = 1.0 - dones  # episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
def exploration_probability(episode, explore=0):
    return min(1, max(1 - (episode - explore)/9000, 0.01))
rewards = []
final_rewards = []
episodes_before_train = 100
for episode in range(10000):
      #treatment_res_list = data_PD[(data_PD['RT Treatment Days'] == str(t_treat_rad_optimal)) & (data_PD['anti-PD-1 Treatment Days'] == str(t_treat_p1_optimal))].values.tolist()[0]
    total_reward = 0
    obs = env.reset(-1, seed=episode)
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    for step in range(26):
        epsilon = exploration_probability(episode, explore=episodes_before_train)
        obs, reward, done = play_one_step(env, obs, epsilon, episode)
        if episode >= 9998:
            print('Q', epsilon_greedy_policy(obs))
        total_reward += reward
        if done:
            break

    # extra code – displays debug info, stores data for the next figure, and
    #              keeps track of the best model weights so far
    #print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, eps: {epsilon:.3f}",
          #end="")
    rewards.append(total_reward)
    final_rewards.append(reward)
    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step

    if episode > 0:
        training_step(batch_size)
        if episode % 2 == 0:                        # <= CHANGED
            target.set_weights(model.get_weights())  # <= CHANGED

    # Alternatively, you can do soft updates at each step:
    #if episode > 50:
        #training_step(batch_size)
        #target_weights = target.get_weights()
        #online_weights = model.get_weights()
        #for index, online_weight in enumerate(online_weights):
        #    target_weights[index] = (0.99 * target_weights[index]
        #                             + 0.01 * online_weight)
        #target.set_weights(target_weights)
import pickle
with open('replay_buffer_' + action_type + '_ddqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(replay_buffer, f)

with open('train_reward_' + action_type + '_ddqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(rewards, f)

with open('train_final_reward_' + action_type + '_ddqn_' + reward_type + '.pkl', 'wb') as f:
    pickle.dump(final_rewards, f)

model.set_weights(best_weights)  # extra code – restores the best model weights

# extra code – this cell plots the learning curve
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.grid(True)
plt.show()
plt.savefig('sum of rewards ' + action_type + ' ddqn ' + reward_type + '.png')

plt.figure(figsize=(8, 4))
plt.plot(final_rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Final rewards", fontsize=14)
plt.grid(True)
plt.show()
plt.savefig('final rewards ' + action_type + ' ddqn ' + reward_type + '.png')
model.compile(optimizer='adam', loss='mse')
model.save(action_type + '_ddqn_' + reward_type + '.weights.keras')
print('model trained and saved successfully')
