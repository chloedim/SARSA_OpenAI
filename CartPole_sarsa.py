import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
#from CP_Env_v2 import CartPoleEnv

#env = CartPoleEnv()
env = gym.make("CartPole-v1")

# Between 1 and 0, how much will new info override old info
# 0 means nothing is learned, 1 means only the recent is considered and old knowledge is discarded
LEARNING_RATE = 0.1
# Between 1 and 0, measure of how much we care about future rewards over immedate rewards
DISCOUNT = 0.95
EPISODES = 50_000        # number of iterations
#DISPLAY_EVERY = 2500    # how often the solution is rendered
DISPLAY_EVERY = EPISODES + 1
COUNT_LIMIT = 200

# Exporation settings
epislon = 0.5           # intial rate, will decay over time
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2    # // means divide to int
epsilon_decay_value = epislon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Environment settings
high_state_value = env.observation_space.high   # high state values for this env, high: [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
low_state_value = env.observation_space.low     # low state values for this env, low: [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
num_actions = env.action_space.n                # number of actions avaliable, num_actions: 2
# due to overflow issues, reducing the decimal places by hardcoding the values 
high_state_value = np.array([4.8, 3.4, 4.1897, 3.4])
low_state_value = np.array([-4.8, -3.4, -4.1897, -3.4])

bucket_num = 20
DISCRETE_OBS_SPACE_SIZE = [bucket_num] * len(env.observation_space.high)   # buckets for discrete space
print(DISCRETE_OBS_SPACE_SIZE)
# find step size for env state 
discrete_obs_step_size = np.array((high_state_value - low_state_value) / DISCRETE_OBS_SPACE_SIZE)

# low = bad reward, high = good reward, shape = (20, 20, 20, 20, 2)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SPACE_SIZE + [num_actions]))
print("q_table shape:", q_table.shape)

# Collect stats
UPDATE_EVERY = 1000
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - low_state_value) / discrete_obs_step_size
    return tuple(discrete_state.astype(int))

def render_env():
    if (episode % DISPLAY_EVERY == 0):
        print("Episode: ", episode)
        return True
    else:
        return False

def sarsa(env):
    global epislon
    count = 0
    #initializing reward
    episode_reward = 0
    done=False

    render = render_env()

    discrete_state = get_discrete_state(env.reset()) #curr_discrete_state = discretised_state(env.reset())

    #choose the next action with epsilon greedy
    if (np.random.random() > epislon):
            action = np.argmax(q_table[discrete_state]) # get max value index bases on q_table
        # Get random action
    else:
        action = np.random.randint(0, num_actions)

    while not done:
        # get the next state
        new_state, reward, done, _ = env.step(action)
        #episode_reward += reward
        new_discrete_state = get_discrete_state(new_state) #new_discrete_state = discretised_state(new_state)

        # reduce the index by 1, if over bucket limit
        new_list = list(new_discrete_state)
        for i in range(len(new_list)):
            if (new_list[i] > bucket_num - 1):
                new_list[i] = bucket_num - 1
                new_discrete_state = tuple(new_list)

        #choose the next action
        if (np.random.random() > epislon):
            new_action = np.argmax(q_table[new_discrete_state])
        else:
            new_action = np.random.randint(0, num_actions)

        if (render):
            env.render()
            pygame.event.get() # Stop pygame from crashing

        count = count + 1
        if (count >= COUNT_LIMIT):  
            done = True

        if not done:
            #learning the Q-value
            # #Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2] - Q[state1, action1])
            current_q = q_table[discrete_state+(action,)]
            max_future_q = q_table[new_discrete_state+(new_action,)]
            new_q = current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q-current_q)
            q_table[discrete_state+(action,)]=new_q
        else:
            q_table[discrete_state + (action,)] = -375 # neagtive reward for falling or out of bounds
            
        discrete_state = new_discrete_state
        action = new_action

        #updating respective value
        episode_reward += reward

    #ep_rewards.append(episode_reward)

    if (END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING):
        epislon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if (episode % UPDATE_EVERY == 0):
        lastest_episodes = ep_rewards[-UPDATE_EVERY:]
        average_reward = sum(lastest_episodes) / len(lastest_episodes)
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(lastest_episodes))
        aggr_ep_rewards['max'].append(max(lastest_episodes))
        print(f"Episode: {episode} avg: {average_reward}, min:{min(lastest_episodes)}, max: {max(lastest_episodes)}")
    
for episode in range(EPISODES):
    #q_learning(env)
    sarsa(env)

# close env after loop
env.close()

# plot stats
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.legend()
plt.show()