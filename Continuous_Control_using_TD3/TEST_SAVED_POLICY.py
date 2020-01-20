
import time, numpy as np, TD3, gym
import roboschool, gym
from TD3 import TD3
from PIL import Image
import matplotlib.pyplot as plt

def plot(rewards):

    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('td3_test_rewards.png')
    # plt.show()


def run_policy(env_name):
    test_rewards = []
    env_name = "MountainCarContinuous-v0"
    env_name = "ContinuousCartPoleEnv"
    random_seed = 0
    n_episodes = 10
    lr = 0.002
    max_timesteps = 2000
    render = True
    save_gif = True

    filename = "TD3_{}_{}".format(env_name, random_seed)
    filename += '_solved'

    env = gym.make(env_name)
    directory = "./preTrained/"+str(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)

    policy.load_actor(directory, filename)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        test_rewards.append(ep_reward)
        print('Evaluation Episode: {}\tEpisode Reward: {}'.format(ep, int(ep_reward)))
        #ep_reward = 0
        env.close()
    plot(test_rewards)

