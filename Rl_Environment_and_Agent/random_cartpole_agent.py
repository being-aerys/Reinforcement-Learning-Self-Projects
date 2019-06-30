import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_rewards = 0
    total_steps_taken = 0
    observation = env.reset()

    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_rewards += reward
        total_steps_taken += 1
        if done:
            break


    print("Episode ended in %d steps with reward equal to %.4f" % (total_steps_taken, total_rewards))
