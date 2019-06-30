import gym, random
class ActionWrapper_Example(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0.1):
        super(ActionWrapper_Example, self).__init__(env)
        self.epsilon = epsilon
        self.random_action_in_an_episode_counter = 0

    def action(self, action):
        if random.random() < self.epsilon:
            self.random_action_in_an_episode_counter += 1
            print("Random action to be selected.")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = ActionWrapper_Example(gym.make("CartPole-v0"))
    env = gym.wrappers.Monitor(env, "recording", force = True)#sudo apt-get install ffmpeg
    state = env.reset()
    total_reward = 0

    while True:
        next_state, reward, done, info = env.step(0) #taking only left. However, the action might change because we have overridden the action method
        total_reward += reward
        if done:
            break

    print("Number of times the random action was selected in the episode: %d"%env.random_action_in_an_episode_counter)
    print("After implementing epsilon-greedy algorithm by overriding the action() method, the total reward is %d."%total_reward)
    print("Note that with the increase in the value of epsilon, the average reward an agent gets tends to increase.")
