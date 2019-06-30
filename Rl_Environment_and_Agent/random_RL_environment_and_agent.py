import random#--------------------------------------I dont think I will be using this pattern as this is not the standard way, at least for me.

class Environment:

    def __init__(self):
        self.total_steps_remaining = 10

    def get_observation(self):
        return [0.0, 0.0, 0.0, 0.0]

    def get_actions_for_current_observation(self):
        return [0, 1]

    def is_done(self):

        return (self.total_steps_remaining == 0)

    def action(self, action_among_many_possible_actions):
        if self.is_done():
            raise Exception("Episode is over.")
        else:
            self.total_steps_remaining -= 1
            return random.random()

class Agent:
    def __init__(self):
        self.total_reward_gained_in_the_episode = 0

    def step(self, env):
        current_env_observation = env.get_observation()
        possible_actions = env.get_actions_for_current_observation()#---------------------not using the current state to get the possible actions now
        reward_for_action_taken = env.action(random.choice(possible_actions)) #-----------implement a policy to take an action at current state
        self.total_reward_gained_in_the_episode += reward_for_action_taken

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward gained in an episode for this random agent is: %.4f"% agent.total_reward_gained_in_the_episode)
