#Pracaticing from the book Deep Reinforcement Learning Hands-On

import gym, collections, numpy as np, time
from tensorboardX import SummaryWriter

env = "FrozenLake-v0"
discount_factor = 0.9
episodes_to_generate_before_every_batch_of_training = 50
no_of_test_episodes = 20

class Agent:

    def __init__(self):
        self.environment = gym.make(env)
        self.current_state_agent_is_in = self.environment.reset()
        self.reward_storage_data_structure = collections.defaultdict(float) #going to contain <key, reward> in the dictionary
        self.transitions = collections.defaultdict(collections.Counter)  #going to contain < key = (state, action, next) , value = Count >
        self.values = collections.defaultdict(float)

        #data structure for the transition model and reward model
        self.transision_probability_cuboid = np.zeros((self.environment.action_space.n, self.environment.observation_space.n, self.environment.observation_space.n))
        self.transition_reward_matrix = np.zeros((self.environment.observation_space.n, self.environment.action_space.n))


    def generate_episodes_to_learn_transition_prob_and_reward_model(self, no_of_episodes_to_play):

        for _ in range(no_of_episodes_to_play):
            #we will be taking random actions to explore the environment since we just want to learn the transition and reward models of the env
            random_action = self.environment.action_space.sample()#remember this is how we sample an action in an environment in gym
            next_state, reward, is_episode_over, _ = self.environment.step(random_action)

            #store this (key = (state, action, next_state) :  value = reward for this transition) pair in self.state_transision_reward_data_structure
            self.reward_storage_data_structure[(self.current_state_agent_is_in, random_action, next_state)] = reward

            #store the transition as well
            self.transitions[(self.current_state_agent_is_in, random_action)] [next_state] += 1



            if is_episode_over:
                self.current_state_agent_is_in = self.environment.reset()
                break
            else:
                self.current_state_agent_is_in = next_state

    def learn_reward_and_transition_model(self):



        for (state, action) in self.transitions:





        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def calculate_values_of_a_state(self):

    def learn_transition_and_reward_models(self):

    '''VVI: for every state, perform a bellman backup using the transition probability and reward model learned'''
    def value_iteration(self):



        for state in range(self.environment.observation_space.n):
            for action in range(self.environment.action_space.n):




    def test_the_environment(self):



if __name__ == "__main__":

    environment = gym.make(env)

    print(environment.action_space.n)
    time.sleep(1111)
    agent_for_Value_Iteration = Agent()

    writer = SummaryWriter(comment= "Learning values of state using Value Iteration")

    '''Now, we generate random episodes, learn a model, perform value iteration and repeat until the agent reaches the goal terminal state in 90%
        of test episodes.
        
       Generally, if the task is of continuous control, we stop after the average reward obtained while testing surpasses the reward boundary 
       for the environment used. E.g., For cartpolev0, if the agent learns to keep the pole up for 195 time steps, it is said
       to have learned. 
       
       However, in frozen lake, the agent gets a 1 only when it reaches the terminal goal state, else 0 in all other cases.
       Hence, we will stop if the agent reaches the goal terminal state 90% of the test episodes.'''



    current_batch_of_training = 0

    while True:

        '''Generate episodes to learn the model of the environment.'''
        agent_for_Value_Iteration.generate_episodes_to_learn_transition_prob_and_reward_model(episodes_to_generate_before_every_batch_of_training)

        '''Learn values of the state using value_iteration'''









