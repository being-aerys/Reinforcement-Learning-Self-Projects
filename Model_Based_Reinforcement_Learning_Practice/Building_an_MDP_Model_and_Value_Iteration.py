
"""Implementing my own version of model-based reinforcement learning"""

import gym, collections, numpy as np, time
from tensorboardX import SummaryWriter

env = "FrozenLake-v0"
discount_factor = 0.9
no_of_random_transitions_per_training_batch = 100
no_of_test_episodes = 20

class Agent:

    def __init__(self):
        self.environment = gym.make(env)
        self.current_state_agent_is_in = self.environment.reset()

        #data structures for the transition model and reward model
        self.transision_probability_cuboid = np.zeros(( self.environment.observation_space.n,self.environment.action_space.n, self.environment.observation_space.n))

        self.no_of_transitions_from_s_to_s_prime_with_action_a = np.zeros((self.environment.observation_space.n, self.environment.action_space.n, self.environment.observation_space.n))
        #shape is (16, 4, 16)
        self.transition_reward_matrix = np.zeros((self.environment.observation_space.n, self.environment.action_space.n))

        self.value_of_all_states = np.zeros((self.environment.observation_space.n))



    def generate_random_transitions(self, no_of_steps_to_generate):


        for _ in range(no_of_steps_to_generate):

            random_action = self.environment.action_space.sample()#remember this is how we sample an action in an environment in gym
            next_state, reward, is_episode_over, _ = self.environment.step(random_action)




            '''Learn transition probability.'''
            self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action][next_state] += 1



            total_no_of_times_this_action_was_taken_in_this_state = (self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action]).sum()


            self.transision_probability_cuboid[self.current_state_agent_is_in][random_action][next_state] = (self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action][next_state]) / float(total_no_of_times_this_action_was_taken_in_this_state)


            '''Learn reward model.'''
            self.transition_reward_matrix[self.current_state_agent_is_in][random_action] = (self.transition_reward_matrix[self.current_state_agent_is_in][random_action] + reward) / (self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action]).sum()

            if reward ==1:

                print("state is ", self.current_state_agent_is_in)
                print(self.transition_reward_matrix[self.current_state_agent_is_in][random_action])
                #time.sleep(1)



            if is_episode_over:



                self.current_state_agent_is_in = self.environment.reset()
                break
            else:
                self.current_state_agent_is_in = next_state



    '''VVI: for every state, perform a bellman backup using the transition probability and reward model learned after every batch of training data
    generated'''

    def value_iteration(self):

        for current_state in range(self.environment.observation_space.n):

            list_of_possible_values_of_this_state_for_actions_taken = np.zeros(self.environment.action_space.n)


            for action in range(self.environment.action_space.n):

                list_of_possible_values_of_this_state_for_actions_taken[action] = self.transition_reward_matrix[current_state][action] + discount_factor *(np.dot(self.transision_probability_cuboid[current_state] [action], self.value_of_all_states ))

                best_value_according_to_best_action = max(list_of_possible_values_of_this_state_for_actions_taken)



                self.value_of_all_states[current_state] = best_value_according_to_best_action




    def calculate_the_value_of_this_action_value_for_current_state(self, state, action):

        action_value = self.transition_reward_matrix[state][action] + discount_factor * ((np.dot(self.transision_probability_cuboid[state] [action], self.value_of_all_states )))

        return action_value


    def select_best_action_according_to_the_state_values(self, current_state):


        list_of_values_for_the_actions = np.zeros(self.environment.observation_space.n)

        for action in range(self.environment.action_space.n):

            action_value = self.calculate_the_value_of_this_action_value_for_current_state(current_state, action)

            list_of_values_for_the_actions[action]

        best_action = list_of_values_for_the_actions.argmax()

        return best_action


    def play_test_episodes(self, test_env_instance):

        total_reward = 0.0

        state = test_env_instance.reset()

        while True:

            action = self.select_best_action_according_to_the_state_values(state)


            new_state, reward, is_episode_over, _ = test_env_instance.step(action)

            total_reward += reward

            if is_episode_over:
                break

            state = new_state

        return total_reward



if __name__ == "__main__":

    environment = gym.make(env)

    agent_for_value_iteration = Agent()

    writer = SummaryWriter(comment= "Learning values of state using Value Iteration")

    '''Now, we generate random episodes, learn a model, perform value iteration and repeat until the agent reaches the goal terminal state in 90%
        of test episodes.
        
       Generally, if the task is of continuous control, we stop after the average reward obtained while testing surpasses the reward boundary 
       for the environment used. E.g., For cartpolev0, if the agent learns to keep the pole up for 195 time steps, it is said
       to have learned. 
       
       However, in frozen lake, the agent gets a 1 only when it reaches the terminal goal state, else 0 in all other cases.
       Hence, we will stop if the agent reaches the goal terminal state 90% of the test episodes.'''



    current_batch_of_training = 0

    max_avg_reward_during_training = 0

    while True:

        '''Generate episodes to learn the transition and reward models of the environment.'''
        agent_for_value_iteration.generate_random_transitions(no_of_random_transitions_per_training_batch)

        '''Learn values of the state using value_iteration'''
        agent_for_value_iteration.value_iteration()

        '''Calculated the average reward obtained for test episodes'''
        total_reward_in_test_episodes = 0.0

        for _ in range(no_of_test_episodes):

            total_reward_in_test_episodes = agent_for_value_iteration.play_test_episodes(environment) #make sure you pass a new environment here,
                                                                                                        #not the one the agent trained on

        average_reward_per_test_episode = total_reward_in_test_episodes/ float(no_of_test_episodes)

        print("Avg reward is ,",average_reward_per_test_episode)

        writer.add_scalar("reward", average_reward_per_test_episode, current_batch_of_training)

        if average_reward_per_test_episode > max_avg_reward_during_training:

            print("New best average test reward updated from %.3f to %.3f" % (best_reward, average_reward_per_test_episode))

            best_reward = average_reward_per_test_episode

        if average_reward_per_test_episode > 0.9:

            print("The agent has learned to solve the environment in 90% of the test episodes in %d iterations!" % current_batch_of_training)

            break

    writer.close()












