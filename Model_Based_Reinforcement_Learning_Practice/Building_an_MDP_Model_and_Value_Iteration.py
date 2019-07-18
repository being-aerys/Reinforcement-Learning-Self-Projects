
"""Implementing my own version of model-based reinforcement learning"""


'''Note that I am assigning a reward to each action, not <action, next state> pair
    The difference I noticed was than it takes more iterations if we assign the reward to action instead
    of <action, next state> pair.'''

import gym, collections, numpy as np, time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import math
env = "FrozenLake8x8-v0"
#env = "FrozenLake-v0"
discount_factor = 0.9
no_of_random_transitions_per_training_batch = 100
no_of_test_episodes = 20
start_time = time.time()

class Agent:

    def __init__(self):
        self.environment = gym.make(env)
        #self.environment.render()

        self.current_state_agent_is_in = self.environment.reset()

        #data structures for the transition model and reward model
        self.transision_probability_cuboid = np.zeros(( self.environment.observation_space.n,self.environment.action_space.n, self.environment.observation_space.n))

        self.no_of_transitions_from_s_to_s_prime_with_action_a = np.zeros((self.environment.observation_space.n, self.environment.action_space.n, self.environment.observation_space.n))
        #shape is (16, 4, 16)

        self.no_of_times_action_a_was_taken_at_state_s = np.zeros((self.environment.observation_space.n,self.environment.action_space.n))

        self.total_reward_obtained_by_taking_action_a_in_state_s = np.zeros((self.environment.observation_space.n,self.environment.action_space.n))

        self.transition_reward_matrix = np.zeros((self.environment.observation_space.n, self.environment.action_space.n))

        self.value_of_all_states = np.zeros((self.environment.observation_space.n))


    def generate_random_transitions(self, no_of_steps_to_generate):
        

        for _ in range(no_of_steps_to_generate):

            random_action = self.environment.action_space.sample()#remember this is how we sample an action in an environment in gym

            next_state, reward, is_episode_over, _ = self.environment.step(random_action)


            #print(self.transision_probability_cuboid[self.current_state_agent_is_in][random_action])

            '''Learn transition probability.'''
            self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action][next_state] += 1

            self.transision_probability_cuboid[self.current_state_agent_is_in][random_action] = (self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action]) / float(self.no_of_transitions_from_s_to_s_prime_with_action_a[self.current_state_agent_is_in][random_action].sum())





            #print(self.transision_probability_cuboid[self.current_state_agent_is_in][random_action])

            #print("-----------------------------------------")

            '''Learn reward model.'''
            self.no_of_times_action_a_was_taken_at_state_s[self.current_state_agent_is_in][random_action] += 1


            self.total_reward_obtained_by_taking_action_a_in_state_s[self.current_state_agent_is_in][random_action] += reward


            self.transition_reward_matrix[self.current_state_agent_is_in][random_action] = self.total_reward_obtained_by_taking_action_a_in_state_s[self.current_state_agent_is_in][random_action]/float(self.no_of_times_action_a_was_taken_at_state_s[self.current_state_agent_is_in][random_action])



            if is_episode_over:

                self.current_state_agent_is_in = self.environment.reset()
                break
            else:
                self.current_state_agent_is_in = next_state



    '''VVI: for every state, perform a bellman backup using the transition probability and reward model learned after every batch of training data
    generated'''

    def value_iteration(self):

        # print("--------------------Value iteration---------------------------------------------")
        #
        #
        # print("State values before value iteration: ",self.value_of_all_states)

        for state in range(self.environment.observation_space.n):

            # print("State\n", state)
            # print("transition probability: \n",self.transision_probability_cuboid[state])

            state_values_possible_due_to_diff_actions = np.zeros(self.environment.action_space.n)


            for action in range(self.environment.action_space.n):

                #print("Action",action)

                #print("transition prob. self.transition_reward_matrix[state][action]: \n",self.transition_reward_matrix[state][action])
                state_values_possible_due_to_diff_actions[action] = self.transition_reward_matrix[state][action] + discount_factor *(np.dot(self.transision_probability_cuboid[state] [action], self.value_of_all_states ))

                #print("possible value due to this action: ",state_values_possible_due_to_diff_actions[action])


            best_value_according_to_best_action = max(state_values_possible_due_to_diff_actions)



            self.value_of_all_states[state] = best_value_according_to_best_action
            #print("Updated state value:\n ",np.reshape(self.value_of_all_states,(8,8)))
            #print("The value of the neighbour should increase in the next value iteraion.")
            #print("-----------------------------------------------------------------")

        #print(self.value_of_all_states)

    # def calculate_the_value_of_this_action_value_for_current_state(self, state, action):
    #     total_action_value = self.transition_reward_matrix[state][action]
    #
    #     for next_state in range(self.environment.observation_space.n):
    #         total_action_value += self.transision_probability_cuboid[state][action][next_state] * self.value_of_all_states[next_state]
    #
    #     return total_action_value

    def calculate_the_value_of_this_action_value_for_current_state(self, state, action):




        action_value = self.transition_reward_matrix[state][action] + discount_factor * ((np.dot(self.transision_probability_cuboid[state] [action], self.value_of_all_states )))


        # total_action_value = self.transition_reward_matrix[state][action]
        # for next_state in range(self.environment.observation_space.n):
        #     total_action_value += discount_factor * (self.transision_probability_cuboid[state][action][next_state] * \
        #                           self.value_of_all_states[next_state])
        # print(total_action_value)
        #
        # print("--------------------------------------------\n")

        return action_value



    def select_best_action_according_to_the_state_values(self, current_state):
        #print("In running test episodes\nInside select best action\nstate values: ",self.value_of_all_states)
        #time.sleep(2)
        #print("current state: ",current_state)
        best_action_value = 0
        best_action = 0

        for action in range(self.environment.action_space.n):

            action_value = self.calculate_the_value_of_this_action_value_for_current_state(current_state, action)

            if best_action_value < action_value:
                #print("New best action value: ",action_value)
                best_action_value = action_value
                best_action = action


        return best_action


    def play_test_episodes(self, test_env_instance):

        total_reward = 0.0

        state = test_env_instance.reset()

        while True:

            action = self.select_best_action_according_to_the_state_values(state)
            #print("action is: ",action)


            new_state, reward, is_episode_over, _ = test_env_instance.step(action)
            #print("next state is ", new_state)


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

        current_batch_of_training += 1

        '''Generate episodes to learn the transition and reward models of the environment.'''
        #print("Generating sample transitions")
        agent_for_value_iteration.generate_random_transitions(no_of_random_transitions_per_training_batch)

        '''Learn values of the state using value_iteration'''
        #print("Doing Value iteration.")
        agent_for_value_iteration.value_iteration()



        '''Calculated the average reward obtained for test episodes'''
        total_reward_in_test_episodes = 0.0

        for _ in range(no_of_test_episodes):

            total_reward_in_test_episodes += agent_for_value_iteration.play_test_episodes(environment) #make sure you pass a new environment here,
                                                                                                        #not the one the agent trained on
        average_reward_per_test_episode = total_reward_in_test_episodes/ float(no_of_test_episodes)


        #print("Avg reward is ,",average_reward_per_test_episode)


        writer.add_scalar("reward", average_reward_per_test_episode, current_batch_of_training)

        if average_reward_per_test_episode > max_avg_reward_during_training:

            print("New best average test reward updated from %.3f to %.3f" % (max_avg_reward_during_training, average_reward_per_test_episode))


            max_avg_reward_during_training = average_reward_per_test_episode

        if average_reward_per_test_episode > 0.99:

            #print(agent_for_value_iteration.transition_reward_matrix)

            #print(agent_for_value_iteration.value_of_all_states)

            if env == "FrozenLake8x8-v0":

                policy_visualization = agent_for_value_iteration.value_of_all_states.reshape(8,8)

            else:

                policy_visualization = agent_for_value_iteration.value_of_all_states.reshape(4,4)

            print(policy_visualization)

            print("The agent has learned to solve the environment in 80 percent of the test episodes in %d iterations!" % current_batch_of_training)
            #x = int(math.sqrt(agent_for_value_iteration.environment.observation_space.n))
            heatmap_state_values = sns.heatmap(policy_visualization,cmap="BuGn_r")
            #Apparently you do not need to link the heatmap of seaborn with matplotlib explicitly
            plt.show()

            #should look something like
            # SFFF
            # FHFH
            # FFFH
            # HFFG

            break


    writer.close()
    print("Total time taken: ",time.time()-start_time)









"""     "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG  
        
        
        
        LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
        
        """


