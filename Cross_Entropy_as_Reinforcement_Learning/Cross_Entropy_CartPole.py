import time, gym
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn
from collections import namedtuple


hidden_size = 64
batch_size = 32
percentile = 80


''' Neural Network that takes in the observations and output an action'''
class generic_Neural_Network(torch.nn.Module):
    def __init__(self, observation_size, hidden_dimension_size, action_dimension_size):

        super(generic_Neural_Network, self).__init__()
        self.policy_network = torch.nn.Sequential(torch.nn.Linear(observation_size, hidden_dimension_size),
                                                  torch.nn.LeakyReLU(),
                                                  torch.nn.Linear(hidden_dimension_size, action_dimension_size),)


    def forward(self, input):
        return self.policy_network(input)


Episode_Data_Structure = namedtuple('Episode', field_names=['reward', 'episode_steps'])
Episode_Step_Data_Structure = namedtuple('Episode_Step', field_names=['observation', 'action'])

'''an iterator: a function that generates batches of episodes'''
def iterate_batches_of_episodes(env, policy_network_to_generate_episodes, batch_size):

    print("Starting to generate batches of episodes.\n")
    this_batch_of_episodes = []
    reward_of_running_episode = 0
    list_of_steps_in_running_episode = []
    current_observation = env.reset()
    #episode_counter = 0
    softmax_for_action_probability_from_logits = torch.nn.Softmax(dim =1) #A dimension along which Softmax will be computed
                                                                          # (so every slice along dim will sum to 1).
                                                                         #Applies the Softmax function to an n-dimensional input Tensor


    #generate episodes forever, training will stop whenever we satisfy the cumulative reward condition in the main training loop
    while True:

        observation_to_pass_to_NN = torch.FloatTensor([current_observation]) #need to convert to FloatTensor to operate in torch
                                                                             #very imp to keep initial_observation into [] here



        observation_to_pass_to_NN = observation_to_pass_to_NN.cuda()

        action_probabilities_distribution = softmax_for_action_probability_from_logits(policy_network_to_generate_episodes(observation_to_pass_to_NN))
        #tensor([[0.5939, 0.4061]], device='cuda:0', grad_fn=<SoftmaxBackward>)

        action_probabilities_distribution_1D = action_probabilities_distribution.cpu().data.numpy()[0]
        #[0.5939372 0.4060628] To go to the left or to the right

        #np.random.choice(5, p=[0.1, 0, 0.3, 0.6, 0]) returns one value from the range 0 to 5 with the prob distribution given as p

        action_to_take = np.random.choice(len(action_probabilities_distribution_1D), p= action_probabilities_distribution_1D)

        next_observation, reward, is_done, _ = env.step(action_to_take)

        reward_of_running_episode += reward



        list_of_steps_in_running_episode.append(Episode_Step_Data_Structure(observation = current_observation, action = action_to_take))

        if is_done:
            #print("Episode "+ str(episode_counter)+" ended.")
            #episode_counter +=1

            this_batch_of_episodes.append(Episode_Data_Structure(reward=reward_of_running_episode, episode_steps=list_of_steps_in_running_episode))

            reward_of_running_episode = 0

            list_of_steps_in_running_episode = []

            next_observation = env.reset()

            if len(this_batch_of_episodes) == batch_size:


                yield this_batch_of_episodes#------------Return sends a specified value back to its caller whereas Yield can produce a sequence of values
                '''Even after returning this_batch_of_episodes, the code continues from here'''
                this_batch_of_episodes = []

        current_observation = next_observation




'''a function that keeps only the episodes with cumulative reward greater than the threshold'''
def filtered_episodes(batch_of_episodes, percentile):


    rewards_in_this_batch = list(map(lambda argument: argument.reward, batch_of_episodes))
    reward_boundary_to_use = np.percentile(rewards_in_this_batch, percentile)
    average_reward_in_this_batch = np.mean(rewards_in_this_batch)

    training_observations_in_the_episode = []

    actions_taken_at_those_observation_states = []

    for episode in batch_of_episodes:


        if episode.reward < reward_boundary_to_use:
            continue

        training_observations_in_the_episode.extend(map(lambda step: step.observation, episode.episode_steps))
        actions_taken_at_those_observation_states.extend(map(lambda step: step.action, episode.episode_steps))



    train_observations = torch.FloatTensor(training_observations_in_the_episode)
    train_actions = torch.LongTensor(actions_taken_at_those_observation_states)


    return train_observations, train_actions, reward_boundary_to_use, average_reward_in_this_batch


if __name__ == "__main__":


    print("Current cuda device is ", torch.cuda.current_device())
    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    print("Current cuda device name is ", torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Create the environment to test on
    env = gym.make("CartPole-v0") #we will be using this continuous action space as Cross Entropy is less robust toward discrete action with a final reward
                                    #setting such as the frozen lake env
    #lets monitor the nehavior of the agent by recording it as well on video
    env = gym.wrappers.Monitor(env, directory = "cross_entropy_cartpole", force = True)

    #------------------------------------------------Important piece
    observation_space_dimension = env.observation_space.shape[0]
    action_space_dimension = env.action_space.n


    '''Instantiate the policy network'''
    policy_network = generic_Neural_Network(observation_space_dimension, 64, action_space_dimension).cuda()

    objective_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params= policy_network.parameters(), lr= 0.001)

    writer = SummaryWriter()



    '''Training Loop'''

    for batch_number, episodes_in_the_batch in enumerate(iterate_batches_of_episodes(env, policy_network, batch_size)):

        filtered_observations, actions_from_good_episodes, reward_boundary, avg_reward_in_the_batch = filtered_episodes(episodes_in_the_batch, percentile)

        #zeroing the existing gradients of the parameters that need gradients to be calculated
        optimizer.zero_grad()



        filtered_observations = filtered_observations.cuda()


        actions_taken_by_the_policy_network = policy_network(filtered_observations)

        actions_from_good_episodes = actions_from_good_episodes.cuda()
        actions_taken_by_the_policy_network = actions_taken_by_the_policy_network.cuda()


        loss = objective_function(actions_taken_by_the_policy_network, actions_from_good_episodes)

        loss.backward()

        optimizer.step()

        print("Batch No: %d, Loss=%.3f, Average Reward in batch=%.1f, Filter threshold for reward=%.1f" % (batch_number, loss.item(), avg_reward_in_the_batch, reward_boundary))


        '''Record for visualization'''
        writer.add_scalar("Loss", loss.item(), batch_number)
        writer.add_scalar("Reward filter threshold", reward_boundary, batch_number)
        writer.add_scalar("Average reward in the batch", avg_reward_in_the_batch, batch_number)


        '''Keeping a threshold to decide when to consider that the environment is solved.'''
        if avg_reward_in_the_batch > 199:
            print("The Cross-Entropy method solved the environment.")
            break
    writer.close()

    # for batch_number, batch_of_episodes in enumerate(generate_batches_of_episodes(env, policy_network, batch_size)):
    #     my_list = ['geeks', 'for']
    #     another_list = [6, 0, 4, 1]
    #     my_list.extend(another_list) gives out
    #
    # ['geeks', 'for', 6, 0, 4, 1]
