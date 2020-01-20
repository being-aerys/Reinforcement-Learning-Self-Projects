#WORKING TD3 code


#taken from sfujim on github
#This implementation of TD3 did not work for mountain car continuous, worked for pendulum and continuous cartpole.
#turns out TD3 is usually unable to solve MountainCar Continuous because of sparse reward in MountainCar continuous --> classic temporal credit assignment problem
#however, Soft Actor Critic solves this problem easily
#https://www.reddit.com/r/MachineLearning/comments/axoqz6/d_state_of_the_art_deeprl_still_struggles_to/

#didnot work even when ran for 3000 episodes for mountaincar continuous
#However, since our env is not sparse, this TD3 implementation should work. WILL USE THIS TD3 code

#

import numpy as np
from PIL import Image
import torch

import gym
import argparse
import os, time, Continuous_Cartpole

import utils
import TD3
import OurDDPG
import DDPG, matplotlib.pyplot as plt
import os
def plot(rewards,train_or_test_flag,env_name):
	if train_or_test_flag == 0:

		plt.figure(figsize=(20,5))
		plt.plot(rewards)
		plt.savefig('td3_training_rewards' + str(env_name)+'.png')
		# plt.show()
	else:
		plt.figure(figsize=(20, 5))
		plt.plot(rewards)
		plt.savefig('td3_testing_rewards' + str(env_name) + '.png')


# plt.show()
'''get cont cartpole working see that wpisode 25 baata nabadhne issue then make sure td3 works on this, yedi yesma kaam gare mountaincar ma kaam garnu paryo'''

def run_policy(env_name):
    test_rewards = []
    # env_name = "MountainCarContinuous-v0"
    random_seed = 0
    n_episodes = 5
    lr = 0.002
    max_timesteps = 200 #evaluating for more timesteps even though trained for less time steps
	#'''Getting only sub 200 rewards because traing only for sub 200?????????????????'''
    render = True
    save_gif = True

    filename = "TD3_{}_{}".format(env_name, random_seed)
    filename += '_solved'

    directory = "./preTrained/"+str(env_name)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    #policy = TD3.TD3(lr, state_dim, action_dim, max_action)

    policy.load(filename)

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

        '''REMOVE THIS env.close() when not using continuous cartpole'''
        #env.close()

    return test_rewards




# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			eval_env.render()
			avg_reward += reward
	eval_env.close()

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over "+ str(eval_episodes) + " episodes: " +str(avg_reward))
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":



	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
	#parser.add_argument("--env", default="MountainCarContinuous-v0")  # OpenAI gym environment name
	#parser.add_argument("--env", default="Pendulum-v0")  # OpenAI continuous action gym environment name
	parser.add_argument("--env", default="ContinuousCartPoleEnv")  # OpenAI continuous action gym environment name
	#BipedalWalker-v2
	#Pendulum-v0
	#CarRacing-v0
	#HalfCheetah-v2
	#MountainCarContinuous-v0

	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=10000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=3000000, type=int)  # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)  # Stdev of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)  # Discount factor
	parser.add_argument("--tau", default=0.005)  # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_false")  # Save model and optimizer parameters #default is TRUE since no default provided
	parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = str(args.policy)+"_"+str(args.env) +"_"+ str(args.seed)
	print("---------------------------------------")
	print("Policy:"+ str(args.policy) + " \nEnv: " + str(args.env) + "\nSeed: " + str(args.seed))
	time.sleep(1)
	print("---------------------------------------")
	env_name = str(args.env)
	directory = "./preTrained/{}".format(env_name)  # save trained models
	#Create a directory to store the results
	if not os.path.exists("./results"):
		os.makedirs("./results")
	#Create a directory to save the models
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	#env = gym.make(args.env)
	env = Continuous_Cartpole.ContinuousCartPoleEnv()
	# Set seeds for gym, pytorch, and random
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	# elif args.policy == "OurDDPG":
	# 	policy = OurDDPG.DDPG(**kwargs)
	# elif args.policy == "DDPG":
	# 	policy = DDPG.DDPG(**kwargs)

	#If we want to continue training from a certain model that was saved earlier
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load("./models/" +str(policy_file))

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	#evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	training_rewards = []
	test_rewards = []

	'''Fill the replay buffer until the minimum number of transitions are available'''
	for timestep in range(int(args.max_timesteps)): #VVI this max timesteps means the sum of the time steps of all the episodes combined
		
		episode_timesteps += 1

		# Select action randomly until the buffer fills then according to policy
		if timestep < args.start_timesteps:
			action = env.action_space.sample()
		else:
			#add gaussian noise to explore, then clip
			action = (policy.select_action(np.array(state))+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		#max_timesteps = 150 if
		'''This done_bool allows us to train beyond the max time steps defined in the gym environment.'''
		done_bool = 0
		if episode_timesteps < env._max_episode_steps:
			done_bool = float(done)
		else:
			done_bool = 1 #*** location

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# At each time step of every episode, train the agent for a fixed number of times
		# required in the buffer
		if timestep >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		'''KEEP THIS TO MAKE SURE YOU DONT GO BEYOND A CERTAIN TIMESTEP IN EACH EPISODE'''

		# if episode_timesteps > env._max_episode_steps:
		# 	print(done_bool+ "here")
		# 	time.sleep(1)

		if done_bool: #made done_bool here instead of done from the original code after changing at *** location

			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print("Total Timesteps: " + str(timestep + 1) + " Episode No: " + str(episode_num + 1) +
				  " Timesteps in this episode:" + str(episode_timesteps) + " Reward in this episode: " + str(episode_reward))
			training_rewards.append(episode_reward)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			#print(episode_num)

		# Evaluate episode after a certain number of timesteps
		if (timestep + 1) % args.eval_freq == 0:
			#print("Episode Num + 1: ", episode_num + 1)
			#print((episode_num + 1) % args.eval_freq)

			plot(training_rewards, 0, env_name)
			#evaluations.append(eval_policy(policy, args.env, args.seed))
			#np.save("./results/"+ str(file_name), evaluations)
			# if args.save_model:
			# 	policy.save("./models/"+str(file_name))

			'''RENDER using the last saved policy'''
			name = file_name + '_solved'
			policy.save(name)
			directory = "./preTrained/{}".format(env_name)
			test_rewards_returned = run_policy(env_name)
			test_rewards = test_rewards + test_rewards_returned
			plot(test_rewards, 1, env_name)

	print("Finished all time steps")
