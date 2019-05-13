import gym
import torch
from ddpg import DDPG
from normalized_actions import NormalizedActions#----------------------------by default [-1,1] so do not need this class here
from ounoise import OUNoise
from param_noise import Adaptive_Parameter_Noise, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from parameters import Parameters
import Continuous_Cartpole
args = Parameters()

env_name = "BipedalWalker-v2"#----------Comment this line and uncomment the next line to see another environment
#env_name = 'MountainCarContinuous-v0'#
env = gym.make(env_name)

#Comment the above lines and uncomment the following if you want to see results faster
# env_name = "Continuous Cartpole"
# env = Continuous_Cartpole.ContinuousCartPoleEnv()

agent = DDPG(args.gamma, args.tau, args.actor_hidden_size, env.observation_space.shape[0], env.action_space)

replay_buffer = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[
                      0]) if args.ou_noise else None  # ---------------------------------enable OU noise if passed as argument else discard

param_noise = Adaptive_Parameter_Noise(initial_action_stddev=0.05, desired_action_stddev=args.noise_scale,
                                       adaptation_coefficient=1.05) if args.param_noise else None

rewards_train = []
rewards_test = []
total_numsteps = 0
global_total_no_of_updates = 0


# ============================================Training
for i_episode in range(args.num_episodes):
    print("Episode No:",i_episode)
    total_numsteps = 0
    state = torch.Tensor([env.reset()])  # -----------------------reset the environment and get the default starting state

    if args.ou_noise:  # ----------------------------------------if OU noise enabled
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise:  # --------------if parameter noise enabled, add noise to the actor's parameters

        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0  # ----------------reward for the episode

    while True:  # -----------------------run the episode until we break by getting done = True after reaching the terminal state

        action = agent.select_action(state, ounoise, param_noise)  # ------------------------>select action using the learning actor

        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])  # ------------------------>returns done value. used by mask as mask = - done,


        # if next state returned is a terminal state then return done = True, hence mask becomes 0  hence V(state before terminal state) = reward + mask * some value
        env.render()
        total_numsteps += 1
        #print("timestep in the episode: ",total_numsteps)
        episode_reward += reward

        action = torch.Tensor(action.cpu())  # --------------------------convert to Tensor
        mask = torch.Tensor([ not done])  # ------------------------mask is used to make sure that we multiply all the future rewards by 0 at the terminal state
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        replay_buffer.push(state, action, mask, next_state, reward)

        state = next_state  # -------------------------------------now this next state is the new state for which we will take the action acc to the perturbed actor

        # Turns out as soon as we have more 1 element more than the replay batch size in the replay buffer, we update both actor and critic network using
        # update_parameters() at each time step at each episode.
        # Also, at the end of this update_parameters() method exists the soft update for both target actor and target critic

        if len(
                replay_buffer) > args.batch_size:  # ---------------if less elements in replay memory than the batch size chosen, dont do this else do this.

            for _ in range(
                    args.updates_per_step):  # -------Note: We can also du multiple updates even for a single timestep

                transitions = replay_buffer.sample(
                    args.batch_size)  # -------sample a number of transitions from the replay meomory

                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)  # ------------>update_parameters() is getting a batch of transitions, returns two loss values


                global_total_no_of_updates += 1
        if done:  # ------------------->if done == True, then break the while loop. Done is the end of this one single action from the actor n/w. we reach next state.
            break


    # Adapt the param_noise based on distance metric after each episode

    if args.param_noise:
        episode_transitions = replay_buffer.memory_list[replay_buffer.position - total_numsteps:replay_buffer.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
        param_noise.adapt(ddpg_dist)

    rewards_train.append(episode_reward)

    # ==============================================Testing after every 10 episodes#---------->Removed in this code because here we just wanna see it
    #learn---so I rendered all training episodes as well instead of just this testing episode
    # if i_episode % 10 == 0:
    #     state = torch.Tensor([env.reset()])
    #     episode_reward = 0
    #     while True:
    #         action = agent.select_action(state)
    #
    #         next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
    #         #env.render()#--------------------------------------------------removing render to run on the star server
    #         episode_reward += reward
    #
    #         next_state = torch.Tensor([next_state])
    #
    #         state = next_state
    #         if done:
    #             break
    #
    #     # writer.add_scalar('reward/test', episode_reward, i_episode)
    #
    #     rewards_test.append(episode_reward)
    #     # Note that this is within this if condition.
    #     print(
    #         "Current Episode No: {}, Total numsteps in the last training episode: {}, Testing reward after the last training episode: {}, "
    #         "Average training reward for the last ten training episodes: {}".format(i_episode, total_numsteps,
    #                                                                                 rewards_test[-1],
    #                                                                                 np.mean(rewards_train[-10:])))

# save the actor and the policy that you get after all the episodes

#agent.save_all_episodes_model(env_name) # ------------>  can call this at some point to save the model
env.close()
