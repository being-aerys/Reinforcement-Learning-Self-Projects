import gym
env = gym.make("CartPole-v0")
initial_observation = env.reset()

print(env.action_space)# should return Discrete(2) showing that contains a Discrete class object with 2 values
                        #the actions are either 0 = push the cartpole platform to the left and 1 = push the platform to the right
print(env.observation_space)# should return Box(4) showing that contains a Box(continuous) class object with 4 values
                            # stick's center of mass, it's speed, its angle wrt the platform, and its angular velocity
                            #range for all these 4 continuous values is [-infinity, infinity]

#We can sample observation and action space using sample() method since these are the instances of the Box and Discrete
# classes extending the Space class. Space class provides two methods: sample() and contains()

sample_action = env.action_space.sample()
sample_observation_of_the_env = env.observation_space.sample()


#Let's tell the env that we want to push to the left
next_state, reward_for_this_action_taken, has_episode_ended, extra_info = env.step(0)









