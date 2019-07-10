import gym, random

'''
Wrappers are used to extend the functionality of the environment.
'''
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


'''
Note:
Question:
I am very very confused about the gradients when we do back-propagation. 
According to the Pytorch’s Autograd webpage, it says that if we want a tensor/variable to contain gradient information, 
we need to set require_grad=True. I have gone through many examples on the web, for instance: reinforce algorithm example 9, 
or policy gradient example 10, they didn’t set require_grad=True.

Answer
I am wondering, in that case, when they do loss.backward(), and they didn’t even specify a single instance that require_grad=True, what exactly is happening? Do they event do back-propagation?
we do not need the gradient of input(In most cases, they are useless, unless some special works like neural style transfer, 
where we only iteratively change the input to optimize the total loss). Usually, we only want to get the model trained. 
Parameters in each layer are default to be requires_grad=True.

 Are you saying that, if we are building network in the following fashion, 
 we don’t need to worry about the setting the any tensor to have require_grad = True?
I am very confused because I was reading the tutorial from here: autograd tutorial 17, 
and it was emphasizing about the require_grad= True, where as in the example below, we don’t care about this.

 The learnable weights are registered as Parameter which is default as requires_grad=True, see here 45. 
 Input of the networks needs no gradient(They are useless in most cases). So everything is fine.

'''