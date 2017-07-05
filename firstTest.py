import gym
import universe
import random
import enum


class InputNode:
    def __init__(self, low, high, value=0):
        self.low = low
        self.high = high
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class Operator(enum):
    RANDOM=-1
    ADD = 0,
    SUBTRACT=1,
    MULTIPLY=2,
    DIVIDE=3,

    def operate(self, left, right):
        if self.value == self.ADD:
            return left+right
        elif self.value == self.SUBTRACT:
            return left-right
        elif self.value == self.MULTIPLY:
            return left*right
        elif self.value == self.DIVIDE:
            return left/right


class MathNode:
    def __init__(self, left, right, operator=Operator.RANDOM):
        self.left = left
        self.right = right
        self.operator = operator if operator != Operator.RANDOM else random.choice(list(Operator))
        self.high = left.high + right.high
        self.low = left.low + right.low

    def get(self):
        return self.operator.operate(self.left.get() + self.right.get())


class SigmoidNeuronNode:
    def __init__(self, inputs, weights, low, high):
        self.inputs = inputs
        self.weights = weights
        self.low = low
        self.high = high

    def get(self):
        output = 0
        for i in range(self.inputs):
            inputValue = self.inputs[i] #because 'input' name is taken
            weight = self.weights[i]
            output += inputValue*weight
        output /= len(self.inputs)
        return output


class SigmoidNetwork:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        pass

env = gym.make('Assault-v0')
print(env.action_space)
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
