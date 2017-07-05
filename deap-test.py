#    This file uses DEAP (https://github.com/DEAP/deap)
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy
import gym
from gym import wrappers
import operator
import numbers

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from scoop import futures

import pygraphviz as pgv


def progn(*args):
    for arg in args:
        arg()


# def if_then_else(condition, out1, out2):
#     out1() if condition() else out2()

def limit(input, minimum, maximum):
    assert isinstance(input, float)
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input

def pluck(input, i):
    assert isinstance(input, list)
    assert isinstance(i, int)
    # print('plucking', i, 'from', input)
    return input[i]

def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1


def add(left,right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left+right


def sub(left,right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left-right


def mul(left,right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left*right


def div(left,right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left/right


def my_abs(input):
    return abs(input)


# Define a new if-then-else function
def if_then_else(input, output1, output2):
    assert isinstance(input, bool)
    if input: return output1
    else: return output2

def gt(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left > right

def lt(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left < right

def eq(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left == right

def ne(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left != right

def lte(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left <= right

def gte(left, right):
    assert isinstance(left, numbers.Number) and isinstance(right, numbers.Number)
    return left >= right

# pset = gp.PrimitiveSet("MAIN", 4)
# pset = gp.PrimitiveSetTyped("MAIN", [list], [int, int, int, int])

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(pluck, 2)
pset.addPrimitive(limit, 3)
pset.addPrimitive(my_abs, 1)
pset.addPrimitive(gt, 2)
pset.addPrimitive(lt, 2)
pset.addPrimitive(lte, 2)
pset.addPrimitive(gte, 2)
pset.addPrimitive(eq, 2)
pset.addPrimitive(ne, 2)
pset.addPrimitive(if_then_else, 3)
pset.addTerminal(0)
pset.addTerminal(1)
pset.addTerminal(2)
pset.addTerminal(3)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=2, max_=10)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


env = gym.make('CartPole-v1')
# env = wrappers.Monitor(env, './out')

def graph(expr):
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw('out.png')


def evalIndividual(individual, render=False):
    # Transform the tree expression to functional Python code
    action = gp.compile(individual, pset)
    # graph(individual)
    # print(action)
    fitness = 0
    failed = False
    for x in range(0, 10):
        done = False
        observation = env.reset()
        while not done:

            # print(action_result)
            try:
                if failed:
                    action_result = 0
                else:
                    action_result = action(observation)
                observation, reward, done, info = env.step(action_result)
            except:
                # failed = True #throw out any individual that throws any type of exception
                # observation, reward, done, info = env.step(0)
                return (0,) #If your not recording you can reset the environment early,
            if render:
                env.render()
            # if(fitness > 450):
            #     env.render()
            fitness += reward
    # envInUse[i] = False
    return (0,) if failed else (fitness,)

toolbox.register("map", futures.map)
toolbox.register("evaluate", evalIndividual)
toolbox.register("select", tools.selTournament, tournsize=100)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.35, 3450, stats=mstats,
    #                                halloffame=hof, verbose=True)
    pop, log = algorithms.eaGenerateUpdate(pop, 500, stats=mstats,
                                            halloffame=hof, verbose=True)

    winner = gp.compile(hof[0], pset)
    graph(hof[0])
    evalIndividual(hof[0], True)

    # print log
    return pop, log, hof


    # pop = toolbox.population(n=300)
    # hof = tools.HallOfFame(1)
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean)
    # stats.register("std", numpy.std)
    # stats.register("min", numpy.min)
    # stats.register("max", numpy.max)
    #
    # algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)
    #
    # return pop, hof, stats


if __name__ == "__main__":
    main()