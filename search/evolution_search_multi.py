import sys
# update your projecty root path before running
sys.path.insert(0, '/home/lnn/tensor-net-search')

import os
import time
import logging
import argparse
from misc import util

import numpy as np
from search import tensor_train

from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize
from collections import Counter

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=30, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=100, help='population size')
parser.add_argument('--n_offspring', type=int, default=30, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--ranknum', type=int, default=20, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=20, help='# of epochs to train during architecture search')
parser.add_argument('--P_gens', type=int, default=100, help='# of epochs to train during architecture search')
args = parser.parse_args()
args.save = 'search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))

util.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=2, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=20, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self.ranknum = 20
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled


    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))
            logging.info('rank code = {}'.format(x[i]))
            if arch_id == 1:
                self.rank_ratio = list(5 for i in range(self.ranknum))

                self.code_bias = list(0 for i in range(self.ranknum))

                self.arc_best = list(100 for i in range(5))
                self.arc_code = [list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum))]
                # best_code1 = [0, 0, 0, 0, 0, 0, 0]
                # best_code2 = [0, 0, 0, 0, 0, 0, 0]

            if arch_id == 1500:
                self.rank_ratio = list(2 for i in range(self.ranknum))
                best_code1 = []
                arc_code_T = list(map(list, zip(*self.arc_code)))
                for j in range(len(arc_code_T)):
                    best_code1.append(max(arc_code_T[j], key=arc_code_T[j].count))
                print("rank_ratio:", self.rank_ratio,
                      "arc_code_T:", arc_code_T,
                      "best_code1:", self.best_code1)

                self.arc_best = list(100 for i in range(5))
                self.arc_code = [list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum)),
                            list(0 for i in range(self.ranknum))]
                # best_code1 = []

                self.code_bias = []
                for b in range(len(best_code1)):
                    if best_code1[b] == 1:
                        self.code_bias.append(0)
                    elif best_code1[b] == 2:
                        self.code_bias.append(4)
                    elif best_code1[b] == 3:
                        self.code_bias.append(10)
                    elif best_code1[b] == 4:
                        self.code_bias.append(14)
                    elif best_code1[b] == 5:
                        self.code_bias.append(20)
                    print("code_bias:", self.code_bias)
            if arch_id == 2400:
                self.rank_ratio = list(1 for i in range(self.ranknum))
                best_code2 = []
                self.code_bias = []
                arc_code_T = list(map(list, zip(*self.arc_code)))
                for j in range(len(arc_code_T)):
                    best_code2.append(max(arc_code_T[j], key=arc_code_T[j].count))
                for i in range(len(best_code2)):
                    if best_code1[i] == 1 and best_code2[i] == 1:
                        self.code_bias.append(0)
                    elif best_code1[i] == 1 and best_code2[i] == 2:
                        self.code_bias.append(1)
                    elif (best_code1[i] == 1 and best_code2[i] == 3) or (best_code1[i] == 2 and best_code2[i] == 1):
                        self.code_bias.append(3)
                    elif (best_code1[i] == 1 and best_code2[i] == 4) or (best_code1[i] == 2 and best_code2[i] == 2):
                        self.code_bias.append(5)
                    elif (best_code1[i] == 1 and best_code2[i] == 5) or (best_code1[i] == 2 and best_code2[i] == 3):
                        self.code_bias.append(7)
                    elif (best_code1[i] == 2 and best_code2[i] == 4) or (best_code1[i] == 3 and best_code2[i] == 1):
                        self.code_bias.append(9)
                    elif (best_code1[i] == 2 and best_code2[i] == 5) or (best_code1[i] == 3 and best_code2[i] == 2):
                        self.code_bias.append(11)
                    elif (best_code1[i] == 3 and best_code2[i] == 3) or (best_code1[i] == 4 and best_code2[i] == 1):
                        self.code_bias.append(13)
                    elif (best_code1[i] == 3 and best_code2[i] == 4) or (best_code1[i] == 4 and best_code2[i] == 2):
                        self.code_bias.append(15)
                    elif (best_code1[i] == 3 and best_code2[i] == 5) or (best_code1[i] == 4 and best_code2[i] == 3):
                        self.code_bias.append(17)
                    elif (best_code1[i] == 4 and best_code2[i] == 4) or (best_code1[i] == 5 and best_code2[i] == 1):
                        self.code_bias.append(19)
                    elif (best_code1[i] == 4 and best_code2[i] == 5) or (best_code1[i] == 5 and best_code2[i] == 2):
                        self.code_bias.append(21)
                    elif best_code1[i] == 5 and best_code2[i] == 3:
                        self.code_bias.append(23)
                    elif best_code1[i] == 5 and best_code2[i] == 4:
                        self.code_bias.append(25)
                    elif best_code1[i] == 5 and best_code2[i] == 5:
                        self.code_bias.append(27)



            # call back-propagation training
            # if self._search_space == 'micro':
            #     genome = micro_encoding.convert(x[i, :])
            # elif self._search_space == 'macro
            # ':
            #     genome = macro_encoding.convert(x[i, :])
            rank_ratio = self.rank_ratio
            code_bias = self.code_bias
            performance = tensor_train.main(rank_code=x[i],
                                            epochs=self._epochs,
                                            rank_ratio=rank_ratio,
                                            code_bias=code_bias,
                                            save='arch_{}'.format(arch_id))

            logging.info('accuracy = {}'.format(performance['valid_acc'],
                                                performance['compression_ratio']))


            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance['valid_acc']
            objs[i, 1] = 100 - performance['compression_ratio']
            if objs[i, 0] <= max(self.arc_best):
                max_index = self.arc_best.index(max(self.arc_best))
                self.arc_best[max_index] = objs[i, 0]
                self.arc_code[max_index] = x[i]


            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
# arc_code = []
def do_every_generations(algorithm):

    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # a = NAS.

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("compression_ratio: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))



def main():
    np.random.seed(args.seed)

    np.random.seed(args.seed)
    logging.info("args = %s", args)

    # # setup NAS search problem
    #
    # n_var = int(args.ranknum * 5)
    # lb = np.zeros(n_var)
    # ub = np.ones(n_var)


    # setup NAS search problem

    n_var = int(args.ranknum)
    lb = np.ones(n_var) * 1
    ub = np.ones(n_var) * 5


    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  # init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return


if __name__ == "__main__":
    main()