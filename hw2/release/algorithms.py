from env_MAB import *
import numpy as np


def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])


class Explore:
    def __init__(self, MAB):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.pull_counts = np.zeros(self.K)
        
    def reset(self):
        self.MAB.reset()
        self.pull_counts = np.zeros(self.K)

    def play_one_step(self):
        chosen_arm = random_argmax(-self.pull_counts)
        self.MAB.pull(chosen_arm)
        self.pull_counts[chosen_arm] += 1


class Greedy:
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        pull_counts = np.sum(record, axis = 1)
        if np.min(pull_counts) == 0:
            chosen_arm = random_argmax(-pull_counts)
        else: 
            chosen_arm = random_argmax(record[:, 1]/pull_counts)
        self.MAB.pull(chosen_arm)


class ETC:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()
        self.N_e = int((((self.T * (np.log(2*self.K/delta)/2) ** 0.5))/self.K) ** (2/3))
        self.best_arm = None

    def reset(self):
        self.MAB.reset()
        self.best_arm = None

    def play_one_step(self):
        record = self.MAB.get_record()
        pull_counts = np.sum(record, axis = 1)
        if np.min(pull_counts) < self.N_e:
            chosen_arm = random_argmax(-pull_counts)
        else: 
            if self.best_arm == None:
                self.best_arm = random_argmax(record[:, 1]/pull_counts)
            chosen_arm = self.best_arm
        self.MAB.pull(chosen_arm)


class Epgreedy:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.T = MAB.get_T()
        self.K = MAB.get_K()

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        # add 1 to avoid dividing by zero
        t = np.sum(record) + 1 
        eps_t = (self.K * np.log(t)/t) ** (1/3)
        u = np.random.random()
        pull_counts = np.sum(record, axis = 1)

        if np.min(pull_counts) == 0:
            chosen_arm = random_argmax(-pull_counts)
        elif u < eps_t:
            chosen_arm = np.random.choice(self.K)
        else: 
            chosen_arm = random_argmax(record[:, 1]/pull_counts)
        self.MAB.pull(chosen_arm)


class UCB:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.T = MAB.get_T()
        self.K = MAB.get_K()
        self.delta = delta

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        pull_counts = np.sum(record, axis = 1)
        if np.min(pull_counts) == 0:
            chosen_arm = random_argmax(-pull_counts)
        else:
            chosen_arm = np.argmax(record[:, 1]/pull_counts + ((np.log(self.K * self.T/self.delta))/pull_counts) ** 0.5)
        self.MAB.pull(chosen_arm)


class Thompson_sampling:
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.MAB.reset()

    def play_one_step(self):
        """
        Implement one step of the Thompson sampling algorithm.
        """
        record = self.MAB.get_record()
        parameters = 1 + record
        samples = np.random.beta(parameters[:, 1], parameters[:, 0])
        chosen_arm = random_argmax(samples)
        self.MAB.pull(chosen_arm)
        
