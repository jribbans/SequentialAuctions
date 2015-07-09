from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp import MDPBidder1
import random
import numpy

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
num_bidders = 2

# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1.0] * len(possible_types)
type_dist_disc = False
bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
learner = MDPBidder1(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
current_round = 1
learner.calc_prob_winning(bidders, current_round)
