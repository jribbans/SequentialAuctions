from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp import MDPBidder
from bidder.menezes_monteiro import MenezesMonteiroBidder
from bidder.simple import SimpleBidder
from bidder.weber import WeberBidder
import random
import numpy

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
# num_bidders = 2
num_bidders = 3

# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1.0] * len(possible_types)
type_dist_disc = False

num_trials_per_action = 100

# bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidder(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
current_round = 1
learner.calc_prob_winning(bidders, current_round, num_trials_per_action)
for a_idx, a in enumerate(learner.action_space):
    print(a, learner.prob_winning[current_round - 1][a_idx])
