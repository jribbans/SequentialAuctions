from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp_uai_aug_s import MDPBidderUAIAugS
from bidder.menezes_monteiro import MenezesMonteiroBidder
from bidder.simple import SimpleBidder
from bidder.weber import WeberBidder
import random
import numpy
import matplotlib.pyplot as plt

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 1
num_bidders = 2

# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist_disc = False
if type_dist_disc:
    type_dist = [1.0 / len(possible_types)] * len(possible_types)
else:
    type_dist = [1.0] * len(possible_types)

num_mc = 100000

# bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_mc)

# Check that this runs
learner.valuations = [.2, .1]
learner.calc_expected_rewards()
learner.solve_mdp()

# Compare against a bidder
bidders[0].reset()
b0 = [0] * len(bidders[0].possible_types)
lb0 = [0] * len(learner.possible_types)
lb10 = [0] * len(learner.possible_types)
lb11 = [0] * len(learner.possible_types)
for t_idx, t in enumerate(learner.possible_types):
    bidders[0].valuations = [t, t / 2.0]
    learner.valuations = [t, t / 2.0]
    if t_idx == 0:
        learner.calc_expected_rewards()
    else:
        learner.calc_terminal_state_rewards()
    learner.solve_mdp()
    b0[t_idx] = bidders[0].place_bid(1)
    lb0[t_idx] = learner.place_bid(1)
    # lb10[t_idx] = learner.place_bid(2)
    # learner.num_goods_won = 1
    # lb11[t_idx] = learner.place_bid(2)
    # learner.num_goods_won = 0
    print(learner.valuations, lb0[t_idx], lb10[t_idx], lb11[t_idx])

"""
plt.figure()
for s in learner.price_prediction.keys():
    if len(s) == 2:
        R = [learner.R[(s,a)] for a in learner.action_space]
        plt.plot(possible_types, R, label=str(s))
    else:
        R = [learner.R[(s,a)] for a in learner.action_space]
        plt.plot(possible_types, R, label=str(s))
    plt.legend()
plt.show()
"""

plt.figure()
plt.plot(possible_types, b0, label='Katzman')
plt.plot(possible_types, lb0, label='MDP', marker='o', markersize=3)
plt.xlabel('Valuation')
plt.ylabel('Bid')
plt.legend()
plt.show()
