from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp import MDPBidder
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
num_rounds = 2
num_bidders = 2

# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1.0] * len(possible_types)
type_dist_disc = False

num_trials_per_action = 100

bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidder(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_trials_per_action)

"""
for r in range(num_rounds):
    print("Round", r)
    print("prob winning")
    for a_idx, a in enumerate(learner.action_space):
        print(a, learner.prob_winning[r][a_idx])
    print("price dist")
    for p_idx, p in enumerate(learner.price[r]):
        print(p, learner.price_dist[r][p_idx], learner.price_cdf[r][p_idx])
"""

"""
plt.figure()
for r in range(num_rounds):
    plt.plot(learner.action_space, learner.prob_winning[r], label='Prob Winning r = ' + str(r))
    plt.plot(learner.price[r], learner.price_dist[r], label='Price PDF r =' + str(r))
    plt.plot(learner.price[r], learner.price_cdf[r], label='Price CDF r = ' + str(r))
plt.xlabel('Bid')
plt.ylabel('Probability/Density')
plt.legend()
plt.show()
"""

learner.valuations = [1, .1]
learner.calc_expected_rewards()
learner.solve_mdp()
print(learner.place_bid(1))
print(learner.place_bid(2))
