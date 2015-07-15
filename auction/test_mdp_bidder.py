from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp_uai import MDPBidderUAI
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
type_dist_disc = False
if type_dist_disc:
    type_dist = [1.0 / len(possible_types)] * len(possible_types)
else:
    type_dist = [1.0] * len(possible_types)

num_trials_per_action = 100

bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidderUAI(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_trials_per_action)

"""
for r in range(num_rounds):
    print("Round", r)
    print("prob winning")
    for a_idx, a in enumerate(learner.action_space):
        print(a, learner.prob_winning[r][a_idx])
    print("price dist")
    for p_idx, p in enumerate(learner.price_prediction[r]):
        print(p, learner.price_pdf[r][p_idx], learner.price_cdf[r][p_idx])
"""

"""
plt.figure()
for r in range(num_rounds):
    plt.plot(learner.action_space, learner.prob_winning[r], label='Prob Winning r = ' + str(r))
    # plt.plot(learner.price_prediction[r], learner.price_pdf[r], label='Price PDF r =' + str(r))
    plt.plot(learner.price_prediction[r], learner.price_cdf[r], label='Price CDF r = ' + str(r))
plt.xlabel('Bid')
plt.ylabel('Probability/Density')
plt.legend()
plt.show()
"""

bidders[0].reset()
b0 = [0] * len(bidders[0].possible_types)
lb0 = [0] * len(learner.possible_types)
for t_idx, t in enumerate(learner.possible_types):
    bidders[0].valuations = [t, t / 2.0]
    learner.valuations = [t, t / 2.0]
    if t_idx == 0:
        learner.calc_expected_rewards()
    else:
        learner.calc_end_state_rewards()
    learner.solve_mdp()
    b20 = learner.place_bid(2)
    learner.num_goods_won += 1
    b21 = learner.place_bid(2)
    learner.num_goods_won = 0
    print(t, learner.place_bid(1), b20, b21)
    b0[t_idx] = bidders[0].place_bid(1)
    lb0[t_idx] = learner.place_bid(1)

plt.figure()
plt.plot(possible_types, b0, label='Katzman')
plt.plot(possible_types, lb0, label='MDP')
plt.xlabel('Valuation')
plt.ylabel('Bid')
plt.legend()
plt.show()
