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
num_rounds = 2
num_bidders = 2

# Values in possible_types must be increasing.
# possible_types = [i / 100.0 for i in range(101)]
possible_types = [i / 20.0 for i in range(21)]
# possible_types = [3, 10]
# possible_types = [i / 10.0 for i in range(11)]
# possible_types = [i for i in range(3)]
type_dist_disc = True
if type_dist_disc:
    type_dist = [1.0 / len(possible_types)] * len(possible_types)
else:
    type_dist = [1.0] * len(possible_types)

num_mc = 10000

bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
#bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_mc)

"""
print('Transitions')
for k in learner.T.keys():
    if learner.T[k] > 0.000000001:
        print(k[0], '    ', k[1], '    ', k[2], '    ', learner.T[k])
"""

# Check that this runs
learner.valuations = [.1, .1]
learner.calc_expected_rewards()
learner.solve_mdp()
final_round_truthful = learner.is_bidding_valuation_in_final_round()
print('Bidding truthfully in final round =', final_round_truthful)

# Compare against a bidder
bidders[0].reset()
num_trials = 100
b0 = [0] * num_trials
lb0 = [0] * num_trials
lb10 = [0] * num_trials
lb11 = [0] * num_trials
val0 = [0] * num_trials
val1 = [0] * num_trials
for r in range(num_trials):
    bidders[0].valuations = bidders[0].make_valuations()
    learner.valuations = bidders[0].valuations
    val0[r] = bidders[0].valuations[0]
    if num_rounds > 1:
        val1[r] = bidders[0].valuations[1]
    if r == 0:
        learner.calc_expected_rewards()
    else:
        learner.calc_terminal_state_rewards()
    learner.solve_mdp()
    b0[r] = bidders[0].place_bid(1)
    lb0[r] = learner.place_bid(1)
    if num_rounds > 1:
        lb10[r] = learner.place_bid(2)
        learner.num_goods_won = 1
        lb11[r] = learner.place_bid(2)
        learner.num_goods_won = 0
    if not(learner.is_bidding_valuation_in_final_round()):
        truthful_result_output = 'Does not bid truthfully in last round.'
    else:
        truthful_result_output = 'Truthful bidding in last round'
    print(learner.valuations, round(b0[r], learner.digit_precision), lb0[r], lb10[r], lb11[r], truthful_result_output)

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
plt.scatter(val0, b0, label='Katzman', marker='x')
plt.scatter(val0, lb0, label='MDP', marker='o')
plt.xlabel('Valuation')
plt.ylabel('Bid')
plt.legend()
plt.show()
