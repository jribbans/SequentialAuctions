from auction.SequentialAuction import SequentialAuction
from bidder.mdp_uai_aug_s import MDPBidderUAIAugS
from bidder.mdp_uai import MDPBidderUAI
from bidder.simple import SimpleBidder
import random
import numpy
import itertools

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
num_bidders = 2

# Valuations
possible_types = [3, 10]
type_dist_disc = True
type_dist = [.8, .2]

# Learn the MDP
num_mc = 30000
bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
learner = MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
# learner = MDPBidderUAI(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_mc)

# Display what was learned
print('Transitions: state \t action \t next state \t probability')
sorted_keys = list(learner.T.keys())
sorted_keys.sort()
for k in sorted_keys:
    print(k[0], '\t', k[1], '\t', k[2], '\t', learner.T[k])

# See how the learner performs
print('See how learner bids')
calc_rewards = True
for v in itertools.product(possible_types, repeat=num_rounds):
    learner.valuations = v
    if calc_rewards:
        learner.calc_expected_rewards()
        calc_rewards = False
    else:
        learner.calc_terminal_state_rewards()
    learner.solve_mdp()

# Compare utility of simple vs learned
print('Run Learner and see how it does')
num_trials = 1000
sa = SequentialAuction(bidders, num_rounds)
util_simple = [-1] * num_trials
for t in range(num_trials):
    for bidder in bidders:
        bidder.reset()
        bidder.valuations = bidder.make_valuations()
    sa.run()
    util_simple[t] = sum(bidders[1].utility)

sa = SequentialAuction([bidders[0], learner], num_rounds)
util_learner = [-1] * num_trials
for t in range(num_trials):
    for bidder in bidders:
        bidder.reset()
        bidder.valuations = bidder.make_valuations()
    learner.reset()
    learner.valuations = bidders[1].valuations
    learner.calc_terminal_state_rewards()
    learner.solve_mdp()
    sa.run()
    util_learner[t] = sum(learner.utility)

print('Avg utility, Simple:', sum(util_simple) / num_trials)
print('Avg utility, Learner:', sum(util_learner) / num_trials)
