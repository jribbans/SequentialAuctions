from auction.SequentialAuction import SequentialAuction
from bidder.mdp_uai_aug_s import MDPBidderUAIAugS
from bidder.mdp_uai import MDPBidderUAI
from bidder.simple import SimpleBidder
import random
import numpy
import itertools
import copy

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
num_mc = 10000
bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
learner = MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
# learner = MDPBidderUAI(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_mc)
learner.valuations = copy.deepcopy(bidders[1].valuations)
learner.calc_expected_rewards()
learner.solve_mdp()
print('Test if the bidder bids truthfully')
final_round_truthful = learner.is_bidding_valuation_in_final_round()
print('Bidding truthfully in final round =', final_round_truthful)

# Display what was learned
print('Transitions: state \t action \t next state \t probability')
sorted_keys = list(learner.T.keys())
sorted_keys.sort()
for k in sorted_keys:
    print(k[0], '\t', k[1], '\t', k[2], '\t', learner.T[k])

# See how the learner performs
policies = {}
exp_payment = {}
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
    policies[tuple(v)] = copy.deepcopy(learner.pi)
    exp_payment[tuple(v)] = copy.deepcopy(learner.exp_payment)
    print('Valuation vector =', v)
    print('First round bid =', learner.place_bid(1))
    for s in learner.state_space:
        Q = {}
        if s in learner.terminal_states:
            continue
        maxQ = -float('inf')
        for a in learner.action_space:
            Q[a] = learner.Q[(s, a)]
            maxQ = max(maxQ, learner.Q[(s, a)])
        maxQ_actions = [a for a in learner.action_space if learner.Q[(s, a)] == maxQ]
        best_action = min(maxQ_actions)
        print('State', s, '. Optimal Action =', best_action, '. Q of each action:', Q)
    print('Values at terminal states')
    for s in learner.terminal_states:
        print('State', s, '. V[s] =', learner.V[s])
    if not (learner.is_bidding_valuation_in_final_round()):
        truthful_result_output = 'Does not bid truthfully in last round.'
    else:
        truthful_result_output = 'Truthful bidding in last round'
    print(truthful_result_output)

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

current_pi = copy.deepcopy(policies)
next_pi = copy.deepcopy(policies)
pi_converged = False
pi_converged_iter = 0
while not pi_converged:
    pi_converged_iter += 1
    print('Convergence iteration =', pi_converged_iter)

    current_pi = copy.deepcopy(next_pi)
    mdp_bidders = []
    for i in range(num_bidders):
        mdp_bidders.append(
            MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc))
        mdp_bidders[i].make_valuations = copy.deepcopy(bidders[i].make_valuations)
        mdp_bidders[i].use_given_pi = True
        mdp_bidders[i].given_pi = copy.deepcopy(current_pi)

    iter_learner = MDPBidderUAIAugS(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
    iter_learner.learn_auction_parameters(mdp_bidders, num_mc)
    calc_rewards = True
    iter_policies = {}
    iter_exp_payment = {}
    for v in itertools.product(possible_types, repeat=num_rounds):
        iter_learner.valuations = v
        if calc_rewards:
            iter_learner.calc_expected_rewards()
            calc_rewards = False
        else:
            iter_learner.calc_terminal_state_rewards()
        iter_learner.solve_mdp()
        iter_learner.is_bidding_valuation_in_final_round()
        iter_policies[tuple(v)] = copy.deepcopy(iter_learner.pi)
        iter_exp_payment[tuple(v)] = copy.deepcopy(iter_learner.exp_payment)
        #print('Valuation vector', v)
        #print('Policy/Iter Policy')
        #print(policies[tuple(v)])
        #print(iter_policies[tuple(v)])

    sa = SequentialAuction([bidders[0], iter_learner], num_rounds)
    util_learner = [-1] * num_trials
    for t in range(num_trials):
        for bidder in bidders:
            bidder.reset()
            bidder.valuations = bidder.make_valuations()
        iter_learner.reset()
        iter_learner.valuations = bidders[1].valuations
        iter_learner.calc_terminal_state_rewards()
        iter_learner.solve_mdp()
        sa.run()
        util_learner[t] = sum(iter_learner.utility)
    print('Avg utility, Simple:', sum(util_simple) / num_trials)
    print('Avg utility, Learner:', sum(util_learner) / num_trials)

    next_pi = copy.deepcopy(iter_policies)
    if next_pi == current_pi:
        pi_converged = True
        print('Converged after', pi_converged_iter, 'iterations')

    for k in current_pi.keys():
        print('orig', k, current_pi[k])
        print('next', k, next_pi[k])
