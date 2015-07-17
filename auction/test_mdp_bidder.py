from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
from bidder.mdp_uai import MDPBidderUAI
from bidder.menezes_monteiro import MenezesMonteiroBidder
from bidder.simple import SimpleBidder
from bidder.weber import WeberBidder
import random
import numpy
import matplotlib.pyplot as plt


def plot_exp_payments(mdp_bidder):
    plt.figure()
    states = set([s[0] for s in mdp_bidder.R.keys()])
    for s in states:
        R = [mdp_bidder.R[(s, a)] for a in mdp_bidder.action_space]
        exp_payment = [mdp_bidder.exp_payment[(s, a)] for a in mdp_bidder.action_space]
        plt.plot(mdp_bidder.action_space, R, label='R' + str(s))
        plt.plot(mdp_bidder.action_space, exp_payment, label='EP' + str(s))
    plt.xlabel('Bid')
    plt.ylabel('Expected Payment')
    plt.title('Calculated and Empirical Expected Payment')
    plt.legend()
    plt.show()


def plot_transition(mdp_bidder):
    plt.figure()
    for s in mdp_bidder.state_space:
        for s_ in mdp_bidder.state_space:
            T = [0] * len(mdp_bidder.action_space)
            for a_idx, a in enumerate(mdp_bidder.action_space):
                if (s, a, s_) not in mdp_bidder.T:
                    continue
                T[a_idx] = mdp_bidder.T[(s, a, s_)]
            if any(T[b] > 0 for b in range(len(mdp_bidder.action_space))):
                plt.plot(mdp_bidder.action_space, T, label=str(s) + ', ' + str(s_))

    plt.xlabel('Bid')
    plt.ylabel('Probability')
    plt.title('Probability of Transitioning to Next State')
    plt.legend()
    plt.show()


def plot_prob_winning_and_transition(mdp_bidder):
    plt.figure()

    for s in mdp_bidder.state_space:
        prob_win = [0] * len(mdp_bidder.action_space)
        for a_idx, a in enumerate(mdp_bidder.action_space):
            prob_win[a_idx] = mdp_bidder.prob_win[(s, a)]
        if any(prob_win[a_idx] > 0 for a_idx in range(len(mdp_bidder.action_space))):
            plt.plot(mdp_bidder.action_space, prob_win, label='PW ' + str(s))

    for s in mdp_bidder.state_space:
        for s_ in mdp_bidder.state_space:
            T = [0] * len(mdp_bidder.action_space)
            for a_idx, a in enumerate(mdp_bidder.action_space):
                if s_[0] < s[0]:
                    continue
                if (s, a, s_) in mdp_bidder.T.keys():
                    T[a_idx] = mdp_bidder.T[(s, a, s_)]
            if any(T[a_idx] > 0 for a_idx in range(len(mdp_bidder.action_space))):
                plt.plot(mdp_bidder.action_space, T, label='T ' + str(s) + ' ' + str(s_))

    plt.xlabel('Bid')
    plt.ylabel('Probability')
    plt.title('Probability of Winning and Transitions')
    plt.legend()
    plt.show()


def plot_Q_values(mdp_bidder):
    plt.figure()
    for s in mdp_bidder.state_space:
        if s[0] <= s[1]:
            Q = [mdp_bidder.Q[(s, a)] for a in mdp_bidder.action_space]
            plt.plot(mdp_bidder.action_space, Q, label=str(s))
    plt.xlabel('Bid')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.show()


def plot_price_pdf(mdp_bidder):
    plt.figure()
    for s in mdp_bidder.state_space:
        if (s[0] <= s[1]) and s in mdp_bidder.price_prediction.keys():
            plt.plot(mdp_bidder.price_prediction[s], mdp_bidder.price_pdf[s], label=str(s))
    plt.xlabel('Price Prediction')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()


# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
num_bidders = 2

# Values in possible_types must be increasing.
possible_types = [i / 200.0 for i in range(201)]
type_dist_disc = False
if type_dist_disc:
    type_dist = [1.0 / len(possible_types)] * len(possible_types)
else:
    type_dist = [1.0] * len(possible_types)

num_mc = 50000

bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
           for i in range(num_bidders)]
# bidders = [MenezesMonteiroBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [SimpleBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
# bidders = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
#           for i in range(num_bidders)]
learner = MDPBidderUAI(num_bidders, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
learner.learn_auction_parameters(bidders, num_mc)

# Plot what the bidder has learned
learner.valuations = [.2, .1]
learner.calc_expected_rewards()
learner.solve_mdp()
plot_exp_payments(learner)
plot_transition(learner)
plot_prob_winning_and_transition(learner)
plot_Q_values(learner)
plot_price_pdf(learner)
print(learner.place_bid(1))
# print(learner.place_bid(2))

# Compare learner to other agents
bidders[0].reset()
b0 = [0] * len(bidders[0].possible_types)
lb0 = [0] * len(learner.possible_types)
for t_idx, t in enumerate(learner.possible_types):
    bidders[0].valuations = [t, t / 2.0]
    learner.valuations = [t, t / 2.0]
    if t_idx == 0:
        learner.calc_expected_rewards()
    else:
        learner.calc_terminal_state_rewards()
    learner.solve_mdp()
    # b20 = learner.place_bid(2)
    # learner.num_goods_won += 1
    # b21 = learner.place_bid(2)
    # learner.num_goods_won = 0
    # print(t, learner.place_bid(1), b20, b21)
    b0[t_idx] = bidders[0].place_bid(1)
    lb0[t_idx] = learner.place_bid(1)

plt.figure()
plt.plot(possible_types, b0, label='Katzman')
plt.plot(possible_types, lb0, label='MDP')
plt.xlabel('Valuation')
plt.ylabel('Bid')
plt.legend()
plt.show()
