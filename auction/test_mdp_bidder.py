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
    for X in range(mdp_bidder.num_rounds):
        for j in range(mdp_bidder.num_rounds):
            if X <= j:
                ls = 'R: X=' + str(X) + ' j=' + str(j)
                plt.plot(mdp_bidder.action_space, mdp_bidder.R[X][j], label=ls)
                ls = 'EP: X=' + str(X) + ' j=' + str(j)
                plt.plot(mdp_bidder.action_space, mdp_bidder.exp_payment[X][j], label=ls)
    plt.xlabel('Bid')
    plt.ylabel('Expected Payment')
    plt.title('Calculated and Empirical Expected Payment')
    plt.legend()
    plt.show()


def plot_transition(mdp_bidder):
    plt.figure()
    for X in range(mdp_bidder.num_rounds + 1):
        for j in range(mdp_bidder.num_rounds + 1):
            for X2 in range(mdp_bidder.num_rounds + 1):
                for j2 in range(mdp_bidder.num_rounds + 1):
                    T = [0] * len(mdp_bidder.action_space)
                    for a in range(len(mdp_bidder.action_space)):
                        T[a] = mdp_bidder.T[X][j][a][X2][j2]
                    if any(T[a] > 0 for a in range(len(mdp_bidder.action_space))):
                        ls = 'T: X=' + str(X) + ' j=' + str(j) + ' X2=' + str(X2) + ' j2=' + str(j2)
                        plt.plot(mdp_bidder.action_space, T, label=ls)
    plt.xlabel('Bid')
    plt.ylabel('Probability')
    plt.title('Probability of Transitioning to Next State')
    plt.legend()
    plt.show()


def plot_prob_winning_and_transition(mdp_bidder):
    plt.figure()
    for X in range(mdp_bidder.num_rounds + 1):
        for j in range(mdp_bidder.num_rounds + 1):
            for X2 in range(mdp_bidder.num_rounds + 1):
                for j2 in range(mdp_bidder.num_rounds + 1):
                    T = [0] * len(mdp_bidder.action_space)
                    for a in range(len(mdp_bidder.action_space)):
                        T[a] = mdp_bidder.T[X][j][a][X2][j2]
                    if X2 <= X:
                        continue
                    if any(T[a] > 0 for a in range(len(mdp_bidder.action_space))):
                        ls = 'T: X=' + str(X) + ' j=' + str(j) + ' X2=' + str(X2) + ' j2=' + str(j2)
                        plt.plot(mdp_bidder.action_space, T, label=ls)
                        ls = 'PW: X=' + str(X) + ' j=' + str(j)
                        plt.plot(mdp_bidder.action_space, mdp_bidder.prob_winning[X][j], label=ls, ls=':')
    plt.xlabel('Bid')
    plt.ylabel('Probability')
    plt.title('Probability of Winning and Transitions')
    plt.legend()
    plt.show()


def plot_Q_values(mdp_bidder):
    plt.figure()
    for X in range(mdp_bidder.num_rounds + 1):
        for j in range(mdp_bidder.num_rounds + 1):
            if X <= j:
                ls = 'X=' + str(X) + ' j=' + str(j)
                plt.plot(mdp_bidder.action_space, mdp_bidder.Q[X][j], label=ls)
    plt.xlabel('Bid')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.show()


def plot_price_pdf(mdp_bidder):
    plt.figure()
    for X in range(mdp_bidder.num_rounds):
        for j in range(mdp_bidder.num_rounds):
            if X <= j:
                ls = 'X=' + str(X) + ' j=' + str(j)
                plt.plot(mdp_bidder.price_prediction[X][j], mdp_bidder.price_pdf[X][j], label=ls)
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

num_mc = 100000

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
        learner.calc_end_state_rewards()
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
