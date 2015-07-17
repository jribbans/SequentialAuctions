"""
Implements a bidder that learns how to bid using a Markov Decision Process described in [1].

[1] Greenwald, Amy, and Justin Boyan. "Bidding under uncertainty: Theory and experiments." Proceedings of the 20th
conference on Uncertainty in artificial intelligence. AUAI Press, 2004.
"""
from bidder.AbstractMDPBidder import AbstractMDPBidder
import math
import numpy
import scipy.integrate
import scipy.interpolate
from auction.SequentialAuction import SequentialAuction
from scipy.stats import bernoulli


class MDPBidderUAI(AbstractMDPBidder):
    """A bidder that learns how to bid using a Markov Decision Process.
    """

    def __init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc):
        """
        :param bidder_id: Integer.  A unique identifier for this given agent.
        :param num_rounds: Integer.  The number of rounds the auction this bidder is participating in will run for.
        :param num_bidders: Integer.  The total number of bidders in the auction this bidder is participating in.
        :param possible_types: List.  A list of all possible types the bidder can take.  Types are arranged in
        increasing order.
        :param type_dist: List.  Probabilities corresponding to each entry in possible_types.
        :param type_dist_disc: Boolean.  True if type_dist is describing a discrete distribution.
        """
        AbstractMDPBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
        self.num_price_samples = len(self.action_space)

    def make_state_space(self):
        """
        """
        for X in range(self.num_rounds + 1):
            for j in range(self.num_rounds + 1):
                self.state_space.append((X, j))
        for X in range(self.num_rounds + 1):
            self.terminal_states.append((X, self.num_rounds))

    def make_action_space(self):
        """
        """
        for a in self.possible_types:
            self.action_space.append(a)

    def learn_auction_parameters(self, bidders, num_mc=1000):
        """
        Learn the highest bid of n - 1 bidders and the probability of winning.

        :param bidders: List.  Bidders to learn from.
        :param num_mc: Integer.  Number of times to test an action.
        """
        exp_payment = {}
        exp_T = {}
        prob_win = {}
        win_count = {}
        sa_counter = {}
        sas_counter = {}
        highest_other_bid = {}
        for s in self.state_space:
            highest_other_bid[s] = []
            for a in self.action_space:
                exp_payment[(s, a)] = 0.0
                prob_win[(s, a)] = 0.0
                win_count[(s, a)] = 0.0
                sa_counter[(s, a)] = 0.0
                for s_ in self.state_space:
                    sas_counter[(s, a, s_)] = 0.0
                    exp_T[(s, a, s_)] = 0.0

        sa = SequentialAuction(bidders, self.num_rounds)
        for t in range(num_mc):
            # Refresh bidders
            for bidder in bidders:
                bidder.valuations = bidder.make_valuations()
                bidder.reset()
            # Run auction and learn results of nth bidder
            sa.run()
            num_won = 0
            for j in range(self.num_rounds):
                largest_bid_amongst_n_minus_1 = max(sa.bids[j][:-1])
                s = (num_won, j)
                highest_other_bid[s].append(largest_bid_amongst_n_minus_1)
                a = min(self.action_space, key=lambda x: abs(x - sa.bids[j][-1]))
                sa_counter[(s, a)] += 1
                if largest_bid_amongst_n_minus_1 < a:
                    win_count[(s, a)] += 1
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1
                    num_won += 1
                    s_ = (num_won, j + 1)
                    sas_counter[(s, a, s_)] += 1
                elif largest_bid_amongst_n_minus_1 == a:
                    num_same_bid = sum(b == a for b in sa.bids[j][:-1])
                    prob_winning_tie = num_same_bid / self.num_bidders
                    win_count[(s, a)] += prob_winning_tie
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1 * prob_winning_tie
                    won_this_round = bernoulli.rvs(prob_winning_tie)
                    num_won += won_this_round
                    s_ = (num_won, j + 1)
                    sas_counter[(s, a, s_)] += 1
                else:
                    s_ = (num_won, j + 1)
                    sas_counter[(s, a, s_)] += 1

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                if sa_counter[(s, a)] > 0:
                    exp_payment[(s, a)] /= sa_counter[(s, a)]
                elif a_idx > 0:
                    exp_payment[(s, a)] = exp_payment[(s, self.action_space[a_idx - 1])]
        self.exp_payment = exp_payment

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                for s_ in self.state_space:
                    if sa_counter[(s, a)] == 0:
                        exp_T[(s, a, s_)] = 0.0
                    else:
                        exp_T[(s, a, s_)] = sas_counter[(s, a, s_)] / sa_counter[(s, a)]
        self.exp_T = exp_T

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                if sa_counter[(s, a)] > 0:
                    prob_win[(s, a)] = win_count[(s, a)] / sa_counter[(s, a)]
                elif a_idx > 0:
                    prob_win[(s, a)] = prob_win[(s, self.action_space[a_idx - 1])]
        self.prob_win = prob_win

        for s in self.state_space:
            # Only proceed if there is something to do
            if not highest_other_bid[s]:
                continue
            highest_other_bid[s].sort()
            # Generate a histogram and obtain probability density values
            hist, bin_edges = numpy.histogram(highest_other_bid[s], self.num_price_samples, density=True)
            self.price_prediction[s] = bin_edges[:-1].tolist()
            self.price_pdf[s] = hist.tolist()
            self.price_cdf[s] = numpy.cumsum(hist * numpy.diff(bin_edges)).tolist()
            # Calculate distribution statistics for possible bids
            interp_pdf = scipy.interpolate.interp1d(self.price_prediction[s], self.price_pdf[s],
                                                    kind='slinear')
            interp_cdf = scipy.interpolate.interp1d(self.price_prediction[s], self.price_cdf[s],
                                                    kind='slinear')
            Fb = [0] * len(self.action_space)
            for a_idx, a in enumerate(self.action_space):
                if a < self.price_prediction[s][0]:
                    Fb[a_idx] = 0.0
                elif a > self.price_prediction[s][-1]:
                    Fb[a_idx] = 1.0
                else:
                    Fb[a_idx] = float(interp_cdf(a))
                if math.isnan(Fb[a_idx]):
                    Fb[a_idx] = 0.0
            # Update the transition matrix
            for a_idx, a in enumerate(self.action_space):
                win_state = (s[0] + 1, s[1] + 1)
                self.T[(s, a, win_state)] = Fb[a_idx]
                lose_state = (s[0], s[1] + 1)
                self.T[(s, a, lose_state)] = 1.0 - Fb[a_idx]

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        for s in self.price_prediction.keys():
            for a_idx, a in enumerate(self.action_space):
                r = [-p if p <= a else 0.0 for p in self.price_prediction[s]]
                to_integrate = [r[p_idx] * self.price_pdf[s][p_idx]
                                for p_idx, p in enumerate(self.price_prediction[s])]
                self.R[(s, a)] = scipy.integrate.trapz(to_integrate, self.price_prediction[s])

        self.calc_terminal_state_rewards()

    def calc_terminal_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        for X, j in self.terminal_states:
            for a in self.action_space:
                self.R[((X, j), a)] = sum(self.valuations[:X])
