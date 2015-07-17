"""
Implements a bidder that learns how to bid using a Markov Decision Process described in [1].

[1] Greenwald, Amy, and Justin Boyan. "Bidding under uncertainty: Theory and experiments." Proceedings of the 20th
conference on Uncertainty in artificial intelligence. AUAI Press, 2004.
"""
from bidder.mdp import MDPBidder
import numpy
import scipy.integrate
import scipy.interpolate
from scipy.stats import bernoulli
import math
import random


class MDPBidderUAI(MDPBidder):
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
        MDPBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)

    def learn_auction_parameters(self, bidders, num_mc=50000):
        """
        Learn the highest bid of n - 1 bidders and the probability of winning.

        :param bidders: List.  Bidders to learn from.
        :param num_mc: Integer.  Number of times to test an action.
        """
        sa_counter = [[[0.0 for b in range(len(self.action_space))]
                       for j in range(self.num_rounds + 1)]
                      for X in range(self.num_rounds + 1)]
        sas_counter = [[[[[0.0 for j2 in range(self.num_rounds + 1)]
                          for X2 in range(self.num_rounds + 1)]
                         for b in range(len(self.action_space))]
                        for j in range(self.num_rounds + 1)]
                       for X in range(self.num_rounds + 1)]
        exp_T = [[[[[0.0 for j2 in range(self.num_rounds + 1)]
                    for X2 in range(self.num_rounds + 1)]
                   for b in range(len(self.action_space))]
                  for j in range(self.num_rounds + 1)]
                 for X in range(self.num_rounds + 1)]
        exp_payment = [[[0.0 for b in range(len(self.action_space))]
                        for j in range(self.num_rounds + 1)]
                       for X in range(self.num_rounds + 1)]
        win_count = [[[0.0] * len(self.action_space)
                      for j in range(self.num_rounds)]
                     for X in range(self.num_rounds)]
        highest_other_bid = [[[] for j in range(self.num_rounds)]
                             for X in range(self.num_rounds)]

        # Have the agent play against N - 1 bidders
        for t in range(num_mc):
            # Refresh bidders.  We will not use the last bidder.
            for bidder in bidders[:-1]:
                bidder.valuations = bidder.make_valuations()
                bidder.reset()
            num_won = 0
            for j in range(self.num_rounds):
                bids = [bidder.place_bid(j + 1) for bidder in bidders[:-1]]
                largest_bid_amongst_n_minus_1 = max(bids)
                highest_other_bid[num_won][j].append(largest_bid_amongst_n_minus_1)
                a_idx = random.randint(0, len(self.action_space) - 1)
                a = self.action_space[a_idx]
                bids.append(a)
                sa_counter[num_won][j][a_idx] += 1
                if largest_bid_amongst_n_minus_1 < a:
                    win_count[num_won][j][a_idx] += 1
                    exp_payment[num_won][j][a_idx] -= largest_bid_amongst_n_minus_1
                    sas_counter[num_won][j][a_idx][num_won + 1][j + 1] += 1
                    num_won += 1
                    for bidder in bidders[:-1]:
                        bidder.set_round_result(j + 1, False, 0.0)
                elif largest_bid_amongst_n_minus_1 == a:
                    # Increment based on how many bidders bid the same bid
                    num_same_bid = sum(b == a for b in bids[:-1])
                    prob_winning_tie = num_same_bid / self.num_bidders
                    win_count[num_won][j][a_idx] += prob_winning_tie
                    exp_payment[num_won][j][a_idx] -= largest_bid_amongst_n_minus_1 * prob_winning_tie
                    won_this_round = bernoulli.rvs(prob_winning_tie)
                    sas_counter[num_won][j][a_idx][num_won + won_this_round][j + 1] += 1
                    num_won += won_this_round
                    if won_this_round:
                        for bidder in bidders[:-1]:
                            bidder.set_round_result(j + 1, False, 0.0)
                    else:
                        gave_object_away = False
                        for bidder in bidders[:-1]:
                            if (bidder.bid[j] == largest_bid_amongst_n_minus_1) and not gave_object_away:
                                bidder.set_round_result(j + 1, True, a)
                            else:
                                bidder.set_round_result(j + 1, False, a)
                else:
                    sas_counter[num_won][j][a_idx][num_won][j + 1] += 1
                    for bidder in bidders[:-1]:
                        if bidder.bid[j] == largest_bid_amongst_n_minus_1:
                            bidder.set_round_result(j + 1, True, sorted(bids)[-2])
                        else:
                            bidder.set_round_result(j + 1, False, 0)

        # Calculate expected payment
        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                for a in range(len(self.action_space)):
                    if sa_counter[X][j][a] > 0:
                        exp_payment[X][j][a] /= sa_counter[X][j][a]
        self.exp_payment = exp_payment

        # Calculate expected transition probabilities
        for X in range(self.num_rounds + 1):
            for j in range(self.num_rounds + 1):
                for b_idx, b in enumerate(self.action_space):
                    if sa_counter[X][j][b_idx] == 0:
                        continue
                    for X2 in range(self.num_rounds + 1):
                        for j2 in range(self.num_rounds + 1):
                            exp_T[X][j][b_idx][X2][j2] = sas_counter[X][j][b_idx][X2][j2] \
                                                         / sa_counter[X][j][b_idx]

        # Calculate the probability of winning
        prob_win = [[[win_count[X][j][a] / sa_counter[X][j][a] if sa_counter[X][j][a] != 0 else 0.0
                      for a in range(len(self.action_space))]
                     for j in range(self.num_rounds)]
                    for X in range(self.num_rounds)]
        self.prob_winning = prob_win

        # Calculate transition probabilities and predicted prices
        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                # Only proceed if there is something to do
                if not highest_other_bid[X][j]:
                    continue
                highest_other_bid[X][j].sort()
                # Generate a histogram and obtain probability density values
                hist, bin_edges = numpy.histogram(highest_other_bid[X][j], self.num_price_samples, density=True)
                self.price_prediction[X][j] = bin_edges[:-1].tolist()
                # self.price_prediction[X][j] = [float(bin_edges[i] + bin_edges[i + 1]) * .5
                #                               for i in range(len(bin_edges) - 1)]
                self.price_pdf[X][j] = hist.tolist()
                self.price_cdf[X][j] = numpy.cumsum(hist * numpy.diff(bin_edges)).tolist()
                # Calculate distribution statistics for possible bids
                interp_pdf = scipy.interpolate.interp1d(self.price_prediction[X][j], self.price_pdf[X][j],
                                                        kind='slinear')
                interp_cdf = scipy.interpolate.interp1d(self.price_prediction[X][j], self.price_cdf[X][j],
                                                        kind='slinear')
                Fb = [0] * len(self.action_space)
                for b_idx, b in enumerate(self.action_space):
                    if b < self.price_prediction[X][j][0]:
                        Fb[b_idx] = 0.0
                    elif b > self.price_prediction[X][j][-1]:
                        Fb[b_idx] = 1.0
                    else:
                        Fb[b_idx] = float(interp_cdf(b))
                    if math.isnan(Fb[b_idx]):
                        Fb[b_idx] = 0.0
                # Update the transition matrix
                for b_idx, b in enumerate(self.action_space):
                    self.T[X][j][b_idx][X + 1][j + 1] = Fb[b_idx]
                    self.T[X][j][b_idx][X][j + 1] = 1.0 - Fb[b_idx]

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        # For states not corresponding to the end of an auction:
        # R((X, j-1), b, p) = \int r((X, j-1), b, p) f(p) dp
        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                for b_idx, b in enumerate(self.action_space):
                    # For now, when there is a tie, assume this bidder wins.
                    r = [-p if p <= b else 0.0 for p_idx, p in enumerate(self.price_prediction[X][j])]
                    to_integrate = [r[p_idx] * self.price_pdf[X][j][p_idx]
                                    for p_idx, p in enumerate(self.price_prediction[X][j])]
                    self.R[X][j][b_idx] = scipy.integrate.trapz(to_integrate, self.price_prediction[X][j])

        self.calc_end_state_rewards()

    def calc_end_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        # R((X, n)) = v(X)
        for X in range(self.num_rounds + 1):
            self.R[X][self.num_rounds] = [sum(self.valuations[:X])] * len(self.action_space)
