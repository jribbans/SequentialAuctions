"""
Implements a bidder that learns how to bid using a Markov Decision Process described in [1].

[1] Greenwald, Amy, and Justin Boyan. "Bidding under uncertainty: Theory and experiments." Proceedings of the 20th
conference on Uncertainty in artificial intelligence. AUAI Press, 2004.
"""
from bidder.mdp import MDPBidder
from auction.SequentialAuction import SequentialAuction
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
        sa_counter = [[[0 for b in range(len(self.action_space))]
                       for j in range(self.num_rounds + 1)]
                      for X in range(self.num_rounds + 1)]
        sas_counter = [[[[[0 for j2 in range(self.num_rounds + 1)]
                          for X2 in range(self.num_rounds + 1)]
                         for b in range(len(self.action_space))]
                        for j in range(self.num_rounds + 1)]
                       for X in range(self.num_rounds + 1)]
        exp_T = [[[[[0 for j2 in range(self.num_rounds + 1)]
                    for X2 in range(self.num_rounds + 1)]
                   for b in range(len(self.action_space))]
                  for j in range(self.num_rounds + 1)]
                 for X in range(self.num_rounds + 1)]
        exp_payment = [[[0 for b in range(len(self.action_space))]
                        for j in range(self.num_rounds + 1)]
                       for X in range(self.num_rounds + 1)]
        win_count = [[[0] * len(self.action_space)
                      for j in range(self.num_rounds)]
                     for X in range(self.num_rounds)]
        highest_other_bid = [[[] for j in range(self.num_rounds)]
                             for X in range(self.num_rounds)]

        sa = SequentialAuction(bidders, self.num_rounds)
        for t in range(num_mc):
            # Refresh bidders
            for bidder in bidders:
                bidder.valuations = bidder.make_valuations()
                bidder.reset()
            # Run an auction and see how randomly sampled actions would perform against n - 1 bidders
            sa.run()
            num_won = 0
            for j in range(self.num_rounds):
                largest_bid_amongst_n_minus_1 = max(sa.bids[j][:-1])
                highest_other_bid[num_won][j].append(largest_bid_amongst_n_minus_1)
                a_idx = random.randint(0, len(self.action_space) - 1)
                a = self.action_space[a_idx]
                sa_counter[num_won][j][a_idx] += 1
                if largest_bid_amongst_n_minus_1 < a:
                    win_count[num_won][j][a_idx] += 1
                    exp_payment[num_won][j][a_idx] -= largest_bid_amongst_n_minus_1
                    sas_counter[num_won][j][a_idx][num_won + 1][j + 1] += 1
                    num_won += 1
                elif largest_bid_amongst_n_minus_1 == a:
                    # Increment based on how many bidders bid the same bid
                    num_same_bid = sum(b == a for b in sa.bids[j][:-1])
                    prob_winning_tie = num_same_bid / self.num_bidders
                    win_count[num_won][j][a_idx] += prob_winning_tie
                    exp_payment[num_won][j][a_idx] -= largest_bid_amongst_n_minus_1 * prob_winning_tie
                    won_this_round = bernoulli.rvs(prob_winning_tie)
                    sas_counter[num_won][j][a_idx][num_won + won_this_round][j + 1] += 1
                    num_won += won_this_round
                else:
                    sas_counter[num_won][j][a_idx][num_won][j + 1] += 1

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

        """
        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                for a in range(len(self.action_space)):
                    print(X, j, a, sa_counter[X][j][a], prob_win[X][j][a], exp_T[X][j][a][X+1][j+1])
        """

        # Calculate transition probabilities and predicted prices
        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                # Only proceed if there is something to do
                if not highest_other_bid[X][j]:
                    continue
                highest_other_bid[X][j].sort()
                # Build a histogram of highest other bids
                # Make bin edges
                if self.type_dist_disc:
                    # In the discrete case, bin edges will be the bids we have seen
                    unique_bids = list(set(highest_other_bid[X][j]))
                    unique_bids.sort()
                    bin_edges = unique_bids
                else:
                    # In the continous case, bins are made by dividing the entire range of bids seen
                    bid_range = highest_other_bid[X][j][-1] - highest_other_bid[X][j][0]
                    bin_width = bid_range / (self.num_price_samples + 2)
                    bin_edges = [bin_width * i for i in range(1, self.num_price_samples + 3)]
                    # Avoid numerical issues.  The last bin edge should be exactly the largest bid observed.
                    bin_edges[-1] = highest_other_bid[X][j][-1]
                # Bin bids
                hist = {e: [] for e in bin_edges}
                idx = 0
                for b in highest_other_bid[X][j]:
                    if bin_edges[idx] >= b:
                        hist[bin_edges[idx]].append(b)
                    else:
                        while bin_edges[idx] < b:
                            idx += 1
                        hist[bin_edges[idx]].append(b)
                # Calculate the average bid in each bin and the CDF
                mean_binned_val = [sum(hist[e]) / len(hist[e]) if len(hist[e]) > 0 else 0.0
                                   for e in bin_edges]
                # Calculate the empirical CDF at each bin
                num_entries_per_bin = [len(hist[bin_edges[i]]) for i in range(len(bin_edges))]
                cdf = [sum(num_entries_per_bin[:e_idx + 1])
                       for e_idx in range(len(bin_edges))]
                # Normalize
                cdf = [cdf[e_idx] / cdf[-1]
                       for e_idx in range(len(bin_edges))]
                # Calculate the PDF of all points except for the first and last.
                pdf = [0] * len(bin_edges)
                for i in range(1, len(pdf) - 1):
                    # Calculate the slope using points i - 1, i and i + 1
                    p = numpy.polyfit(mean_binned_val[i - 1:i + 2], cdf[i - 1:i + 2], 1).tolist()
                    if math.isnan(p[0]):
                        dy = cdf[i] - cdf[i - 1]
                        dx = mean_binned_val[i] - mean_binned_val[i - 1]
                        if (dx <= 1e-5):
                            pdf[i] = 0
                        else:
                            pdf[i] = dy / dx
                    else:
                        pdf[i] = p[0]
                # The sampled prices are what we will store as price predictions
                sampled_prices = mean_binned_val[1:-1]
                self.price_prediction[X][j] = sampled_prices
                # Store distribution information about prices
                self.price_pdf[X][j] = pdf[1:-1]
                self.price_cdf[X][j] = cdf[1:-1]
                # Calculate the probability of exceeding a predicted price for each action this bidder can perform.
                interp_pdf = scipy.interpolate.interp1d(sampled_prices, pdf[1:-1])
                interp_cdf = scipy.interpolate.interp1d(sampled_prices, cdf[1:-1])
                Fb = [0] * len(self.action_space)
                for b_idx, b in enumerate(self.action_space):
                    if b < sampled_prices[0]:
                        Fb[b_idx] = 0
                    elif b > sampled_prices[-1]:
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
