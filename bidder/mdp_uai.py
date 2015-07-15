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

    def learn_auction_parameters(self, bidders, num_trials_per_action=100):
        """
        Learn the highest bid of n - 1 bidders and the probability of winning.

        :param bidders: List.  Bidders to learn from.
        :param num_trials_per_action: Integer.  Number of times to test an action.
        """
        # exp_payment[X][j]
        sa_counter = [[[0 for b in range(len(self.action_space))]
                       for j in range(self.num_rounds + 1)]
                      for X in range(self.num_rounds + 1)]
        exp_payment = [[[0 for b in range(len(self.action_space))]
                        for j in range(self.num_rounds + 1)]
                       for X in range(self.num_rounds + 1)]
        win_count = {r: [0] * len(self.action_space) for r in range(self.num_rounds)}
        highest_other_bid = {r: [] for r in range(self.num_rounds)}
        sa = SequentialAuction(bidders, self.num_rounds)
        for a_idx, a in enumerate(self.action_space):
            for t in range(num_trials_per_action):
                # Have bidders sample new valuations
                for bidder in bidders:
                    bidder.valuations = bidder.make_valuations()
                    bidder.reset()
                # Run an auction
                sa.run()
                # See if the action we are using leads to a win
                num_won = 0
                for r in range(self.num_rounds):
                    sa_counter[num_won][r][a_idx] += 1
                    if max(sa.bids[r][:-1]) < a:
                        win_count[r][a_idx] += 1
                        exp_payment[num_won][r][a_idx] -= max(sa.bids[r][:-1])
                        num_won += 1
                    elif max(sa.bids[r][:-1]) == a:
                        # Increment based on how many bidders bid the same bid
                        num_same_bid = sum(b == a for b in sa.bids[r][:-1])
                        prob_winning_tie = num_same_bid / self.num_bidders
                        win_count[r][a_idx] += prob_winning_tie
                        exp_payment[num_won][r][a_idx] -= max(sa.bids[r][:-1]) * prob_winning_tie
                        num_won += bernoulli.rvs(prob_winning_tie)
                    highest_other_bid[r].append(max(sa.bids[r][:-1]))

        for X in range(self.num_rounds):
            for j in range(self.num_rounds):
                for a in range(len(self.action_space)):
                    if sa_counter[X][j][a] > 0:
                        exp_payment[X][j][a] /= sa_counter[X][j][a]
        self.exp_payment = exp_payment

        prob_win = [[win_count[r][i] / num_trials_per_action
                     for i in range(len(self.possible_types))]
                    for r in range(self.num_rounds)]
        self.prob_winning = prob_win

        # Calculate the CDF and PDF of the price distribution by building a histogram
        interp_cdf = []
        interp_pdf = []
        for r in range(self.num_rounds):
            highest_other_bid[r].sort()
            # If we are dealing with discrete bidder types, the space of possible bids and prices is finite,
            # so just use what we see as bin edges.  Otherwise, divide the range of bids into bins
            if self.type_dist_disc:
                unique_bids = list(set(highest_other_bid[r]))
                unique_bids.sort()
                bin_edges = unique_bids
            else:
                bid_range = highest_other_bid[r][-1] - highest_other_bid[r][0]
                bin_width = bid_range / (self.num_price_samples + 2)
                bin_edges = [bin_width * i for i in range(1, self.num_price_samples + 3)]
                # Avoid numerical issues.  The last bin edge should be exactly the largest bid observed.
                bin_edges[-1] = highest_other_bid[r][-1]

            # Bin the values
            hist = {e: [] for e in bin_edges}
            idx = 0
            for b in highest_other_bid[r]:
                if bin_edges[idx] >= b:
                    hist[bin_edges[idx]].append(b)
                else:
                    while bin_edges[idx] < b:
                        idx += 1
                    hist[bin_edges[idx]].append(b)

            # Calculate the average bid in each bin and the CDF
            mean_binned_val = [0] * len(bin_edges)
            emp_cdf = [0] * len(bin_edges)
            count = [len(hist[bin_edges[i]]) for i in range(len(bin_edges))]
            for e_idx, e in enumerate(bin_edges):
                if len(hist[e]) > 0:
                    mean_binned_val[e_idx] = sum(hist[e]) / len(hist[e])
                emp_cdf[e_idx] = sum(count[:e_idx + 1])
            # Divide the count so that we have a proper CDF.  The last value should be 1.
            for e_idx in range(len(bin_edges)):
                emp_cdf[e_idx] /= emp_cdf[-1]

            # Calculate the PDF of all points except for the first and last.
            pdf = [0] * len(bin_edges)
            for i in range(1, len(pdf) - 1):
                p = numpy.polyfit(mean_binned_val[i - 1:i + 2], emp_cdf[i - 1:i + 2], 1).tolist()
                pdf[i] = p[0]

            # The price points we store will be all the values except for the first and last.
            sampled_prices = mean_binned_val[1:-1]
            interp_pdf.append(scipy.interpolate.interp1d(sampled_prices, pdf[1:-1]))
            interp_cdf.append(scipy.interpolate.interp1d(sampled_prices, emp_cdf[1:-1]))
            self.price_prediction[r] = sampled_prices
            self.price_pdf[r] = pdf[1:-1]
            self.price_cdf[r] = emp_cdf[1:-1]

        # Calculate the probability of exceeding a predicted price for each action this bidder can perform.
        Fb = [[0] * len(self.action_space) for r in range(self.num_rounds)]
        for r in range(self.num_rounds):
            for b_idx, b in enumerate(self.action_space):
                if b < min(self.price_prediction[r]):
                    Fb[r][b_idx] = 0
                elif b > max(self.price_prediction[r]):
                    Fb[r][b_idx] = 1.0
                else:
                    Fb[r][b_idx] = float(interp_cdf[r](b))
        self.price_cdf_at_bid = Fb

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        # For states not corresponding to the end of an auction:
        # R((X, j-1), b, p) = \int r((X, j-1), b, p) f(p) dp
        for j in range(self.num_rounds):
            for b_idx, b in enumerate(self.action_space):
                # For now, when there is a tie, assume this bidder wins.
                r = [-p if p <= b else 0.0 for p_idx, p in enumerate(self.price_prediction[j])]
                to_integrate = [r[p_idx] * self.price_pdf[j][p_idx]
                                for p_idx, p in enumerate(self.price_prediction[j])]
                for X in range(self.num_rounds + 1):
                    self.R[X][j][b_idx] = scipy.integrate.trapz(to_integrate, self.price_prediction[j])

        self.calc_end_state_rewards()

    def calc_end_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        # R((X, n)) = v(X)
        for X in range(self.num_rounds + 1):
            self.R[X][self.num_rounds] = [sum(self.valuations[:X])] * len(self.action_space)
