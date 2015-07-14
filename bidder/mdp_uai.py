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
        win_count = {r: [0] * len(self.action_space) for r in range(self.num_rounds)}
        prices = {r: [] for r in range(self.num_rounds)}
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
                for r in range(self.num_rounds):
                    if max(sa.bids[r][:-1]) < a:
                        win_count[r][a_idx] += 1
                    elif max(sa.bids[r][:-1]) == a:
                        # Increment based on how many bidders bid the same bid
                        num_same_bid = sum(b == a for b in sa.bids[r][:-1])
                        win_count[r][a_idx] += num_same_bid / self.num_bidders
                    prices[r].append(max(sa.bids[r][:-1]))

        prob_win = [[win_count[r][i] / num_trials_per_action
                     for i in range(len(self.possible_types))]
                    for r in range(self.num_rounds)]
        self.prob_winning = prob_win

        interp_cdf = []
        for r in range(self.num_rounds):
            prices[r].sort()
            # Let N = num price pts - 1
            # cdf = 0/N, 1/N, ..., N/N
            cdf_of_prices = [i / (len(prices[r]) - 1) for i in range(len(prices[r]))]
            interp_cdf.append(scipy.interpolate.interp1d(prices[r], cdf_of_prices))
            # Sample from the N pts
            sampled_prices = numpy.linspace(prices[r][0], prices[r][-2], self.num_price_samples).tolist()
            # PDF = rise / run of CDF
            pdf = []
            cdf = []
            for i in range(self.num_price_samples - 1):
                rise = interp_cdf[r](sampled_prices[i + 1]) - interp_cdf[r](sampled_prices[i])
                run = sampled_prices[i + 1] - sampled_prices[i]
                pdf_val = float(rise / run)
                pdf.append(pdf_val)
                cdf.append(float(interp_cdf[r](sampled_prices[i])))
            # last point.  Go back a few points to avoid unusually large numbers
            rise = 1 - interp_cdf[r](sampled_prices[-3])
            run = prices[r][-1] - sampled_prices[-3]
            pdf_val = float(rise / run)
            pdf.append(pdf_val)
            cdf.append(float(interp_cdf[r](sampled_prices[-1])))
            self.price[r] = sampled_prices
            self.price_dist[r] = pdf
            self.price_cdf[r] = cdf

        Fb = [[0] * len(self.action_space) for r in range(self.num_rounds)]
        for r in range(self.num_rounds):
            for b_idx, b in enumerate(self.action_space):
                if b < min(self.price[r]):
                    Fb[r][b_idx] = 0
                elif b > max(self.price[r]):
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
                r = [-p if p <= b else 0.0 for p_idx, p in enumerate(self.price[j])]
                to_integrate = [r[p_idx] * self.price_dist[j][p_idx]
                                for p_idx, p in enumerate(self.price[j])]
                for X in range(self.num_rounds + 1):
                    self.R[X][j][b_idx] = scipy.integrate.trapz(to_integrate, self.price[j])

        self.calc_end_state_rewards()

    def calc_end_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        # R((X, n)) = v(X)
        for X in range(self.num_rounds + 1):
            self.R[X][self.num_rounds] = [sum(self.valuations[:X])] * len(self.action_space)
