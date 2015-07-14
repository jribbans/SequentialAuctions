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
                for r in range(self.num_rounds):
                    if max(sa.bids[r][:-1]) < a:
                        win_count[r][a_idx] += 1
                    elif max(sa.bids[r][:-1]) == a:
                        # Increment based on how many bidders bid the same bid
                        num_same_bid = sum(b == a for b in sa.bids[r][:-1])
                        win_count[r][a_idx] += num_same_bid / self.num_bidders
                    highest_other_bid[r].append(max(sa.bids[r][:-1]))

        prob_win = [[win_count[r][i] / num_trials_per_action
                     for i in range(len(self.possible_types))]
                    for r in range(self.num_rounds)]
        self.prob_winning = prob_win

        # Calculate the distribution of predicted prices
        # We have a total of N pts, 0 to N-1.  Calculate the distribution for pts 1 to N-2.
        # Since N is large, we shouldn't be losing too much doing this.
        interp_cdf = []
        interp_pdf = []
        for r in range(self.num_rounds):
            highest_other_bid[r].sort()
            # 0 / (N-1), 1 / (N-1), ..., (N-1)/(N-1)
            cdf_of_prices = [i / (len(highest_other_bid[r]) - 1) for i in range(len(highest_other_bid[r]))]
            # Compute PDF for all but the end points
            pdf_of_prices = [0] * len(highest_other_bid[r])
            for i in range(1, len(pdf_of_prices) - 1):
                # PDF is the derivative of CDF, so the slope of a line fit in a region of the CDF = PDF
                # Degree 1 polynomial fit: p[0] x + p[1]
                p = numpy.polyfit(highest_other_bid[r][i - 1:i + 2], cdf_of_prices[i - 1:i + 2], 1).tolist()
                pdf_of_prices[i] = p[0]
                # Simpler method: find the slope between points i-1 and i+1
                # rise = cdf_of_prices[i + 1] - cdf_of_prices[i - 1]
                # run = highest_other_bid[r][i + 1] - highest_other_bid[r][i - 1]
                # pdf_of_prices[i] = rise / run
            interp_pdf.append(scipy.interpolate.interp1d(highest_other_bid[r][1:-1], pdf_of_prices[1:-1]))
            interp_cdf.append(scipy.interpolate.interp1d(highest_other_bid[r][1:-1], cdf_of_prices[1:-1]))
            sampled_prices = numpy.linspace(highest_other_bid[r][1], highest_other_bid[r][-2],
                                            self.num_price_samples).tolist()
            self.price_prediction[r] = sampled_prices
            self.price_pdf[r] = interp_pdf[r](sampled_prices).tolist()
            self.price_cdf[r] = interp_cdf[r](sampled_prices).tolist()

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
