"""
Implements a sequential auction bidder using the analysis found in [1].

[1] Katzman, Brett. "A two stage sequential auction with multi-unit demands."
Journal of Economic Theory 86.1 (1999): 77-99.
"""

import numpy
import scipy.integrate


class KatzmanBidder:
    """A bidder that bids using the strategy described in [1].
    """

    def __init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist):
        """
        :param bidder_id: Integer.  A unique identifier for this given agent.
        :param num_rounds: Integer.  The number of rounds the auction this bidder is participating in will run for.
        :param num_bidders: Integer.  The total number of bidders in the auction this bidder is participating in.
        :param possible_types: List.  A list of all possible types the bidder can take.  Types are arranged in
        increasing order.
        :param type_dist: List.  Probabilities corresponding to each entry in possible_types.
        """
        self.bidder_id = bidder_id
        self.num_rounds = num_rounds
        self.num_bidders = num_bidders
        self.num_goods_won = 0
        self.possible_types = possible_types
        self.type_dist = type_dist
        self.type_dist_cdf = self.calc_type_dist_cdf()
        self.valuations = self.make_valuations()

    def calc_type_dist_cdf(self):
        """
        Calculates the cumulative distribution function of the type distribution
        :return: type_dist_cdf: List.  The CDF of the type distribution.
        """
        type_dist_cdf = list(numpy.cumsum(self.type_dist))
        return type_dist_cdf

    def make_valuations(self):
        """
        Samples from the type distribution and assigns the bidder valuations.
        :return: valuations: List.  Valuations of each good.  All values are strictly decreasing and unique.
        """
        valuations = list(numpy.random.choice(self.possible_types, self.num_rounds, False, self.type_dist))
        valuations.sort(key=lambda x: x, reverse=True)
        return valuations

    def place_bid(self, current_round):
        """
        Places a bid.

        In round 1, the bid placed is described in equation 1 of [1].
        In round 2, sincere bidding takes place.

        :param current_round: Integer.  The current auction round.
        :return: bid: Float.  The bid the bidder will place.
        """
        if current_round == 1:
            m = 2 * (self.num_bidders - 1) - 1
            possible_val_idx = self.possible_types.index(self.valuations[0])
            cdf_pow_m = [self.type_dist_cdf[i] ** m for i in range(possible_val_idx + 1)]
            int_cdf_pow_m_to_v = scipy.integrate.simps(cdf_pow_m, self.possible_types[:possible_val_idx + 1])
            bid = self.valuations[0] - (int_cdf_pow_m_to_v / cdf_pow_m[-1])
        else:
            if self.num_goods_won == 0:
                bid = self.valuations[0]
            else:
                bid = self.valuations[1]
        return bid
