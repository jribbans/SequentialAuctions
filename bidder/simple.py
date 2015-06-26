"""
Implements a bidder which places bids equal to the valuation of the good in each round.
"""

import numpy


class SimpleBidder:
    """A truthful bidder.
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
        self.possible_types = possible_types
        self.type_dist = type_dist
        self.num_goods_won = 0
        self.type_dist_cdf = self.calc_type_dist_cdf()
        self.valuations = self.make_valuations()
        # Keep track of what happens in each round
        self.bid = [0] * num_rounds
        self.win = [False] * num_rounds
        self.payment = [0] * num_rounds
        self.utility = [0] * num_rounds

    def calc_type_dist_cdf(self):
        """
        Calculates the cumulative distribution function of the type distribution.

        The probability function and the CDF will be normalized so that the CDF end value is 1.

        :return: type_dist_cdf: List.  The CDF of the type distribution.
        """
        type_dist_cdf = numpy.cumsum(self.type_dist).tolist()
        # Ensure that the CDF ends at 1
        # First, normalize the density
        normalization = type_dist_cdf[-1]
        for i in range(len(self.type_dist)):
            self.type_dist[i] /= normalization
        # Normalize the CDF
        for i in range(len(type_dist_cdf)):
            type_dist_cdf[i] = float(type_dist_cdf[i] / normalization
        return type_dist_cdf

    def make_valuations(self):
        """
        Samples from the type distribution and assigns the bidder valuations.
        :return: valuations: List.  Valuations of each good.
        """
        valuations = list(numpy.random.choice(self.possible_types, self.num_rounds, True, self.type_dist))
        return valuations

    def place_bid(self, current_round):
        """
        Places a bid.  The bid placed is equal to the valuation of the kth good won.

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        self.bid[r] = self.valuations[self.num_goods_won]
        return self.bid[r]

    def set_round_result(self, current_round, is_winner, payment):
        """
        Notifies the bidder of the results of the current auction round.

        :param current_round: Integer.  The current auction round.
        :param is_winner: Boolean.  If the bidder has won, this is set to True.  Otherwise, False.
        :param payment: Float.  The payment the bidder is required to make.
        """
        r = current_round - 1
        x = 1 if is_winner else 0
        if is_winner:
            self.win[r] = True
            self.payment[r] = payment
            self.utility[r] = self.valuations[self.num_goods_won]*x - payment
            # Do this last, since it's the current valuation we need to compute utility.
            self.num_goods_won += 1
        else:
            self.win[r] = False
            self.payment[r] = payment
            self.utility[r] = self.valuations[self.num_goods_won]*x - payment
