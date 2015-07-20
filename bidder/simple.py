"""
Implements a bidder which places bids equal to the valuation of the good in each round.
"""

import random
import numpy
import scipy.integrate
import scipy.interpolate


class SimpleBidder:
    """A truthful bidder.
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
        possible_types_is_sorted = all(possible_types[i] <= possible_types[i + 1]
                                       for i in range(len(possible_types) - 1))
        assert possible_types_is_sorted, 'possible types list must be non-decreasing'
        self.bidder_id = bidder_id
        self.num_rounds = num_rounds
        self.num_bidders = num_bidders
        self.possible_types = possible_types
        self.type_dist = type_dist
        self.type_dist_disc = type_dist_disc
        self.num_goods_won = 0
        self.type_dist_cdf = self.calc_type_dist_cdf()
        self.type_dist_icdf = scipy.interpolate.interp1d(self.type_dist_cdf, self.possible_types)
        self.valuations = self.make_valuations()
        # Keep track of what happens in each round
        self.bid = [0] * num_rounds
        self.win = [False] * num_rounds
        self.payment = [0] * num_rounds
        self.utility = [0] * num_rounds
        self.anounced_price = [0] * num_rounds

    def reset(self):
        """
        Reset values as if this bidder has not done anything yet.
        """
        self.num_goods_won = 0
        self.bid = [0] * self.num_rounds
        self.win = [False] * self.num_rounds
        self.payment = [0] * self.num_rounds
        self.utility = [0] * self.num_rounds
        self.anounced_price = [0] * self.num_rounds

    def calc_type_dist_cdf(self):
        """
        Calculates the cumulative distribution function of the type distribution.

        The probability function and the CDF will be normalized so that the CDF end value is 1.

        :return: type_dist_cdf: List.  The CDF of the type distribution.
        """
        if self.type_dist_disc:
            type_dist_cdf = numpy.cumsum(self.type_dist).tolist()
        else:
            type_dist_cdf = [0] * len(self.possible_types)
            for idx, v in enumerate(self.type_dist):
                type_dist_cdf[idx] = float(scipy.integrate.simps(
                    self.type_dist[:idx + 1], self.possible_types[:idx + 1]))
        # The CDF of the largest valuation may not be 1.  This can happen because of numerical issues,
        # or the supplied probability function was not proper.  Here, we can try and fix these issues by normalizing
        # the probability function and its CDF by the ending CDF value.
        normalization = type_dist_cdf[-1]
        for i in range(len(self.type_dist)):
            self.type_dist[i] /= normalization
        for i in range(len(type_dist_cdf)):
            type_dist_cdf[i] = float(type_dist_cdf[i] / normalization)
        return type_dist_cdf

    def make_valuations(self):
        """
        Samples from the type distribution and assigns the bidder valuations.
        :return: valuations: List.  Valuations of each good.
        """
        if self.type_dist_disc:
            valuations = numpy.random.choice(self.possible_types, self.num_rounds, True, self.type_dist).tolist()
        else:
            rn = [random.random() for i in range(self.num_rounds)]
            valuations = [float(self.type_dist_icdf(rn[i])) for i in range(self.num_rounds)]
        return valuations

    def place_bid(self, current_round):
        """
        Places a bid.  The bid placed is equal to the valuation of the kth good won.

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        self.bid[r] = self.valuations[self.num_goods_won]
        return self.bid[r]

    def set_round_result(self, current_round, is_winner, payment, announced_price=None):
        """
        Notifies the bidder of the results of the current auction round.

        :param current_round: Integer.  The current auction round.
        :param is_winner: Boolean.  If the bidder has won, this is set to True.  Otherwise, False.
        :param payment: Float.  The payment the bidder is required to make.
        """
        r = current_round - 1
        x = 1 if is_winner else 0
        self.payment[r] = payment
        self.utility[r] = self.valuations[self.num_goods_won] * x - payment
        self.anounced_price[r] = announced_price
        if is_winner:
            self.win[r] = True
            # Do this last, since it's the current valuation we need to compute utility.
            self.num_goods_won += 1
        else:
            self.win[r] = False

    def get_types_le_val(self, val):
        """
        Get all possible types from the type list that are at most val, and their corresponding CDF and PMF/PDF values.

        :param: val.  Float.  The largest type of the list of types to return.
        :return: types_le_val.  List.  All types less than or equal to val.
        :return: cdf_types_le_val.  List.  CDF values of all types less than or equal to val.
        :return: pdf_types_le_val.  List.  PDF/PMF values of all types less than or equal to val.
        """
        types_le_val = [self.possible_types[i] for i in range(len(self.possible_types))
                        if self.possible_types[i] <= val]
        cdf_types_le_val = self.type_dist_cdf[:len(types_le_val)]
        pdf_types_le_val = self.type_dist[:len(types_le_val)]
        # Add val and corresponding CDF value, if necessary.  If we sampled valuations from a discrete distribution,
        # then the bidder's valuation should be in the newly constructed list.  This should only need to be done when
        # dealing with continuous type distributions.
        if not self.type_dist_disc and val != types_le_val[-1]:
            types_le_val.append(val)
            # CDF
            cdf = scipy.interpolate.interp1d(self.possible_types, self.type_dist_cdf)
            cdf_at_val = float(cdf(val))
            cdf_types_le_val.append(cdf_at_val)
            # PDF
            pdf = scipy.interpolate.interp1d(self.possible_types, self.type_dist)
            pdf_at_val = float(pdf(val))
            pdf_types_le_val.append(pdf_at_val)
        return types_le_val, cdf_types_le_val, pdf_types_le_val
