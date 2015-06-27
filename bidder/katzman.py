"""
Implements a sequential auction bidder using the analysis found in [1].

[1] Katzman, Brett. "A two stage sequential auction with multi-unit demands."
Journal of Economic Theory 86.1 (1999): 77-99.
"""

from bidder.simple import SimpleBidder
import numpy
import random
import scipy.integrate
import scipy.interpolate


class KatzmanBidder(SimpleBidder):
    """A bidder that bids using the strategy described in [1].
    """

    def __init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc):
        """
        :param bidder_id: Integer.  A unique identifier for this given agent.
        :param num_rounds: Integer.  The number of rounds the auction this bidder is participating in will run for.
        :param num_bidders: Integer.  The total number of bidders in the auction this bidder is participating in.
        :param possible_types: List.  A list of all possible types the bidder can take.  Types are arranged in
        increasing order.
        :param type_dist: List.  Probabilities corresponding to each entry in possible_types.
        """
        SimpleBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
        self.valuations = self.make_valuations()

    def make_valuations(self):
        """
        Samples from the type distribution and assigns the bidder valuations.
        :return: valuations: List.  Valuations of each good.  All values are strictly decreasing and unique.
        """
        if self.type_dist_disc:
            valuations = numpy.random.choice(self.possible_types, self.num_rounds, True, self.type_dist).tolist()
        else:
            while True:
                rn = [random.random() for i in range(self.num_rounds)]
                valuations = [float(self.type_dist_icdf(rn[i])) for i in range(self.num_rounds)]
                if len(set(valuations)) == self.num_rounds:
                    break
        valuations.sort(reverse=True)
        return valuations

    def place_bid(self, current_round):
        """
        Places a bid.

        In round 1, the bid placed is described in equation 1 of [1].
        In round 2, sincere bidding takes place.

        With respect to equation 1 of [1]:
        H = valuations[0]
        L = valuations[1]

        :param current_round: Integer.  The current auction round.
        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        if current_round == 1:
            m = 2 * (self.num_bidders - 1) - 1  # 2N - 1 in equation 1 of [1]
            if self.type_dist_disc:
                idx = self.possible_types.index(self.valuations[0])
                cdf_pow_m = [self.type_dist_cdf[i] ** m for i in range(idx + 1)]
                int_cdf_pow_m_to_v = scipy.integrate.simps(cdf_pow_m, self.possible_types[:idx + 1])
                bid = self.valuations[0] - (int_cdf_pow_m_to_v / cdf_pow_m[-1])
            else:
                # Closest element in self.possible_types to the bidder's valuation
                idx, closest_val = min(enumerate(self.possible_types),
                                       key=lambda x: abs(self.possible_types[1] - self.valuations[0]))
                # Calculate the CDF of the bidder's valuation
                cdf = scipy.interpolate.interp1d(self.possible_types, self.type_dist_cdf)
                cdf_at_val = float(cdf(self.valuations[0]))
                # The x (types) and y (CDF) we need to perform the integration
                types_le_val = self.possible_types[:idx + 1]
                types_le_val.append(self.valuations[0])
                cdf_types_le_val = self.type_dist_cdf[:idx + 1]
                cdf_types_le_val.append(cdf_at_val)
                cdf_pow_m = [cdf_types_le_val[i] ** m for i in range(len(cdf_types_le_val))]
                int_cdf_pow_m_to_v = scipy.integrate.simps(cdf_pow_m, types_le_val)
                bid = self.valuations[0] - (int_cdf_pow_m_to_v / cdf_pow_m[-1])

        else:
            if self.num_goods_won == 0:
                bid = self.valuations[0]
            else:
                bid = self.valuations[1]
        self.bid[r] = bid
        return self.bid[r]
