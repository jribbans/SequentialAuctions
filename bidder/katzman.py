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
            # We need to compute equation 1:
            # b^*(H) = H - \frac{\int_{\underline{v}}^{H} F(x)^{2N - 1} \, dx}{F(H)^{2N - 1}}
            # Let m = 2N - 1, where there are N + 1 bidders total
            m = 2 * (self.num_bidders - 1) - 1  # 2N - 1 in equation 1 of [1]
            # To perform the integration, we need to have types from \underline{v} to H and their corresponding CDF
            # values.  First, collect all possible types that are less than or equal to the bidder's valuation,
            # and their corresponding CDF values.
            types_le_val = [self.possible_types[i] for i in range(len(self.possible_types))
                            if self.possible_types[i] <= self.valuations[0]]
            cdf_types_le_val = self.type_dist_cdf[:len(types_le_val)]
            # Add the bidder's valuation and corresponding CDF value, if necessary.  If we sampled valuations from a
            # discrete distribution, then the bidder's valuation should be in the newly constructed list.  This should
            # only need to be done when dealing with continuous type distributions.
            if self.valuations[0] != types_le_val[-1]:
                types_le_val.append(self.valuations[0])
                cdf = scipy.interpolate.interp1d(self.possible_types, self.type_dist_cdf)
                cdf_at_val = float(cdf(self.valuations[0]))
                cdf_types_le_val.append(cdf_at_val)
            # Compute F(x)^{2N - 1}
            cdf_pow_m = [cdf_types_le_val[i] ** m for i in range(len(cdf_types_le_val))]
            # Perform integration
            int_cdf_pow_m_to_v = scipy.integrate.simps(cdf_pow_m, types_le_val)
            # Calculate bid
            bid = self.valuations[0] - (int_cdf_pow_m_to_v / cdf_pow_m[-1])
        else:
            if self.num_goods_won == 0:
                bid = self.valuations[0]
            else:
                bid = self.valuations[1]
        self.bid[r] = bid
        return self.bid[r]
