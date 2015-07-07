"""
Implements a sequential auction bidder using the analysis found in [1], in the case of sequential second-price
auctions where one good is being assinged to one bidder in each round.
[1] Weber, Robert J. "MULTIPLE—OBJECT AUCTIONS." (1981).
"""
from bidder.simple import SimpleBidder
from math import factorial
from math import isnan
import scipy.integrate
import scipy.interpolate
import scipy.misc
from utility.order_statistics import *
from copy import deepcopy


class WeberBidder(SimpleBidder):
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
        :param type_dist_disc: Boolean.  True if type_dist is describing a discrete distribution.
        """
        assert not type_dist_disc, "Only continuous distributions are supported."
        SimpleBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)

    def place_bid(self, current_round):
        """
        Places a bid.  See Theorem 3, part (b) of [1].

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        num_rounds_remaining = self.num_rounds - current_round

        if self.num_goods_won > 0:
            bid = 0.0
        else:
            # There are k rounds total, and we are at round \ell.  Y_k will be the k - \ell highest bid
            # Example: there are 2 rounds, and we are at round 1.  We want to know what Y_2 is, if Y_1 = x.
            # This will be the first highest bid after x.
            if self.type_dist_disc:
                pass
            else:
                if current_round == self.num_rounds:
                    bid = self.valuations[0]
                else:
                    # Define parameters to compute order statistic distribution values
                    n = (self.num_bidders - 1)  # Weber: n - 1 values other than bidders own
                    j = n - self.num_rounds + 1  # Weber: k
                    k = n - current_round + 1  # Weber: \ell
                    y = self.valuations[0]

                    # f_{Y_k \mid Y_{\ell}} (z \mid x) = f_{Y_k, Y_{\ell}} (z, x) / f_{Y_{\ell}(x)}
                    cond_dist = [0] * len(self.possible_types)
                    for i in range(len(self.possible_types)):
                        if self.possible_types[i] < y:
                            numerator = calc_joint_dist_val_jk_os(self.possible_types,
                                                                  self.type_dist,
                                                                  self.type_dist_cdf,
                                                                  self.type_dist_disc,
                                                                  j, k, n, self.possible_types[i], y)
                            denominator = calc_dist_val_k_os(self.possible_types, self.type_dist, self.type_dist_cdf,
                                                             self.type_dist_disc, k, n, y)
                            """
                            numerator = calc_joint_dist_val_jk_os_u01(self.type_dist_disc,
                                                                  j, k, n, self.possible_types[i], y)
                            denominator = calc_dist_val_k_os_u01(self.type_dist_disc, k, n, y)
                            """
                            cond_dist[i] = numerator / denominator
                        else:
                            cond_dist[i] = 0.0
                    # z * f_{Y_k \mid Y_{\ell}} (z \mid x)
                    to_integrate = [0] * len(self.possible_types)
                    for i in range(len(self.possible_types)):
                        to_integrate[i] = self.possible_types[i] * cond_dist[i]
                    # Trapezoidal rule seems to do better for the uniform distribution
                    bid = scipy.integrate.trapz(to_integrate, self.possible_types)

        self.bid[r] = bid
        return self.bid[r]
