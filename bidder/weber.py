"""
Implements a sequential auction bidder using the analysis found in [1], in the case of sequential second-price
auctions where one good is being assinged to one bidder in each round.
[1] Weber, Robert J. "MULTIPLE—OBJECT AUCTIONS." (1981).
"""
from bidder.simple import SimpleBidder
import scipy.integrate
import scipy.interpolate
import scipy.misc


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
        """
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
                # b_{\ell}^S(x) = E[Y_k \mid Y_{\ell} = x]
                types_le_val, cdf_types_le_val, pdf_types_le_val = self.get_types_le_val(self.valuations[0])
                # Remove the element that corresponds to Y_{\ell} = x
                types_le_val.pop()
                cdf_types_le_val.pop()
                pdf_types_le_val.pop()
                # Retain what we need for the kth round
                del types_le_val[-num_rounds_remaining:]
                del cdf_types_le_val[-num_rounds_remaining:]
                del pdf_types_le_val[-num_rounds_remaining:]
                num_types_left = len(types_le_val)
                if num_types_left == 0:
                    bid = 0.0
                else:
                    # Compute f(z | Y_l = x)
                    normalization = sum(pdf_types_le_val)
                    pdf_types_le_val = [pdf_types_le_val[i] / normalization for i in range(num_types_left)]
                    cdf_types_le_val = [cdf_types_le_val[i] / normalization for i in range(num_types_left)]
                    val_times_dist = [types_le_val[i] * pdf_types_le_val[i] for i in range(num_types_left)]
                    bid = sum(val_times_dist)
            else:
                # Compute conditional probabilities
                dist = [0] * len(self.possible_types)
                for i in range(len(self.possible_types)):
                    if self.possible_types[i] < self.valuations[0]:
                        dist[i] = self.type_dist[i]
                    else:
                        dist[i] = 0.0
                dist_cdf = [0] * len(self.possible_types)
                for idx, v in enumerate(self.possible_types):
                    dist_cdf[idx] = float(scipy.integrate.simps(
                        dist[:idx + 1], self.possible_types[:idx + 1]))
                normalization = max(dist_cdf)
                dist = [dist[i] / normalization for i in range(len(self.possible_types))]
                dist_cdf = [dist_cdf[i] / normalization for i in range(len(self.possible_types))]
                val_times_dist = [self.possible_types[i] * dist[i] for i in range(len(self.possible_types))]
                types_to_int = [self.possible_types[i] for i in range(len(self.possible_types))
                                if self.possible_types[i] < self.valuations[0]]
                val_times_dist = [val_times_dist[i] for i in range(len(types_to_int))]
                bid = scipy.integrate.simps(val_times_dist, types_to_int)

        self.bid[r] = bid
        return self.bid[r]
