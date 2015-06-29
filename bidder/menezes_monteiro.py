"""
Implements a sequential auction bidder using the analysis found in [1].

[1] Menezes, Flavio M., and Paulo K. Monteiro. "Synergies and price trends in sequential auctions." Review of Economic
Design 8.1 (2003): 85-98.
"""
from bidder.simple import SimpleBidder
import scipy.integrate
import scipy.interpolate


class MenezesMonteiroBidder(SimpleBidder):
    """A bidder that bids using the strategy described in [1].
    In this class, valuations are marginal valuations.  valuations[1] is the marginal valuation of good 1, given good 0.
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
        assert not type_dist_disc, "Valuations must come from a continuous distribution."
        SimpleBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)

    def place_bid(self, current_round):
        """
        Places a bid.  The bid placed is equal to the valuation of the kth good won.

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        positive_synergy = self.valuations[0] + self.valuations[1] > 2 * self.valuations[0]
        if current_round == 1:
            if self.num_bidders == 2:
                bid = self.valuations[1]
            else:
                types_le_val, cdf_types_le_val, pdf_types_le_val = self.get_types_le_val(self.valuations[0])
                # F(x)^{n - 2}
                cdf_at_x_pow_n_minus_2 = cdf_types_le_val[-1] ** (self.num_bidders - 2.0)
                # max(delta(x) - x, y)
                z = [max(self.valuations[1], types_le_val[i]) for i in range(len(types_le_val))]
                # F(y)^{n - 3}
                cdf_pow_n_minus_3 = [cdf_types_le_val[i] ** (self.num_bidders - 3.0)
                                     for i in range(len(cdf_types_le_val))]
                # max(delta(x) - x, y) * F(y)^{n-3} f(y)
                to_int = [z[i] * cdf_pow_n_minus_3[i] * pdf_types_le_val[i] for i in range(len(types_le_val))]
                # Result of integration
                integral = scipy.integrate.simps(to_int, types_le_val)
                # Use all of the above and compute
                # # b(x) = \frac{n-2}{F(x)^{n-2}} \int_{0}^{x} \max \{ \delta(x) - x, y \} F(y)^{n - 3} f(y) \, dy
                if abs(cdf_at_x_pow_n_minus_2) <= 1e-5:
                    bid = self.valuations[1];
                else:
                    bid = ((self.num_bidders - 2.0) / cdf_at_x_pow_n_minus_2) * integral
                print(self.valuations[0], cdf_at_x_pow_n_minus_2, integral, to_int)
        else:
            if self.num_goods_won == 0:
                bid = self.valuations[0]
            else:
                bid = self.valuations[1]
        self.bid[r] = bid
        return bid
