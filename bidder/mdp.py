"""
Implements a bidder that learns how to bid in the first round of an auction using Markov Decision Processes.
"""
from bidder.simple import SimpleBidder
from auction.SequentialAuction import SequentialAuction
from copy import deepcopy
import random
import numpy

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)


class MDPBidder(SimpleBidder):
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
        SimpleBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
        self.action_space = possible_types
        self.prob_winning = [[0.0] * len(self.action_space) for i in range(num_rounds)]

    def calc_prob_winning(self, bidders, current_round):
        r = current_round - 1
        num_trials_per_a = 100
        win_count = [0] * len(self.action_space)
        sa = SequentialAuction(bidders, self.num_rounds)
        for a_idx, a in enumerate(self.action_space):
            for t in range(num_trials_per_a):
                for bidder in bidders:
                    bidder.valuations = bidder.make_valuations()
                    bidder.reset()
                sa.bidders = bidders
                sa.run()
                if max(sa.bids[r]) < a:
                    win_count[a_idx] += 1
                elif max(sa.bids[0]) == a:
                    win_count[a_idx] += 0.5
        prob_win = [win_count[i] / num_trials_per_a for i in range(len(self.possible_types))]
        self.prob_winning[r] = prob_win

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        pass
