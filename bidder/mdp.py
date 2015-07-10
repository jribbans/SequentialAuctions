"""
Implements a bidder that learns how to bid in the first round of an auction using Markov Decision Processes.
"""
from bidder.simple import SimpleBidder
from auction.SequentialAuction import SequentialAuction
from copy import deepcopy
import random
import numpy
import heapq

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
        self.num_price_samples = 101
        self.price_dist = [0] * self.num_price_samples
        self.price_cdf = [0] * self.num_price_samples
        self.price = [0] * self.num_price_samples

    def calc_prob_winning(self, bidders, current_round, num_trials_per_action=100):
        """
        Calculate the probability of winning against a set of bidders for each action in the action space.

        :param bidders: List.  Bidders to learn from.
        :param current_round: Integer.  Round of interest.
        :param num_trials_per_action: Integer.  Number of times to test an action.
        """
        r = current_round - 1
        win_count = [0] * len(self.action_space)
        prices = []
        sa = SequentialAuction(bidders, self.num_rounds)
        for a_idx, a in enumerate(self.action_space):
            for t in range(num_trials_per_action):
                # Have bidders sample new valuations
                for bidder in bidders:
                    bidder.valuations = bidder.make_valuations()
                    bidder.reset()
                # Run an auction
                sa.bidders = bidders
                sa.run()
                # See if the action we are using leads to a win
                if max(sa.bids[r]) < a:
                    win_count[a_idx] += 1
                elif max(sa.bids[0]) == a:
                    # Increment based on how many bidders bid the same bid
                    num_same_bid = sum(b == a for b in sa.bids[0])
                    win_count[a_idx] += num_same_bid / self.num_bidders
                # Keep track of prices
                prices.append(sa.payments[r])

        prob_win = [win_count[i] / num_trials_per_action for i in range(len(self.possible_types))]
        self.prob_winning[r] = prob_win
        # Get the density and values associated with the density of the prices seen
        # hist has n elements, and bin_edges has n+1 elements
        hist, bin_edges = numpy.histogram(prices, self.num_price_samples, density=True)
        cdf = numpy.cumsum(hist * numpy.diff(bin_edges))
        price = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(self.num_price_samples)]
        self.price = price
        self.price_dist = hist
        self.price_cdf = cdf

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        pass
