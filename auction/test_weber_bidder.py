"""
Test the WeberBidder implementation
"""
from auction.SequentialAuction import SequentialAuction
from bidder.weber import WeberBidder
import random
import numpy

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
num_bidders = 2

# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]

# Discrete type distribution
print("----------")
print("Discrete Distribution")
type_dist = [1.0 / len(possible_types)] * len(possible_types)
# Generate bidders
bidders_disc = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, True)
                for i in range(num_bidders)]
# Run auction
auction = SequentialAuction(bidders_disc, num_rounds)
auction.run()
auction.print_summary()
auction.print_round_overview()
auction.print_bidder_results()

# Continuous type distribution
print("----------")
print("Continuous Distribution")
type_dist = [1.0] * len(possible_types)
# Generate bidders
bidders_cont = [WeberBidder(i, num_rounds, num_bidders, possible_types, type_dist, False)
                for i in range(num_bidders)]
for i in range(num_bidders):
    bidders_cont[i].valuations = bidders_disc[i].valuations
# Run auction
auction = SequentialAuction(bidders_cont, num_rounds)
auction.run()
auction.print_summary()
auction.print_round_overview()
auction.print_bidder_results()
