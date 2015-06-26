"""
Test the KatzmanBidder implementation
"""
from auction.SequentialAuction import SequentialAuction
from bidder.katzman import KatzmanBidder
import random
import numpy

# Initialize random number seeds for repeatability
random.seed(0)
numpy.random.seed(0)

# Auction parameters
num_rounds = 2
num_bidders = 5
# values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1 / 101.0 for i in range(101)]

# Generate bidders
bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist)
           for i in range(num_bidders)]

# Run auction
auction = SequentialAuction(bidders, num_rounds)
auction.run()
auction.print_summary()
auction.print_round_overview()
auction.print_bidder_results()
