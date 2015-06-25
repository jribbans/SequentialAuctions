"""
Test the KatzmanBidder implementation
"""
from bidder.katzman import KatzmanBidder
from heapq import nlargest
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

# Initialize list to store auction outcome
bids = [[0] * num_bidders for r in range(num_rounds)]
winners = [0] * num_rounds
payments = [0] * num_rounds
total_revenue = 0.0

# Generate bidders
bidders = [KatzmanBidder(i, num_rounds, num_bidders, possible_types, type_dist)
           for i in range(num_bidders)]

# Run auction
for r in range(num_rounds):
    round_number = r + 1
    # Place bids
    bids[r] = [bidders[i].place_bid(round_number) for i in range(num_bidders)]
    # Determine winner
    winners[r] = bids[r].index(max(bids[r]))
    # Determine payment
    payments[r] = float(nlargest(2, bids[r])[1])
    total_revenue += payments[r]
    # Notify winning bidder
    for i, b in enumerate(bidders):
        if i == winners[r]:
            bidders[i].set_round_result(round_number, True, payments[r])
        else:
            bidders[i].set_round_result(round_number, False, 0.0)

# Display outcome to terminal
print('Bidder information')
for i in range(num_bidders):
    print('Bidder', i, ' v =', bidders[i].valuations)
for r in range(num_rounds):
    round_number = r + 1
    print('Auction round', round_number)
    print('Bids placed')
    for i in range(num_bidders):
        print('Bidder', i, 'b =', bids[r][i])
    print('Winner: Bidder', winners[r])
    print('Payment:', payments[r])

print('Summary')
print('Winners:', winners)
print('Payments:', payments)
print('Total Revenue:', total_revenue)

print('Bidder Data')
for i, b in enumerate(bidders):
    print('Bidder', i)
    print('Win Result:', b.win)
    print('Valuation:', b.valuations)
    print('Payment:', b.payment)
    print('Utility:', b.utility)
    print('Number of goods procured:', b.num_goods_won)
