"""
Plot equation 1 of [1] for a uniform distribution of valuations in [0,1].

[1] Katzman, Brett. "A two stage sequential auction with multi-unit demands."
Journal of Economic Theory 86.1 (1999): 77-99.
"""

from bidder.katzman import KatzmanBidder
import matplotlib.pyplot as plt

# Auction parameters
num_rounds = 2
num_bidders = 2
# Values in possible_types must be increasing.
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1 / 101.0 for i in range(101)]

# Round of interest
round_number = 1

# Get the bid values for each type
bidder = KatzmanBidder(0, num_rounds, num_bidders, possible_types, type_dist)
bids = [0] * len(possible_types)
for idx, val in enumerate(possible_types):
    bidder.valuations[0] = val
    bids[idx] = bidder.place_bid(1)

# Make a plot of the results
fig1 = plt.plot()
plt.plot(possible_types, bids)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Valuation')
plt.ylabel('Bid')
plt.title('KatzmanBidder Round 1 Bids, Uniform(0,1), N = ' + str(num_bidders))

# Save the output to a file, then show
plt.savefig('katzman_bid1_N_2.png')
plt.show()
