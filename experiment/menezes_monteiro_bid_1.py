"""
Plot equation 1 of [1] for a uniform distribution of valuations in [0,1].

[1] Menezes, Flavio M., and Paulo K. Monteiro. "Synergies and price trends in sequential auctions." Review of Economic
Design 8.1 (2003): 85-98.
"""

from bidder.menezes_monteiro import MenezesMonteiroBidder
import matplotlib.pyplot as plt
import scipy.stats
from math import floor, ceil


def make_plot(possible_types, type_dist, type_dist_disc, plot_title, filename):
    bidder = MenezesMonteiroBidder(0, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
    bids = [0] * len(possible_types)
    for idx, val in enumerate(possible_types):
        bidder.valuations[0] = val
        bidder.valuations[1] = val + .01
        bids[idx] = bidder.place_bid(1)

    # Make a plot of the results
    plt.figure()
    plt.plot(possible_types, bids)
    plt.xlabel('Valuation')
    plt.ylabel('Bid')
    plt.title(plot_title)

    # Save the output to a file, then show
    plt.savefig(filename)
    plt.show()

# Auction parameters
num_rounds = 2
num_bidders = 3
# Round of interest
round_number = 1

# Generate figures.  Note that values in possible_types must be increasing.
# Uniform
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1.0 for i in range(101)]
type_dist_disc = False
plot_title = 'MenezesMonteiroBidder Round 1 Bids, Uniform(0,1), N = ' + str(num_bidders)
filename = 'menezes_monteiro_bid1_N_2_uniform.png'
make_plot(possible_types, type_dist, type_dist_disc, plot_title, filename)
