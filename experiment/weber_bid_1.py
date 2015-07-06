"""
Plot equation 1 of [1] for a uniform distribution of valuations in [0,1].

[1] Weber, Robert J. "MULTIPLE—OBJECT AUCTIONS." (1981).
"""

from bidder.weber import WeberBidder
import matplotlib.pyplot as plt
import scipy.stats
from math import floor, ceil


def make_plot(possible_types, type_dist, type_dist_disc, plot_title, filename):
    bidder = WeberBidder(0, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
    bids = [0] * len(possible_types)
    for idx, val in enumerate(possible_types):
        bidder.valuations[0] = val
        bids[idx] = bidder.place_bid(1)

    # Make a plot of the results
    plt.figure()
    plt.plot(possible_types, bids)
    plt.xlabel('Valuation')
    plt.ylabel('Bid')
    plt.title(plot_title)
    xlim = [floor(min(bidder.valuations)), ceil(max(bidder.valuations))]
    ylim = [floor(min(bids)), ceil(max(bids))]
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save the output to a file, then show
    plt.savefig(filename)
    plt.show()

# Auction parameters
num_rounds = 2
num_bidders = 4
# Round of interest
round_number = 1

# Generate figures.  Note that values in possible_types must be increasing.
# Uniform
possible_types = [i / 100.0 for i in range(101)]
type_dist = [1 / 101.0 for i in range(101)]
type_dist_disc = True
plot_title = 'WeberBidder Round 1 Bids, Uniform(0,1), N = ' + str(num_bidders)
filename = 'weber_bid1_N_4_uniform.png'
make_plot(possible_types, type_dist, type_dist_disc, plot_title, filename)

# Geometric
p = 0.5
possible_types = [i for i in range(1, 11)]
type_dist = scipy.stats.geom.pmf(possible_types, p).tolist()
plot_title = 'WeberBidder Round 1 Bids, Geometric, p = ' + str(p) + ', N = ' + str(num_bidders)
filename = 'weber_bid1_N_4_geometric.png'
make_plot(possible_types, type_dist, type_dist_disc, plot_title, filename)
