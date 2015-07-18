"""
Implements a class that runs a sequential auction of homogeneous goods, where one good is auctioned off in each
round.  Bidders pay the second highest price in each round.
"""
import random


class SequentialAuction:
    """Sequential auction where the bidder with the highest bid in each round pays the second highest price.
    """

    def __init__(self, bidders, num_rounds):
        """
        :param bidders: List.  A list of bidders participating in the auction.
        :param num_rounds: Integer.  The number of rounds the auction will run for.
        """
        self.bidders = bidders
        self.num_bidders = len(bidders)
        self.num_rounds = num_rounds
        self.bids = [[0] * self.num_bidders for r in range(num_rounds)]
        self.winners = [0] * num_rounds
        self.payments = [0] * num_rounds
        self.total_revenue = 0.0

    def run(self):
        """
        Runs the auction to completion.
        """
        for r in range(self.num_rounds):
            round_number = r + 1
            # Place bids
            for i in range(self.num_bidders):
                self.bids[r][i] = self.bidders[i].place_bid(round_number)
            # Winner determination
            candidates = [i for i in range(self.num_rounds)
                          if self.bids[r][i] == max(self.bids[r])]
            self.winners[r] = random.choice(candidates)
            # Determine payment
            self.payments[r] = float(sorted(self.bids[r])[-2])
            self.total_revenue += self.payments[r]
            # Notify winning bidder
            for i, b in enumerate(self.bidders):
                if i == self.winners[r]:
                    self.bidders[i].set_round_result(round_number, True, self.payments[r], self.payments[r])
                else:
                    self.bidders[i].set_round_result(round_number, False, 0.0, self.payments[r])

    def print_bidder_results(self):
        """
        Prints out the informatin of each bidder for each round.
        """
        print('Bidder Information')
        for i, bidder in enumerate(self.bidders):
            print('Bidder', i, ' v =', bidder.valuations)
            print('Bids:', bidder.bid)
            print('Win Result:', bidder.win)
            print('Valuation:', bidder.valuations)
            print('Payment:', bidder.payment)
            print('Utility:', bidder.utility)
            print('Number of goods procured:', bidder.num_goods_won)

    def print_round_overview(self):
        """
        Prints out a high-level overview of the results of each round.
        """
        print('Round overview')
        for r in range(self.num_rounds):
            round_number = r + 1
            print('Auction round', round_number)
            print('Bids', self.bids[r])
            print('Winner: Bidder', self.winners[r])
            print('Payment:', self.payments[r])

    def print_summary(self):
        """
        Prints out a summary of the result of the auction.
        """
        print('Summary')
        print('Winners:', self.winners)
        print('Payments:', self.payments)
        print('Total Revenue:', self.total_revenue)
