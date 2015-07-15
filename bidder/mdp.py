"""
Abstract class that implements a bidder that learns how to bid using a Markov Decision Process.
"""
from bidder.simple import SimpleBidder


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
        self.price_pdf = [[0] * self.num_price_samples for r in range(self.num_rounds)]
        self.price_cdf = [[0] * self.num_price_samples for r in range(self.num_rounds)]
        self.price_prediction = [[0] * self.num_price_samples for r in range(self.num_rounds)]
        self.price_cdf_at_bid = [[0] * self.num_price_samples for r in range(self.num_rounds)]
        # Parameters for solving the Markov Decision Process
        # States: (X, j).  X goods won at round j
        # Actions: b.  Bid b from the action space.
        self.R = [[[0 for b in range(len(self.action_space))]
                   for j in range(self.num_rounds + 1)]
                  for X in range(self.num_rounds + 1)]
        self.Q = [[[0 for b in range(len(self.action_space))]
                   for j in range(self.num_rounds + 1)]
                  for X in range(self.num_rounds + 1)]
        self.V = [[0 for j in range(self.num_rounds + 1)]
                  for X in range(self.num_rounds + 1)]

    def learn_auction_parameters(self, bidders):
        """
        Learn how to bid.

        :param bidders: List.  Bidders to learn from.
        """
        pass

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        pass

    def solve_mdp(self):
        """
        Run value iteration.

        Q(s,a) = \sum_{s'} (T(s,a,s')*(R(s,a,s') + V(s'))
        V(s) = max_a Q(s,a)
        \pi(s) = argmax_a Q(s,a)
        """
        # Initialize all values to 0
        for X in range(self.num_rounds + 1):
            for j in range(self.num_rounds + 1):
                self.V[X][j] = 0
                self.Q[X][j] = [0] * len(self.action_space)

        for X in range(self.num_rounds + 1):
            self.V[X][self.num_rounds] = self.R[X][self.num_rounds][0]

        # Value iteration
        num_iter = 0
        convergence_threshold = 0.00001
        largest_diff_V = float('inf')
        while largest_diff_V > convergence_threshold:
            num_iter += 1
            # Update Q
            for X in range(self.num_rounds):
                for j in range(self.num_rounds):
                    # Bidder at round j can't have more than j goods
                    if X <= j:
                        for b_idx, b in enumerate(self.action_space):
                            Q_win = self.price_cdf_at_bid[j][b_idx] * (self.R[X][j][b_idx] + self.V[X + 1][j + 1])
                            Q_lose = (1.0 - self.price_cdf_at_bid[j][b_idx]) * (0.0 + self.V[X][j + 1])
                            self.Q[X][j][b_idx] = Q_win + Q_lose

            # Update V
            largest_diff_V = -float('inf')
            for X in range(self.num_rounds):
                for j in range(self.num_rounds):
                    if X <= j:
                        old_V = self.V[X][j]
                        new_V = max(self.Q[X][j])
                        self.V[X][j] = new_V
                        largest_diff_V = max(largest_diff_V, abs(new_V - old_V))

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        bid = argmax_a Q(s,a)

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        maxQ = max(self.Q[self.num_goods_won][r])
        maxQ_idxs = [i for i in range(len(self.Q[self.num_goods_won][r]))
                     if self.Q[self.num_goods_won][r][i] == maxQ]
        possible_bids = [self.action_space[idx] for idx in maxQ_idxs]
        bid = min(possible_bids)
        self.bid[r] = bid
        return bid
