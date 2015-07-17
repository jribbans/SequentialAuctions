"""
Abstract class that implements a bidder that learns how to bid using a Markov Decision Process.
"""
from bidder.simple import SimpleBidder


class AbstractMDPBidder(SimpleBidder):
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
        self.state_space = []
        self.terminal_states = []
        self.action_space = []
        self.T = {}
        self.R = {}
        self.Q = {}
        self.V = {}

        self.price_prediction = {}
        self.price_pdf = {}
        self.price_cdf = {}

        self.make_state_space()
        self.make_action_space()

    def make_state_space(self):
        """
        Make a list of all possible states.
\       """
        pass

    def make_action_space(self):
        """
        Make a list of all possible actions.
\       """
        pass

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

        Q(s,a) = R(s,a) + \sum_{s'} (T(s,a,s')*V(s'))
        V(s) = max_a Q(s,a)
        \pi(s) = argmax_a Q(s,a)
        """
        self.V, self.Q = self.value_iteration()

    def value_iteration(self):
        """
        Solve the MDP using value iteration.

        :return: V: List.  V-values
        :return: Q: List.  Q-values.
        """
        # Initialize
        Q = {}
        V = {}
        for s in self.state_space:
            for a in self.action_space:
                Q[(s, a)] = 0.0
                V[s] = 0.0
        for s in self.terminal_states:
            for a in self.action_space:
                V[s] = self.R[(s, a)]

        # Solve
        num_iter = 0
        convergence_threshold = 0.00001
        converged = False
        while not converged:
            num_iter += 1
            # Update Q
            for s in self.state_space:
                for a in self.action_space:
                    if (s, a) not in self.R.keys():
                        continue
                    Q[(s, a)] = self.R[(s, a)]
                    for s_ in self.state_space:
                        if (s, a, s_) not in self.T.keys():
                            continue
                        Q[(s, a)] += self.T[(s, a, s_)] * V[s_]

            # Update V
            largest_diff_V = -float('inf')
            for s in self.state_space:
                old_V = V[s]
                new_V = -float('inf')
                for a in self.action_space:
                    new_V = max(new_V, Q[(s, a)])
                V[s] = new_V
                largest_diff_V = max(largest_diff_V, abs(new_V - old_V))

            # Test for convergence
            converged = largest_diff_V <= convergence_threshold

        return V, Q

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        bid = argmax_a Q(s,a)

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        s = (self.num_goods_won, r)
        maxQ = -float('inf')
        for a in self.action_space:
            maxQ = max(maxQ, self.Q[(s, a)])
        maxQ_actions = [a for a in self.action_space if self.Q[(s, a)] == maxQ]
        bid = min(maxQ_actions)
        self.bid[r] = bid
        return bid
