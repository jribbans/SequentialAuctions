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
        self.state_space = set()
        self.terminal_states = set()
        self.action_space = set()
        self.T = {}
        self.R = {}
        self.Q = {}
        self.V = {}
        self.pi = {}

        self.price_prediction = {}
        self.price_pdf = {}
        self.price_cdf = {}

        self.make_state_space()
        self.make_action_space()

        # Force the bidder to bid truthfully in the last round.
        self.bid_val_in_last_round = False
        if self.bid_val_in_last_round:
            for v in self.valuations:
                self.action_space.add(v)

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

    def perform_price_prediction(self):
        """
        Determine price statistics.
        """
        pass

    def calc_transition_matrix(self):
        """
        Calculate the probability of transitioning from state s and action a to state s_.
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
        self.pi = self.determine_policy()

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
                    if s in self.terminal_states:
                        continue
                    num_won = sum(s[i][0] for i in range(len(s)))
                    if (len(s) == self.num_rounds - 1) and (
                                a != self.valuations[num_won]) and self.bid_val_in_last_round:
                        Q[(s, a)] = -float('inf')
                        continue
                    Q[(s, a)] = self.R[(s, a)]
                    for s_ in self.state_space:
                        if (s, a, s_) not in self.T.keys():
                            continue
                        Q[(s, a)] += self.T[(s, a, s_)] * V[s_]

            # Update V
            largest_diff_V = -float('inf')
            for s in self.state_space:
                if s in self.terminal_states:
                    continue
                old_V = V[s]
                new_V = -float('inf')
                for a in self.action_space:
                    new_V = max(new_V, Q[(s, a)])
                V[s] = new_V
                largest_diff_V = max(largest_diff_V, abs(new_V - old_V))

            # Test for convergence
            converged = largest_diff_V <= convergence_threshold

        return V, Q

    def determine_policy(self):
        """
        """
        pi = {}
        for s in self.state_space:
            if s in self.terminal_states:
                continue
            maxQ = -float('inf')
            for a in self.action_space:
                maxQ = max(maxQ, self.Q[(s, a)])
            maxQ_actions = [a for a in self.action_space if self.Q[(s, a)] == maxQ]
            # Pick the action that corresponds to the lowest bid.  Since all possible bids have the same
            # Q value, there shouldn't be a reason to bid more than what's needed.
            # However, if we can bid true valuation, then do that.
            num_won = sum(s[i][0] for i in range(len(s)))
            if float(self.valuations[num_won]) in maxQ_actions:
                pi[s] = self.valuations[num_won]
            else:
                pi[s] = min(maxQ_actions)

        return pi

    def is_bidding_valuation_in_final_round(self):
        """
        """
        # Keep track of reachable states, starting from the initial state.
        to_process = [()]
        is_truthful = True
        while len(to_process) > 0:
            current_s = to_process[0]
            del to_process[0]
            # If we have reached a state corresponding to the final round, then check if we bid truthfully here.
            if len(current_s) == self.num_rounds - 1:
                # The number of goods won according to the state
                num_won = sum(current_s[i][0] for i in range(self.num_rounds - 1))
                # If the bidder is bidding truthfully in the last round, then the bid should equal valuation
                if self.pi[current_s] != self.valuations[num_won]:
                    print('Possible path does not have truthful bidding in last round')
                    print(current_s)
                    print('Num goods won =', num_won)
                    print('Valuation vector =', self.valuations)
                    print('Should bid =', self.valuations[num_won])
                    print('Policy at this state =', self.pi[current_s])
                    is_truthful = False
                    break
            else:
                # Add to the list of states we will process all states we can transition to using the current policy.
                for s in self.state_space:
                    # The next state has one entry more than the current state
                    if len(current_s) == len(s) - 1:
                        # The next state should contain the information of the current state.
                        if all(current_s[i] == s[i] for i in range(len(current_s))):
                            # The transition being made
                            # sas_ = (current_s, self.pi[current_s], s)
                            # Cannot reach states where the bidder bid more than the payment.
                            did_not_win_against_higher = ((s[-1][1] > self.pi[current_s]) and (s[-1][0] == 0))
                            did_not_lose_against_lower = ((s[-1][1] < self.pi[current_s]) and (s[-1][0] > 0))
                            is_tie = s[-1][1] == self.pi[current_s]
                            """
                            print(s, s[-1])
                            print(s[-1][0])
                            print(s[-1][1])
                            print(self.pi[current_s])
                            print(is_tie)
                            print(did_not_win_against_higher, did_not_lose_against_lower)
                            """
                            if is_tie or (did_not_win_against_higher or did_not_lose_against_lower):
                                print(self.valuations, current_s, self.pi[current_s], 'adding', s)
                                to_process.append(s)
                            else:
                                print(self.valuations, current_s, self.pi[current_s], 'NOT adding', s)
                                print(is_tie, did_not_win_against_higher, did_not_lose_against_lower)

        return is_truthful

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        bid = argmax_a Q(s,a)

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        s = (self.num_goods_won, r)
        if self.bid_val_in_last_round and (current_round == self.num_rounds):
            bid = self.valuations[self.num_goods_won]
        else:
            bid = self.pi[s]
        self.bid[r] = bid
        return bid
