"""
Implements a bidder that learns how to bid using a Markov Decision Process described in [1].

The states have been modified so that after round 1, prices are stored.

[1] Greenwald, Amy, and Justin Boyan. "Bidding under uncertainty: Theory and experiments." Proceedings of the 20th
conference on Uncertainty in artificial intelligence. AUAI Press, 2004.
"""
from bidder.mdp_uai import MDPBidderUAI
import math
import scipy.integrate
import scipy.interpolate
from auction.SequentialAuction import SequentialAuction
from scipy.stats import bernoulli


class MDPBidderUAIAugS(MDPBidderUAI):
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
        # Init will call make_state_space, but we need some more information before going through this process.
        MDPBidderUAI.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
        self.num_price_samples = len(self.action_space)
        self.num_prices_for_state = 10
        state_price_delta = float(max(possible_types) - min(possible_types)) / self.num_prices_for_state
        self.prices_in_state = [i * state_price_delta for i in range(self.num_prices_for_state)]
        self.make_state_space_after_init()

    def make_state_space(self):
        pass

    def make_state_space_after_init(self):
        """
        """
        # Round 0 will not keep price information in the state
        for X in range(self.num_rounds + 1):
            self.state_space.append((X, 0))

        # Rounds > 0 will keep price information in the state, except for the terminal states
        for X in range(self.num_rounds + 1):
            for j in range(1, self.num_rounds):
                for p in self.prices_in_state:
                    self.state_space.append((X, j, p))

        # Terminal states do not store price information
        for X in range(self.num_rounds + 1):
            self.state_space.append((X, self.num_rounds))
            self.terminal_states.append((X, self.num_rounds))

    def learn_auction_parameters(self, bidders, num_mc=1000):
        """
        Learn the highest bid of n - 1 bidders and the probability of winning.

        :param bidders: List.  Bidders to learn from.
        :param num_mc: Integer.  Number of times to test an action.
        """
        exp_payment = {}
        exp_T = {}
        prob_win = {}
        win_count = {}
        sa_counter = {}
        sas_counter = {}
        highest_other_bid = {}
        for s in self.state_space:
            highest_other_bid[s] = []
            for a in self.action_space:
                exp_payment[(s, a)] = 0.0
                prob_win[(s, a)] = 0.0
                win_count[(s, a)] = 0.0
                sa_counter[(s, a)] = 0.0
                for s_ in self.state_space:
                    sas_counter[(s, a, s_)] = 0.0
                    exp_T[(s, a, s_)] = 0.0

        sa = SequentialAuction(bidders, self.num_rounds)
        for t in range(num_mc):
            # Refresh bidders
            for bidder in bidders:
                bidder.valuations = bidder.make_valuations()
                bidder.reset()
            # Run auction and learn results of nth bidder
            sa.run()
            num_won = 0
            last_price_seen = None
            for j in range(self.num_rounds):
                largest_bid_amongst_n_minus_1 = max(sa.bids[j][:-1])
                if last_price_seen is None:
                    s = (num_won, j)
                else:
                    s = (num_won, j, last_price_seen)
                highest_other_bid[s].append(largest_bid_amongst_n_minus_1)
                a = min(self.action_space, key=lambda x: abs(x - sa.bids[j][-1]))
                sa_counter[(s, a)] += 1
                if largest_bid_amongst_n_minus_1 < a:
                    win_count[(s, a)] += 1
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1
                    num_won += 1
                    last_price_seen = min(self.prices_in_state, key=lambda x: abs(x - sa.payments[j]))
                    s_ = self.get_next_state(j + 1, num_won, last_price_seen)
                    sas_counter[(s, a, s_)] += 1
                elif largest_bid_amongst_n_minus_1 == a:
                    num_same_bid = sum(b == a for b in sa.bids[j][:-1])
                    prob_winning_tie = num_same_bid / self.num_bidders
                    win_count[(s, a)] += prob_winning_tie
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1 * prob_winning_tie
                    won_this_round = bernoulli.rvs(prob_winning_tie)
                    num_won += won_this_round
                    last_price_seen = min(self.prices_in_state, key=lambda x: abs(x - sa.payments[j]))
                    s_ = self.get_next_state(j + 1, num_won, last_price_seen)
                    sas_counter[(s, a, s_)] += 1
                else:
                    last_price_seen = min(self.prices_in_state, key=lambda x: abs(x - sa.payments[j]))
                    s_ = self.get_next_state(j + 1, num_won, last_price_seen)
                    sas_counter[(s, a, s_)] += 1

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                if sa_counter[(s, a)] > 0:
                    exp_payment[(s, a)] /= sa_counter[(s, a)]
                elif a_idx > 0:
                    exp_payment[(s, a)] = exp_payment[(s, self.action_space[a_idx - 1])]
        self.exp_payment = exp_payment

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                for s_ in self.state_space:
                    if sa_counter[(s, a)] == 0:
                        exp_T[(s, a, s_)] = 0.0
                    else:
                        exp_T[(s, a, s_)] = sas_counter[(s, a, s_)] / sa_counter[(s, a)]
        self.exp_T = exp_T

        for s in self.state_space:
            for a_idx, a in enumerate(self.action_space):
                if sa_counter[(s, a)] > 0:
                    prob_win[(s, a)] = win_count[(s, a)] / sa_counter[(s, a)]
                elif a_idx > 0:
                    prob_win[(s, a)] = prob_win[(s, self.action_space[a_idx - 1])]
        self.prob_win = prob_win

        self.perform_price_prediction(highest_other_bid)
        self.calc_transition_matrix()

    def get_next_state(self, next_round, num_won, last_price_seen):
        if next_round == self.num_rounds:
            s_ = (num_won, next_round)
        else:
            s_ = (num_won, next_round, last_price_seen)
        return s_

    def calc_transition_matrix(self):
        """
        Calculate the probability of transitioning from state s and action a to state s_ for all possible seen states
        and actions.
        """
        for s in self.state_space:
            if s not in self.price_prediction.keys():
                continue
            # Calculate distribution statistics for possible bids
            interp_pdf = scipy.interpolate.interp1d(self.price_prediction[s], self.price_pdf[s],
                                                    kind='slinear')
            interp_cdf = scipy.interpolate.interp1d(self.price_prediction[s], self.price_cdf[s],
                                                    kind='slinear')
            Fb = [0] * len(self.action_space)
            for a_idx, a in enumerate(self.action_space):
                if a < self.price_prediction[s][0]:
                    Fb[a_idx] = 0.0
                elif a > self.price_prediction[s][-1]:
                    Fb[a_idx] = 1.0
                else:
                    Fb[a_idx] = float(interp_cdf(a))
                if math.isnan(Fb[a_idx]):
                    Fb[a_idx] = 0.0
            # Update the transition matrix
            for a_idx, a in enumerate(self.action_space):
                if s[1] == self.num_rounds - 1:
                    win_state = (s[0] + 1, s[1] + 1)
                    self.T[(s, a, win_state)] = Fb[a_idx]
                    lose_state = (s[0], s[1] + 1)
                    self.T[(s, a, lose_state)] = 1.0 - Fb[a_idx]
                else:
                    for p in self.prices_in_state:
                        win_state = (s[0] + 1, s[1] + 1, p)
                        self.T[(s, a, win_state)] = Fb[a_idx]
                        lose_state = (s[0], s[1] + 1, p)
                        self.T[(s, a, lose_state)] = 1.0 - Fb[a_idx]

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        bid = argmax_a Q(s,a)

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        if (r == 0) or self.num_rounds == 1:
            s = (self.num_goods_won, r)
        else:
            s = (self.num_goods_won, r, self.anounced_price[r])
        maxQ = -float('inf')
        for a in self.action_space:
            maxQ = max(maxQ, self.Q[(s, a)])
        maxQ_actions = [a for a in self.action_space if self.Q[(s, a)] == maxQ]
        bid = min(maxQ_actions)
        self.bid[r] = bid
        return bid
