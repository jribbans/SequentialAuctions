"""
Implements a bidder that learns how to bid using a Markov Decision Process described in [1].

[1] Greenwald, Amy, and Justin Boyan. "Bidding under uncertainty: Theory and experiments." Proceedings of the 20th
conference on Uncertainty in artificial intelligence. AUAI Press, 2004.
"""
from bidder.AbstractMDPBidder import AbstractMDPBidder
import math
import numpy
import scipy.integrate
import scipy.interpolate
from auction.SequentialAuction import SequentialAuction
from collections import defaultdict


class MDPBidderUAI(AbstractMDPBidder):
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
        AbstractMDPBidder.__init__(self, bidder_id, num_rounds, num_bidders, possible_types, type_dist, type_dist_disc)
        self.num_price_samples = len(self.action_space)
        self.num_prices_for_state = len(self.action_space) - 1
        state_price_delta = float(max(possible_types) - min(possible_types)) / self.num_prices_for_state
        self.prices_in_state = [i * state_price_delta for i in range(self.num_prices_for_state)]
        self.prices_in_state = set()
        self.state_space = set()
        self.terminal_states = set()
        self.action_space = set()

    def make_state_space(self):
        pass

    def make_action_space(self):
        pass

    def make_state_space_after_init(self):
        """
        """
        for X in range(self.num_rounds + 1):
            for j in range(self.num_rounds + 1):
                # Number of goods won cannot exceed the round number
                if X <= j:
                    self.state_space.append((X, j))
                    if j == self.num_rounds:
                        self.terminal_states.append((X, j))

    def learn_auction_parameters(self, bidders, num_mc=1000):
        """
        Learn the highest bid of n - 1 bidders and the probability of winning.

        :param bidders: List.  Bidders to learn from.
        :param num_mc: Integer.  Number of times to test an action.
        """
        exp_payment = defaultdict(float)
        exp_T = defaultdict(float)
        prob_win = defaultdict(float)
        win_count = defaultdict(float)
        sa_counter = defaultdict(float)
        sas_counter = defaultdict(float)
        highest_other_bid = defaultdict(list)

        sa = SequentialAuction(bidders, self.num_rounds)
        self.state_space.add((0, 0))
        for t in range(num_mc):
            # Refresh bidders
            for bidder in bidders:
                bidder.valuations = bidder.make_valuations()
                bidder.reset()
            # Run auction and learn results of nth bidder
            sa.run()
            num_won = 0
            last_price_seen = None
            s = s_ = (0, 0)
            for j in range(self.num_rounds):
                s = s_
                largest_bid_amongst_n_minus_1 = round(max(sa.bids[j][:-1]), 2)
                highest_other_bid[s].append(largest_bid_amongst_n_minus_1)
                # The action closest to the Nth bidder
                a = round(sa.bids[j][-1], 2)
                self.action_space.add(a)
                sa_counter[(s, a)] += 1
                won_this_round = bidders[-1].win[j]
                # Outcome depends on the action we placed, which is hopefully close to what the Nth bidder used.
                if won_this_round:
                    win_count[(s, a)] += 1
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1
                    num_won += 1
                    p = round(sa.payments[j], 2)
                    self.prices_in_state.add(p)
                    last_price_seen = min(self.prices_in_state, key=lambda x: abs(x - p))
                else:
                    last_price_seen = 0.0
                s_ = self.get_next_state(s, won_this_round)
                self.state_space.add(s_)
                sas_counter[(s, a, s_)] += 1
            self.state_space.add(s_)
            self.terminal_states.add(s_)

        self.num_price_samples = len(self.action_space)

        # Turn these into lists and sort them, so that access is ordered and predictable.
        self.state_space = list(self.state_space)
        self.state_space.sort()
        self.terminal_states = list(self.terminal_states)
        self.terminal_states.sort()
        self.action_space = list(self.action_space)
        self.action_space.sort()
        self.prices_in_state = list(self.prices_in_state)
        self.prices_in_state.sort()
        self.num_price_samples = len(self.prices_in_state)

        self.exp_payment = {(s, a): exp_payment[(s, a)] / sa_counter[(s, a)]
                            for (s, a) in sa_counter.keys()}

        self.exp_T = {(s, a, s_): sas_counter[(s, a, s_)] / sa_counter[(s, a)]
                      for (s, a, s_) in sas_counter.keys()}

        self.prob_win = {(s, a): win_count[(s, a)] / sa_counter[(s, a)]
                         for (s, a) in sa_counter.keys()}

        self.perform_price_prediction(highest_other_bid)
        self.calc_transition_matrix()

    def get_next_state(self, current_state, won_this_round):
        X = current_state[0] + int(won_this_round)
        j = current_state[1] + 1
        s_ = (X, j)
        return s_

    def perform_price_prediction(self, highest_other_bid):
        """
        From the bids seen, determine price statistics.
        """
        for s in self.state_space:
            # Only proceed if there is something to do
            if not highest_other_bid[s]:
                continue
            highest_other_bid[s].sort()
            if self.type_dist_disc:
                # Generate a histogram and obtain probability mass values
                hist, bin_edges = numpy.histogram(highest_other_bid[s], self.num_price_samples, density=False)
                self.price_prediction[s] = bin_edges[:-1].tolist()
                hist = hist.tolist()
                total = float(sum(hist))
                hist = [h / total for h in hist]
                assert abs(sum(hist) - 1.0) < .00001, "Probability mass values are not correct"
                self.price_pdf[s] = hist
                self.price_cdf[s] = numpy.cumsum(hist).tolist()
            else:
                # Generate a histogram and obtain probability density values
                hist, bin_edges = numpy.histogram(highest_other_bid[s], self.num_price_samples, density=True)
                self.price_prediction[s] = bin_edges[:-1].tolist()
                self.price_pdf[s] = hist.tolist()
                self.price_cdf[s] = numpy.cumsum(hist * numpy.diff(bin_edges)).tolist()

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
                win_state = (s[0] + 1, s[1] + 1)
                self.T[(s, a, win_state)] = Fb[a_idx]
                lose_state = (s[0], s[1] + 1)
                self.T[(s, a, lose_state)] = 1.0 - Fb[a_idx]

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        for s in self.price_prediction.keys():
            for a_idx, a in enumerate(self.action_space):
                r = [-p if p <= a else 0.0 for p in self.price_prediction[s]]
                if self.type_dist_disc:
                    to_sum = [r[p_idx] * self.price_pdf[s][p_idx]
                              for p_idx, p in enumerate(self.price_prediction[s])]
                    self.R[(s, a)] = sum(to_sum[i]
                                         for i in range(len(to_sum)))
                else:
                    to_integrate = [r[p_idx] * self.price_pdf[s][p_idx]
                                    for p_idx, p in enumerate(self.price_prediction[s])]
                    self.R[(s, a)] = scipy.integrate.trapz(to_integrate, self.price_prediction[s])

        self.calc_terminal_state_rewards()

    def calc_terminal_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        for X, j in self.terminal_states:
            for a in self.action_space:
                self.R[((X, j), a)] = sum(self.valuations[:X])
