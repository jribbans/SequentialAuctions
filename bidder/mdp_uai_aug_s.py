"""
Implements a bidder that learns how to bid using a Markov Decision Process.

States are tuples that store whether the bidder has won and the announced price.
"""
from bidder.mdp_uai import MDPBidderUAI
from auction.SequentialAuction import SequentialAuction
from collections import defaultdict


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
        self.num_price_samples = len(self.action_space)  # Not used
        self.num_prices_for_state = len(self.action_space) - 1
        state_price_delta = float(max(possible_types) - min(possible_types)) / self.num_prices_for_state
        self.prices_in_state = [i * state_price_delta for i in range(self.num_prices_for_state)]
        self.digit_precision = 2 # Learn values to this precision
        self.prices_in_state = set()
        self.state_space = set()
        self.terminal_states = set()
        self.action_space = set()
        self.use_given_pi = False
        self.given_pi = {}

    def make_state_space(self):
        pass

    def make_action_space(self):
        pass

    def make_state_space_after_init(self):
        """
        """
        pass

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
        self.state_space.add(())
        for t in range(num_mc):
            # Refresh bidders
            for bidder in bidders:
                bidder.valuations = bidder.make_valuations()
                for v in bidder.valuations:
                    self.action_space.add(round(v, self.digit_precision))
                bidder.reset()
            # Run auction and learn results of nth bidder
            sa.run()
            num_won = 0
            last_price_seen = None
            s = s_ = (())
            for j in range(self.num_rounds):
                s = s_
                largest_bid_amongst_n_minus_1 = round(max(sa.bids[j][:-1]), self.digit_precision)
                highest_other_bid[s].append(largest_bid_amongst_n_minus_1)
                # The action closest to the Nth bidder
                a = round(sa.bids[j][-1], self.digit_precision)
                self.action_space.add(a)
                sa_counter[(s, a)] += 1
                won_this_round = bidders[-1].win[j]
                price_this_round = round(sa.payments[j], self.digit_precision)
                # Outcome depends on the action we placed, which is hopefully close to what the Nth bidder used.
                if won_this_round:
                    win_count[(s, a)] += 1
                    exp_payment[(s, a)] -= largest_bid_amongst_n_minus_1
                    num_won += 1
                    self.prices_in_state.add(price_this_round)
                s_ = self.get_next_state(s, won_this_round, price_this_round)
                self.state_space.add(s_)
                sas_counter[(s, a, s_)] += 1
            self.state_space.add(s_)
            self.terminal_states.add(s_)

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

        self.T = {(s, a, s_): sas_counter[(s, a, s_)] / sa_counter[(s, a)]
                  for (s, a, s_) in sas_counter.keys()}

        self.prob_win = {(s, a): win_count[(s, a)] / sa_counter[(s, a)]
                         for (s, a) in sa_counter.keys()}

        self.perform_price_prediction(highest_other_bid)

    def get_next_state(self, current_state, won_this_round, price_this_round):
        s_ = current_state + ((int(won_this_round), price_this_round),)
        return s_

    def calc_transition_matrix(self):
        pass

    def calc_expected_rewards(self):
        """
        Calculate expected rewards using learned prices.
        """
        for s in self.state_space:
            for a in self.action_space:
                if s not in self.terminal_states:
                    self.R[(s, a)] = 0.0

        self.calc_terminal_state_rewards()

    def calc_terminal_state_rewards(self):
        """
        Calculate rewards for states corresponding to the end of an auction.
        """
        for s in self.terminal_states:
            for a in self.action_space:
                total_paid = 0.0
                num_won = 0
                for results in s:
                    num_won += results[0]
                    total_paid += results[1] * results[0]
                self.R[(s, a)] = sum(self.valuations[:num_won]) - total_paid

    def place_bid(self, current_round):
        """
        Places a bid based on what the bidder has learned.

        bid = argmax_a Q(s,a)

        :return: bid: Float.  The bid the bidder will place.
        """
        r = current_round - 1
        if self.use_given_pi:
            if self.bid_val_in_last_round and (current_round == self.num_rounds):
                bid = self.valuations[self.num_goods_won]
            else:
                if r > 0:
                    self.announced_price[r - 1] = round(self.announced_price[r - 1], self.digit_precision)
                s = ()
                for t in range(r):
                    s = s + ((int(self.win[t]), self.announced_price[t]),)
                bid = self.given_pi[tuple(self.valuations)][s]
        else:
            if self.bid_val_in_last_round and (current_round == self.num_rounds):
                bid = self.valuations[self.num_goods_won]
            else:
                if r > 0:
                    self.announced_price[r - 1] = round(self.announced_price[r - 1], self.digit_precision)
                s = ()
                for t in range(r):
                    s = s + ((int(self.win[t]), self.announced_price[t]),)
                bid = self.pi[s]
        self.bid[r] = bid
        return bid
