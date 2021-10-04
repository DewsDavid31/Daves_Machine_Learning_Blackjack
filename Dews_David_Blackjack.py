import numpy as np
import sklearn.linear_model as ln
import sklearn.naive_bayes as sk
import sklearn.preprocessing as pp

class Ensemble:
    """ Ensemble Class - A class that contains classifiers and scrubs single integer inputs
    to fit and predict as well as produce policy data about the classifiers"""
    def __init__(self, classifiers, is_partial_fit):
        """classifiers - List of non-polynomial classifier model objects\n
        is_partial_fit - Boolean list aligned with classifiers with boolean value for if classifier
         has partial_fit()"""
        self.classifiers = classifiers
        self.x_history = np.array(21).reshape(-1, 1)
        self.y_history = np.array(0).reshape(-1, 1)
        self.is_partial_fit = is_partial_fit
        self.answers = []
        self.init()

    def policy(self):
        """Policy - returns the current actions at each hand value 4 - 22 for each classifier"""
        for classifier in self.classifiers:
            print(f'for classifier {classifier}:')
            for i in range(4,22):
                if classifier.predict(np.array(i).reshape(-1,1)) == 1:
                    print(f'at {i} I HIT')
                else:
                    print(f'at {i} I STAY')



    def init(self):
        """init - fits any classifiers to starting sample values for partial_fit usage"""
        index = 0
        for classifier in self.classifiers:
            classifier.fit(self.x_history, self.y_history.ravel())

    def partial_fit(self, x_val, y_val):
        """partial_fit - calls partial fit or fit with the current value history accordingly"""
        index = 0
        x = np.array(x_val).reshape(-1, 1)
        y = np.array(y_val).reshape(-1, 1)
        for classifier in self.classifiers:
            np.append(self.x_history, np.array(x_val).reshape(-1, 1))
            np.append(self.y_history, np.array(y_val).reshape(-1, 1))
            if self.is_partial_fit[index]:
                classifier.partial_fit(np.array(x_val).reshape(-1, 1), np.array(y_val).reshape(-1, 1).ravel())
            else:
                classifier.fit(self.x_history, self.y_history)
            index += 1

    def majority(self, list):
        """majority - checks the action of each classifier from input list and returns the most voted option"""
        """list - List of 0 or 1 from other helper method"""
        hits = 0
        stays = 0
        for action in list:
            if action == 0:
                stays += 1
            else:
                hits += 1
        if hits > stays:
            return 1
        else:
            return 0

    def predict(self, x_val):
        """predict - Calls predict of classifier using reformatted single integer into numpy array, places into result
         list contained in ensemble, then calls a majority vote on results, returning the winner"""
        results = []
        index = 0
        x = np.array(x_val).reshape(-1, 1)
        for classifier in self.classifiers:
            results.append(classifier.predict(x))
            index += 1
        self.answers.append(results)
        return self.majority(results)


class Flat_Average_Classifier:
    """Flat_Average_Classifier - Not a classifier per-say, but a set of average ratios of a hit or stay in 1 or 0 form
     that are modified based on wins, uses all standard classifier calls"""
    def __init__(self, policies):
        self.n = policies
        self.policy_totals = []
        self.hit_totals = []
        self.policy = []
        for i in range(self.n):
            self.policy.append(0.5)
            self.hit_totals.append(0)
            self.policy_totals.append(0)

    def predict(self, x_val):
        """predict - returns current predicted action at given hand value, if average wins when hitting > 0.5, it hits
        and vice-versa"""
        index = x_val.item(0)
        if index >= self.n:
            return
        if self.policy[index] >= 0.5:
            return 1
        else:
            return 0

    def fit(self, x_val, y_val):
        """fit - overloaded fit method, since fitting works the same as partial here\n
        x_val - numpy array\n
        y_val - numpy array"""
        self.partial_fit(x_val, y_val)

    def partial_fit(self, x_val, y_val):
        """partial_fit - standard partial fit classifier method, will extract numpy values and adjust action list
        AKA policy list according to average correct hits/stays\n
        x_val - numpy array of hand value\n
        y_val - numpy array of 1 for hit, 0 for stay"""
        index = x_val.item(0)
        answer = y_val.item(0)
        index = int(index)
        if index >= self.n:
            return
        self.policy_totals[index] += 1
        self.hit_totals[index] += answer
        self.policy[index] = self.hit_totals[index] / self.policy_totals[index]


class Poly_Classifier:
    """Poly_Classifier  - A wrapper class for any classifier that needs polynomial features on its input"""
    def __init__(self, classifier, degree):
        """classifier - classifier model object that MUST have partial_fit()\n
        degree - integer value > 0, the degree of polynomial transform"""
        self.classif = classifier
        self.classif.fit(pp.PolynomialFeatures(degree=degree).fit_transform(np.array(21).reshape(-1, 1)),
                         pp.PolynomialFeatures(degree=degree).fit_transform(np.array(0).reshape(-1, 1)))
        self.x_history = np.array(21).reshape(-1, 1)
        self.y_history = np.array(0).reshape(-1, 1)
        self.pfeat = pp.PolynomialFeatures(degree=degree)
        self.pfeaty = pp.PolynomialFeatures(degree=degree)

    def predict(self, x_val):
        """predict - standard predict function of classifier, but done with poly transformed data"""
        x = self.pfeat.fit_transform(np.array(x_val).reshape(-1, 1))
        self.classif.predict(x)

    def partial_fit(self, x_val, y_val):
        """partial_fit - standard partial fit method on integer hand value\n
        x_val - integer value\n
        y_val - 0 for stay, 1 for hit\n"""
        np.append(self.x_history, np.array(x_val).reshape(-1, 1))
        np.append(self.y_history, np.array(y_val).reshape(-1, 1))
        x = np.array(self.pfeat.fit_transform(self.x_history))
        y = np.array(self.pfeat.fit_transform(self.y_history))
        self.classif.fit(x, y)

    def fit(self, x_val, y_val):
        """fit - overloaded fit method for calling partial_fit when calling fit()\n
        x_val  - integer hand value\n
        y_val - 0 for stay 1 for hit"""
        self.partial_fit(x_val, y_val)


class matchbox:
    """matchbox - class for individual shaken matchbox from vsauce\'s matchbox computer, can act, produce its policy\n
    punish and reward itself"""
    def __init__(self):
        #begins with equal probability of pulling a hit or stay action from a matchbox
        self.hit_prob = 0.5
        self.stay_prob = 0.5

    def policy(self):
        """policy - returns current majority action at this matchbox\'s hand value"""
        if self.hit_prob > 0.5:
            return 1
        else:
            return 0

    def act(self):
        """act - produces random action 0 or 1 with current probability of 1"""
        policy = np.random.binomial(1, self.hit_prob)
        return policy

    def punish(self, action):
        """punish - changes the probability distribution by 10% based away from the the given action\n
        action - 0 or 1 for stay or hit respectively, punishing using that action"""
        if action == 1:
            self.hit_prob *= 0.9
            self.stay_prob = abs(1 - self.hit_prob)
        else:
            self.stay_prob *= 0.9
            self.hit_prob = abs(1 - self.stay_prob)

    def reward(self, action):
        """reward - increasing the probability of an action by 10% using punish in reverse\n
        action - 0 or 1 for stay or hit respectively, making that action more likely"""
        self.punish(abs(action - 1))


class Matchbox_classifier:
    """Matchbox_classifier - class I created modeling the matchbox computer from vsauce,\n
     has a matchbox for each hand value that by remove bad actions from each matchbox statistially\n
     matchboxes - > 0 integer of matchbox classes to be used, for blackjack use 21"""
    def __init__(self, matchboxes):
        self.n = matchboxes
        self.matchboxes = []
        for i in range(matchboxes):
            self.matchboxes.append(matchbox())

    def policy(self, n):
        """policy - returns the current best action that each matchbox has found\n
        n - integer that must equal the number of matchboxes"""
        return self.matchboxes[n].policy()

    def predict(self, x_val):
        """predict - standard classifier predict method, returns predicted result\n
        x_val - numpy array of integers < number of matchboxes"""
        index = x_val.item(0)
        if index >= self.n:
            return
        return self.matchboxes[index].act()

    def fit(self, x_val, y_val):
        """fit - overloaded partial_fit method for calling fit, since both are the same for this class"""
        self.partial_fit(x_val, y_val)

    def partial_fit(self, x_val, y_val):
        """partial_fit - standard classifier partial_fit method, rewarding/punishing\n
         on predicted action versus correct one given\n
         x_val - numpy array of integer of hand value\n
         y_val - numpy array of integer 0 or 1 for stay or hit respectively"""
        index = x_val.item(0)
        index = int(index)
        answer = y_val.item(0)
        if index >= self.n:
            return
        action = self.matchboxes[index].act()
        if action is not answer:
            self.matchboxes[index].punish(action)
        else:
            self.matchboxes[index].reward(action)


# From visualstudomagazine.com, Dr. JAmes MacCaffrey
class Thomas_sampler:
    """Thomas_sampler - A model-less reinforcement classifier explained in visualstudiomagazine.com\n
     predicts true or false on distributions skewed by rewards, uses losses and wins directly to reshape distribution\n
    unlike matchboxing or naive bayes\n
    n - number of distributions, set to 21 for blackjack"""
    def __init__(self, n):
        self.n = n
        self.means = np.zeros(self.n).fill(0.5)
        self.probs = np.zeros(self.n)
        self.S = np.zeros(self.n, dtype=int)
        self.F = np.zeros(self.n, dtype=int)
        self.rnd = np.random.RandomState(7)
        self.policy = [abs(np.random.logistic())]
        for i in range(self.n):
            self.policy.append(abs(np.random.binomial(0, 1)))

    def fit(self,x_val,y_val):
        """fit - overloaded partial_fit method since both are the same for this classifier\n
        x_val - numpy array of integer for hand value\n
        y_val - numpy array of 0 or 1 for stay or hit respectively"""
        self.partial_fit(x_val, y_val)

    def partial_fit(self, x_val, y_val):
        """partial_fit - standard partial fit method, modifies probabilities given on wins/losses\n
        x_val - numpy array of integer for hand value\n
        y_val - numpy array of 0 or 1 for stay or hit respectively"""
        index = x_val.item(0)
        answer = y_val.item(0)
        if index >= self.n:
            return
        self.probs[index] = self.rnd.beta(self.S[index] + 1, self.F[index] + 1)
        if self.probs[index] >= 0.5:
            self.policy[index] = 1
        if self.probs[index] < 0.5:
            self.policy[index] = 0
        p = index
        if self.predict(x_val) == answer:
            self.S[index] += 1
        else:
            # lost
            self.F[index] += 1
            if self.policy[index] == 1:
                self.policy[index] = 0
            else:
                self.policy[index] = 1

    def predict(self, x_val):
        """predict - standard prediction method, predicts hit or stay on given hand value\n
        x_val - numpy array of integer of hand value"""
        index = x_val.item(0)
        if index > 21:
            return 0
        return self.policy[index]


class AI_Player:
    """AI_Player - Machine learning adapter class for when a classifier controls the player, keeps a classifier\n
     and game history\n
     classifier - any non polynomial and partial_fit compatible classifier, set by default by deck class"""
    def __init__(self, classifier):
        self.win_rate = 0.0
        self.wins = 0.0
        self.loses = 0.0
        self.games = 0.0
        self.classifier = classifier
        self.hand_value = 0
        self.player_lost = False
        self.dealer_lost = False
        self.move = 0
        self.draws = 0
        self.history = [21, 2]
        self.targets = [0, 1]

    def look(self, hand):
        """look - player method to count the total hand value of cards\n
        hand - list of card objects"""
        self.hand_value = 0
        for card in hand:
            self.hand_value += card.value

    def take_turn_non_verbose(self, hand):
        """take_turn_non_verbose - non printed version of turn algorithm,\n
         looks at hand and returns string of next action"""
        self.look(hand)
        hit = self.predict_non_verbose()
        if hit == [1]:
            return 'hit'
        else:
            return 'stay'

    def take_turn(self, hand):
        """take_turn - printed version of turn alorithm,\n
        looks at hand and returns string of next action\n
        hand - List of card objects"""
        self.look(hand)
        hit = self.predict()
        if hit == [1]:
            return 'hit'
        else:
            return 'stay'

    def predict_non_verbose(self):
        """predict_non_verbose - printless version of predicting public version\n
         of predict method of player\'s classifier predict method"""
        hit = self.classifier.predict(self.hand_value)
        return hit

    def predict(self):
        """predict - predicting public version\n
            of predict method of player\'s classifier predict method"""
        print(self.hand_value)
        hit = self.classifier.predict(self.hand_value)
        if hit == [1]:
            print(f'AI: I want to hit (%{self.win_rate * 100:.3})')
        elif hit == [0]:
            print(f'AI: I want to stay (%{self.win_rate * 100:.3})')
        return hit

    def punish_non_verbose(self):
        """punish_non_verbose - non-verbose version of applying rewards to classifiers based\n
         on if hit or stay busted the hand"""
        hit = self.move
        if hit == 1:
            if self.player_lost:
                self.classifier.partial_fit(self.hand_value, 0)
            elif self.dealer_lost:
                self.classifier.partial_fit(self.hand_value, 1)
            else:
                self.classifier.partial_fit(self.hand_value, 1)
        else:
            if self.player_lost:
                self.classifier.partial_fit(self.hand_value, 1)
            elif self.dealer_lost:
                self.classifier.partial_fit(self.hand_value, 0)
            else:
                self.classifier.partial_fit(self.hand_value, 0)

    def punish(self):
        """punish_non_verbose - non-verbose version of applying rewards to classifiers based\n
            on if hit or stay busted the hand"""
        hit = self.move
        if hit == 1:
            if self.player_lost:
                print('AI: I was wrong')
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(0).ravel())
            elif self.dealer_lost:
                print('AI: I was right')
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(1).ravel())
            else:
                print('AI: going steady')
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(1).ravel())
        else:
            if self.player_lost:
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(1).ravel())
                print('AI: I was wrong')
            elif self.dealer_lost:
                print('AI: I was right')
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(0).ravel())
            else:
                print('AI: going steady')
                self.classifier.partial_fit(np.array(self.hand_value).reshape(-1, 1), np.array(0).ravel())

    def my_rules(self):
        """my_rules - method that prints prediction of every hand value from 4 to 21 in stay or hit form"""
        for value in range(4, 21):
            move = self.classifier.predict(np.array(value))
            move_str = 'Stay'
            if move == 1:
                move_str = 'Hit'
            print(f'at value of {value} I {move_str}')
        self.win_rate = self.wins / self.games
        print(f'my win rate is: {self.win_rate}')
        print(
            f'i played: {self.games} games dealer won {self.loses} times, I won {self.wins} times, draw {self.draws} times')


class Card:
    """Card - object that stores the value and suite of a card\n
    value - integer value of card in blackjack form\n
    suite - integer encoding of suite 0 - 3"""
    def __init__(self, value, suite):
        self.value = value
        self.suite = suite


class Deck:
    """Deck - Deck object containing all cards, contains a suite of methods for game algorithms,\n
     discard piles, hands and data\n
     classifier - classifier or ensemble that supports partial_fit and supports partial_fit\n
     train_threshold - float or double such that 1.0 > train_threshold > 0.0, for the sake of halting,\n
      use value < 0.3, the win-rate at which a machine learning player is acceptable for making predictions"""
    def __init__(self, classifier, train_threshold):
        self.player_wins = False
        self.player_loses = False
        self.game_over = False
        self.moron_mode = False
        classifier = classifier
        self.ai = AI_Player(classifier)
        self.suites = ['♤', '♥', '♧', '♦']
        self.cards = []
        self.__array_of_52__()
        self.dealer_hand = []
        self.player_hand = []
        self.player_played = []
        self.dealer_played = []
        self.discard = []
        self.player_passes = False
        self.dealer_passes = False
        self.turn = True
        self.shuffle_non_verbose()
        self.ai_mode = False
        self.load = 0
        self.thresh = train_threshold
        while self.ai.win_rate < self.thresh:
            self.start_bj_non_verbose(2000)
            print(self.ai.win_rate)
            self.ai.wins = 0
            self.ai.loses = 0
            self.ai.games = 0
            self.ai.draws = 0

    def need_shuffle(self):
        """need_shuffle - method to check if deck too small to deal, returns boolean"""
        if len(self.cards) <= 1:
            return True

    def draw_non_verbose(self, dealer, num):
        """draw_non_verbose - non-printed version of method to draw card objects and put into a hand. Actually randomly\n
        selects a card instead of shuffling and drawing from top for speed, removes and appends to list needed\n
        dealer - boolean on if the dealer is drawing, otherwise goes to player\n
        num - number of cards to draw"""
        if self.need_shuffle():
            self.shuffle_non_verbose()
        for n in range(0, num):
            selected = np.random.randint(len(self.cards))
            drawn = self.cards[selected]
            self.cards.remove(drawn)
            if dealer:
                self.dealer_hand.append(drawn)
            else:
                self.player_hand.append(drawn)
        if not dealer:
            self.ai.hand_value = self.__bj__hand_score__(False)

    def draw(self, dealer, num):
        """draw - printed version of method to draw card objects and put into a hand. Actually randomly\n
        selects a card instead of shuffling and drawing from top for speed, removes and appends to list needed\n
        dealer - boolean on if the dealer is drawing, otherwise goes to player\n
        num - number of cards to draw"""
        if self.need_shuffle():
            self.shuffle()
        if dealer:
            print(f'-Dealer draws {num} cards-')

        if not dealer:
            print(f'-Player draws {num} cards-')

        for n in range(0, num):
            selected = np.random.randint(len(self.cards))
            drawn = self.cards[selected]
            self.cards.remove(drawn)
            if dealer:
                self.dealer_hand.append(drawn)
            else:
                self.player_hand.append(drawn)
        if not dealer:
            self.ai.hand_value = self.__bj__hand_score__(False)

    def shuffle_non_verbose(self):
        """shuffle_non_verbose - non-printed version that shuffles by removing all hand lists and placing back into deck,\n
         discard not needed or supported yet"""
        for card in self.player_hand:
            self.cards.append(card)
        for card in self.dealer_hand:
            self.cards.append(card)
        self.player_hand = []
        self.dealer_hand = []

    def shuffle(self):
        """shuffle_non_verbose - shuffles by removing all hand lists and placing back into deck,\n
         discard not needed or supported yet"""
        print('Shuffling deck...')
        self.show_hand(True)
        self.show_hand(False)
        for card in self.player_hand:
            self.cards.append(card)
        for card in self.dealer_hand:
            self.cards.append(card)
        self.player_hand = []
        self.dealer_hand = []

    def __print_card__(self, card):
        """__print_card__ - private method that prints the picture version of a card\n
        card - given card object"""
        print(f' {card.suite}{card.value} ')

    def __bj__hand_score__(self, dealer):
        """__bj_hand_score__ - private method that tallies the blackjack value of a hand and returns that as integer\n
        dealer - boolean if dealer\'s hand is being read, if not player is used"""
        score = 0
        if dealer:
            for card in self.dealer_hand:
                # TODO: aces don't necessarily work this way
                if card.value == 1 and score + 11 <= 21:
                    score += 11
                elif card.value > 10:
                    score += 10
                else:
                    score += card.value
            return score
        else:
            for card in self.player_hand:
                if card.value > 10:
                    score += 10
                else:
                    score += card.value
            return score

    def bust_non_verbose(self, dealer):
        """bust_non_verbose - non-printed version of bust call, called when bust is found\n
        dealer - boolean if dealer busted, if false then player busted"""
        self.game_over = True
        if dealer:
            self.win_non_verbose(not dealer)
        else:
            self.win_non_verbose(not dealer)

    def bust(self, dealer):
        """bust - bust call, called when bust is found\n
            dealer - boolean if dealer busted, if false then player busted"""
        self.game_over = True
        if dealer:
            print('-Dealer busts-')
            self.show_hand(True)
            self.win(not dealer)
        else:
            print('-player busts-')
            self.show_hand(False)
            self.win(not dealer)

    def win_non_verbose(self, dealer):
        """win_non_verbose - non verbose version of win call for when a win is declared, flagging who lost and tallying wins/losses\n
        dealer - boolean signifying if dealer won, if false then player won"""
        self.game_over = True
        if dealer:
            self.ai.loses += 1
            self.player_loses = True
            if self.ai_mode:
                self.ai.player_lost = self.player_loses
                self.ai.dealer_lost = not self.player_loses
                self.ai.punish_non_verbose()
        else:
            self.ai.wins += 1
            self.player_wins = True
            self.player_loses = False
            if self.ai_mode:
                self.ai.player_lost = self.player_loses
                self.ai.dealer_lost = not self.player_loses
                self.ai.punish_non_verbose()

    def win(self, dealer):
        """win - win call for when a win is declared, flagging who lost and tallying wins/losses\n
        dealer - boolean signifying if dealer won, if false then player won"""
        self.game_over = True
        if dealer:
            self.ai.loses += 1
            print('-Dealer wins-')
            self.player_loses = True
            if self.ai_mode:
                self.ai.player_lost = self.player_loses
                self.ai.dealer_lost = not self.player_loses
                self.ai.punish()
            self.show_hand(True)
        else:
            print('-player wins-')
            self.ai.wins += 1
            self.player_wins = True
            self.player_loses = False
            if self.ai_mode:
                self.ai.player_lost = self.player_loses
                self.ai.dealer_lost = not self.player_loses
                self.ai.punish()
            self.show_hand(False)

    def show_hand(self, dealer):
        """show_hand - method for displaying hand in non-encoded, picture form\n
        dealer - boolean if dealer hand is shown, if false shows player hand"""
        hand = ''
        parsed_value = ''
        if dealer:
            print('Dealers Hand: ')
            for card in self.dealer_hand:
                if card.value == 11:
                    parsed_value = 'J'
                if card.value == 12:
                    parsed_value = 'Q'
                if card.value == 13:
                    parsed_value = 'K'
                if card.value < 11:
                    parsed_value = card.value
                hand += card.suite + '' + str(parsed_value) + ' '
        if not dealer:
            print('Players Hand: ')
            for card in self.player_hand:
                if card.value == 11:
                    parsed_value = 'J'
                if card.value == 12:
                    parsed_value = 'Q'
                if card.value == 13:
                    parsed_value = 'K'
                if card.value < 11:
                    parsed_value = card.value
                hand += card.suite + '' + str(parsed_value) + ' '
        print(hand)

    def bj_hit(self, dealer):
        """bj_hit - blackjack hit method, draws a card for given player and checks for busting, then shows hand\n
        dealer - boolean if dealer, if false then player is hit"""
        self.draw(dealer, 1)
        score = self.__bj__hand_score__(dealer)
        if score > 21:
            self.bust(dealer)
        else:
            self.show_hand(dealer)

    def bj_hit_non_verbose(self, dealer):
        """bj_hit_non_verbose - non-printed blackjack hit method, draws a card for given player and checks for busting, then shows hand\n
        dealer - boolean if dealer, if false then player is hit"""
        self.draw_non_verbose(dealer, 1)
        score = self.__bj__hand_score__(dealer)
        if score > 21:
            self.bust_non_verbose(dealer)

    def bj_fold(self, dealer):
        """bj_fold - method for folding, currently acts like a stay until fold support is needed\n
        dealer - boolean if dealer, if false player folds"""
        if dealer:
            print('-Dealer folds-')
            self.bj_best_play()
        else:
            print('-player folds-')
            self.bj_best_play()

    def bj_fold_non_verbose(self, dealer):
        """bj_fold_non_verbose - non-printed method for folding, currently acts like a stay until fold support is needed\n
                dealer - boolean if dealer, if false player folds"""
        if dealer:
            self.bj_best_play_non_verbose()
        else:
            self.bj_best_play_non_verbose()

    def __array_of_52__(self):
        """__array_of_52__ - private helper method that creates a standard 52 card set and builds card classes of them"""
        for index in range(1, 13):
            for st in self.suites:
                self.cards.append(Card(index, st))

    def bj_stay(self, dealer):
        """bj_stay  -method for when a player stays, passes turn\n
        dealer - boolean if dealer, if false then player stays"""
        if dealer:
            print('-Dealer Stays-')
            self.dealer_passes = True
        else:
            print('-Player Stays-')
            self.player_passes = True

    def bj_stay_non_verbose(self, dealer):
        """bj_stay_non_verbose  -non-printed method for when a player stays, passes turn\n
        dealer - boolean if dealer, if false then player stays"""
        if dealer:
            self.dealer_passes = True
        else:
            self.player_passes = True

    def declare_draw(self):
        """declare_draw - method called when a draw is determined, shows hands and applies 21 rule\n
         if a player has 21 wins by least cards"""
        dealer_size = len(self.dealer_hand)
        player_size = len(self.player_hand)
        player = self.__bj__hand_score__(False)
        dealer = self.__bj__hand_score__(True)
        self.show_hand(True)
        self.show_hand(False)
        if player == dealer:
            if dealer == 21:
                if player_size < dealer_size:
                    self.win(False)
                if player_size > dealer_size:
                    self.win(True)
                else:
                    print('-Game is a draw-')
                    self.game_over = True
                    self.ai.draws += 1
            else:
                print('-Game is a draw-')
                self.game_over = True
                self.ai.draws += 1
        if player > dealer:
            self.win(False)
        if dealer > player:
            self.win(True)

    def declare_draw_non_verbose(self):
        """declare_draw_non_verbose - non-printed method called when a draw is determined, shows hands and applies 21\n
         rule if a player has 21 wins by least cards"""
        dealer_size = len(self.dealer_hand)
        player_size = len(self.player_hand)
        player = self.__bj__hand_score__(False)
        dealer = self.__bj__hand_score__(True)
        if player == dealer:
            if dealer == 21:
                if player_size < dealer_size:
                    self.win_non_verbose(False)
                if player_size > dealer_size:
                    self.win_non_verbose(True)
                else:
                    self.game_over = True
            else:
                self.game_over = True
        if player > dealer:
            self.win_non_verbose(False)
        if dealer > player:
            self.win_non_verbose(True)

    def bj_best_play(self):
        """bj_best_play - method for determining which player wins when both pass turn or stay"""
        dealer = self.__bj__hand_score__(True)
        player = self.__bj__hand_score__(False)
        if player > dealer:
            self.win(False)
        elif dealer > player:
            self.win(True)
        else:
            self.declare_draw()

    def bj_best_play_non_verbose(self):
        """bj_best_play_non_verbose - non-printed method for determining which player wins when both pass turn or stay"""
        dealer = self.__bj__hand_score__(True)
        player = self.__bj__hand_score__(False)
        if player > dealer:
            self.win_non_verbose(False)
        elif dealer > player:
            self.win_non_verbose(True)
        else:
            self.declare_draw_non_verbose()

    def dealer_turn(self):
        """dealer_turn - algorithm for dealer moves, hits until 18, stays otherwise"""
        score = self.__bj__hand_score__(True)
        if score >= 17:
            self.bj_stay(True)
            self.turn = not self.turn
        else:
            self.bj_hit(True)
            self.turn = not self.turn

    def dealer_turn_non_verbose(self):
        """dealer_turn_non_verbose - algorithm for dealer moves that calls printless methods,\n
         hits until 18, stays otherwise"""
        score = self.__bj__hand_score__(True)
        if score >= 17:
            self.bj_stay_non_verbose(True)
            self.turn = not self.turn
        else:
            self.bj_hit_non_verbose(True)
            self.turn = not self.turn

    def player_turn_non_verbose(self):
        """player_turn_non_verbose - non-printed adapter method for machine learning player to take turn and end turn"""
        decision = self.ai.take_turn_non_verbose(self.player_hand)
        if decision == 'hit':
            self.bj_hit_non_verbose(False)
            self.turn = not self.turn
        else:
            self.bj_stay_non_verbose(False)
            self.turn = not self.turn

    def player_turn(self):
        """player_turn - adapter method for machine learning player or human player to take turn and end turn\n
        moron_mode - if moron mode is set using set_moron_mode() this turn is decided randomly"""
        decision = '?'
        if self.ai_mode:
            self.show_hand(False)
            if not self.moron_mode:
                decision = self.ai.take_turn(self.player_hand)
            if self.moron_mode:
                d_number = np.random.choice([1,2], 1)
                if d_number == [1]:
                    decision = 'hit'
                else:
                    decision = 'stay'

            if decision == 'hit':
                self.bj_hit(False)
                self.turn = not self.turn
            else:
                self.bj_stay(False)
                self.turn = not self.turn

        else:
            self.show_hand(False)
            if self.suggestions:
                suggested = self.ai.predict()
                sugg_str = '?'
                if suggested == 1:
                    sugg_str = 'hit'
                else:
                    sugg_str = 'stay'
                print(f'AI: I would {sugg_str}({self.ai.win_rate * 100}%)')

            valid_n = ['no', 'n', 'N', 'No']
            decision = input('Hit (Y/n)? (folding = fold):')
            if decision == 'fold':
                self.bj_fold(False)
            elif decision not in valid_n:
                self.bj_hit(False)
                self.turn = not self.turn
            else:
                self.bj_stay(False)
                self.turn = not self.turn

    def set_moron_mode(self):
        """set_moron_mode - sets moron flag to make all decisions random for accuracy testing purposes\n
        note: I chose the word moron_mode because of an inside joke with a pal of mine, it stuck"""
        self.moron_mode = True

    def start_bj(self, matches, ai, suggestions):
        """start_bj - starts a blackjack game with current settings in arguments\n
        matches - number of matches until game is over\n
        ai - boolean on if machine learning player plays instead of human, false = human is player\n
        suggestions - boolean on if machine learning suggestions should be turned on"""
        self.ai_mode = ai
        self.shuffle()
        for match in range(0, matches):
            self.shuffle()
            self.player_passes = False
            self.dealer_passes = False
            self.player_wins = False
            self.ai.player_lost = False
            self.player_loses = False
            self.ai.dealer_lost = False
            self.game_over = False
            self.draw(True, 2)
            self.draw(False, 2)
            self.show_hand(True)
            self.show_hand(False)
            self.suggestions = suggestions
            while not self.game_over:
                if self.player_passes and self.dealer_passes:
                    self.bj_best_play()
                elif self.turn:
                    self.dealer_turn()
                else:
                    self.player_turn()

            self.ai.games += 1

    def start_bj_non_verbose(self, matches):
        """start_bj_non_verbose - starts a computer-played blackjack game\n
                matches - number of matches until game is over\n"""
        self.ai_mode = True
        print('loading ai: [', end='')
        self.shuffle_non_verbose()
        for match in range(0, matches):
            self.shuffle_non_verbose()
            if match % 200 == 0:
                print("[]", end='')
            self.player_passes = False
            self.dealer_passes = False
            self.player_wins = False
            self.ai.player_lost = False
            self.player_loses = False
            self.ai.dealer_lost = False
            self.game_over = False
            self.draw_non_verbose(True, 2)
            self.draw_non_verbose(False, 2)
            while not self.game_over:
                if self.player_passes and self.dealer_passes:
                    self.bj_best_play_non_verbose()
                elif self.turn:
                    self.dealer_turn_non_verbose()
                else:
                    self.player_turn_non_verbose()
            self.ai.games += 1
            self.ai.win_rate = self.ai.wins / self.ai.games
        if self.ai.win_rate < self.thresh:
            print('] INACCURATE')
        else:
            print('] DONE')


def main():
    # main method for demo

    train_threshold = 0.3
    #how accurate of a machine learning player you need, refitting until it hits that value
    #WARNING: value above 0.3 become near-impossible, could take days

    classifiers = [
        Thomas_sampler(21),
        sk.BernoulliNB(),
        Poly_Classifier(ln.LinearRegression(),2),
        Matchbox_classifier(21),
        Flat_Average_Classifier(21)
        ]
    #list of classifiers inserted into an ensemble to be used in machine learning player
    #Warning: only use classifiers with partial_fit() supported and no polynomial features
    #if is_partial_fit is true in the same index of the list below.
    #use Poly_classifier(<your classifier>) if using polynomial feature pipeline is needed

    ensemble = Ensemble(classifiers, [True, True, False, False, True])
    #ensemble for machine learning player, Ensemble(<classifier list>, <list of booleans for if partial_fit is used>)

    test_deck = Deck(ensemble, train_threshold)
    #Deck object

    test_deck.start_bj(1000, True, False)
    #Starts blackjack game:
    #start_bj(<number of deals>, <boolean: machine learning plays for you?>, <boolean: should machine learning give
    #you suggestions?>)

    test_deck.ai.my_rules()
    #Prints the current policy/strategy found


if __name__ == '__main__':
    main()
