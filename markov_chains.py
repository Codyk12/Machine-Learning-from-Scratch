# markov_chains.py
"""Volume II: Markov Chains.
Cody Kesler
320
11/1/2018
"""

import numpy as np
from scipy.linalg import norm

# Problem 1
def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    A = np.random.random((n,n))
    # Make the columns sum to 1 by dividing by its sum
    return A / A.sum(axis=0)


# Problem 2
def forecast(days):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    # Set the initial prediction in the prediction array
    weather = [np.random.binomial(1, transition[0, 1])]

    for _ in range(days-1):
        # Calculate the probability of the previous day from a binomial dist given the matrix and add it to the predicted weather array
        weather.append(np.random.binomial(1, transition[1, weather[-1]]))

    # Sample from a binomial distribution to choose a new state.
    return weather


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    # Transition Matrix for days of hot, mild, cold, freezing
    transition = np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[0,.1,.2,.2]])
    # Set the initial prediction in the prediction array
    weather = [np.argmax(np.random.multinomial(1, transition[:,1]))]

    for _ in range(days-1):
        # Calculate the probability of the previous day from a multiomial dist given the matrix and add it to the predicted weather array
        weather.append(np.argmax(np.random.multinomial(1, transition[:, weather[-1]])))

    return weather


# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    # Normalize the vector
    x = np.random.random((A.shape[0], 1))
    x /= np.sum(x)
    for _ in range(N):
        # If the tolerance level is reached before the number of iterations then kick out
        if(norm(x - A @ x) < tol):
            return A @ x
        # Compute the next x by multipliying it by A
        x = A @ x

    raise ValueError("A does not converge!")


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        self.words = set()
        # Dictionary to hold words to their index in the matrix
        self.states = {}
        # Dictionary to hold index in the matrix to the word it represents
        self.opp_states = {}
        i = 0
        self.opp_states[i] = "$tart"
        self.states["$tart"] = i
        # Loop through the input file
        with open(filename) as file:
            text = file.read().split("\n")
            for line in text:
                for word in line.split(" "):
                    # If the word in unique append it to each dictionary and increase the index for the matrix
                    if(word not in self.words):
                        i += 1
                        self.states[word] = i
                        self.opp_states[i] = word
                        self.words.add(word)

        # Initialize the transition matrix for the given input
        A = np.zeros((len(self.words)+2, len(self.words)+2))
        # Add the Stop states into the dictionaries
        self.states["$top"] = len(self.words)+1
        self.opp_states[len(self.words)+1] = ""

        for i, line in enumerate(text):
            words = line.split(" ")
            # Loop through each word for each line in the input
            for j, word in enumerate(words):
                # If its the first word in the line put a 1 for the word index, start token index
                if j == 0:
                    A[self.states[word], self.states['$tart']] = 1.0
                # If its a word in the line add 1 for the word index, the word it came from index
                if(j < len(words)-1):
                    A[self.states[words[j + 1]], self.states[words[j]]] += 1.0
                # If its the last word in the line put a 1 for the stop token index, word it came from index
                else:
                    A[self.states['$top'], self.states[word]] = 1.0

        # Normalize the transition matrix
        self.A = A / A.sum(axis=0)


    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        # Initialize the word predictor list with a multinomial draw using the prob for the entire col of the start token
        words = [np.argmax(np.random.multinomial(1, self.A[:, 0]))]

        while words[-1] != self.states['$top']:
            # If the state is not the end token add the multimonial draw from the col of the previous word
            words.append(np.argmax(np.random.multinomial(1, self.A[:, words[-1]])))

        # Return the list of words corresponding to the predicted numbers
        return " ".join([self.opp_states[i] for i in words])
