# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Start from the game score of the successor, then add features on top of it
        score = successorGameState.getScore()

        # Closer food is better
        foodList = newFood.asList()
        if foodList:
            closestFood = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / closestFood

        # Avoid ghosts that are not scared. A scared ghost is not dangerous, so we ignore it
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if scaredTime == 0 and ghostDistance < 2:
                score -= 500

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Pacman is agent 0 and is the max player. We pick the action with the highest minimax value
        # The first ghost (agent 1) moves next, and we are still at depth 0
        bestValue = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.minimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
    
    def minimax(self, state: GameState, agentIndex, depth):
        # Leaves are won/lost states or states where we reached the depth limit. We score them with self.evaluationFunction
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        # Agents move in order. After the last ghost it is Pacman's turn again, and that completes one ply, so depth + 1
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        values = []
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            values.append(self.minimax(successor, nextAgent, nextDepth))

        # Pacman maximizes, every ghost minimizes
        if agentIndex == 0:
            return max(values)
        return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Same as minimax, but now alpha is updated at the root too, so later actions can be removed
        alpha = float("-inf")
        beta = float("inf")
        bestValue = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.alphaBeta(successor, 1, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

    def alphaBeta(self, state: GameState, agentIndex, depth, alpha, beta):
        # Leaves are won/lost states or states where we reached the depth limit
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        if agentIndex == 0:
            # Max player. If value is already bigger than beta, the min player above will never let us get here, so we stop
            value = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value = max(value, self.alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        # Min player (every ghost). If value is already smaller than alpha, the max player above will never let us get here, so we stop
        value = float("inf")
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value = min(value, self.alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Same as minimax. Pacman is the max player and picks the action with the highest expectimax value
        bestValue = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.expectimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
    
    def expectimax(self, state: GameState, agentIndex, depth):
        # Leaves are won/lost states or states where we reached the depth limit
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        values = []
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            values.append(self.expectimax(successor, nextAgent, nextDepth))

        # Pacman maximizes. Ghosts choose uniformly at random, so their value is the average of their children
        if agentIndex == 0:
            return max(values)
        return sum(values) / len(values)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    # Start from the game score of the state, then add features on top of it
    score = currentGameState.getScore()

    # Closer food is better. Less food and less capsules left is better too
    if foodList:
        closestFood = min(manhattanDistance(position, food) for food in foodList)
        score += 10.0 / closestFood
    score -= 4 * len(foodList)
    score -= 20 * len(capsules)

    # Scared ghosts are worth chasing. Ghosts that are not scared are dangerous when they are close
    for ghostState in ghostStates:
        ghostDistance = manhattanDistance(position, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            score += 200.0 / (ghostDistance + 1)
        elif ghostDistance < 2:
            score -= 500

    return score

# Abbreviation
better = betterEvaluationFunction
