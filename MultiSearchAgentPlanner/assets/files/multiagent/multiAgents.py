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
import sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        score=successorGameState.getScore()
        #To find the minimum distance between the ghost and pacman
        minGhostDist=sys.maxsize
        for ghostLoc in successorGameState.getGhostPositions():
            manhattanDistance=util.manhattanDistance(newPos,ghostLoc)
            minGhostDist=min(minGhostDist,manhattanDistance)
        #We add minGhostDist to score. Larger the value of score means being closer to a ghost has negative impact on the score.
        #Pacman chooses positions which are farther to the ghosts.
        score=score+minGhostDist
        
        #To find the least food distance to pacman
        minFoodDist=sys.maxsize
        for foodPos in newFood.asList():
            manhattanDistance=util.manhattanDistance(foodPos,newPos)
            minFoodDist=min(minFoodDist,manhattanDistance) 
        #To favor closer food we subtract minFoodDist(distance between pacman Position and Food Position) from score. Lesser the value of score means being closer to a Food has Positive impact on the score. 
        #Pacman chooses positions which are closer to the remaining food.
        score=score-minFoodDist

        #To make sure pacman does not stop
        score -= 10 if action==Directions.STOP else 0

        #If game state is winning state then high score is returned
        if(successorGameState.isWin()):
            return sys.maxsize
        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        #List of ghost agents, 0 excluded as it indicates pacman agent.
        ghostLst=list(range(1,gameState.getNumAgents()))

        #In minimax, maxValue finds the maximum value of the successors.
        def maxValue(gameState,depth):
            if(gameState.isWin() or gameState.isLose() or depth==self.depth):
                #If game state is winning or losing state or depth is reached, the evaluation function is returned.
                return self.evaluationFunction(gameState)
            v=-sys.maxsize 
            #To check legal actions of pacman agent
            for action in gameState.getLegalActions(0):
                successor=gameState.generateSuccessor(0,action)
                #calculates the minimum value among the possible outcomes of the ghost agents in the given successor state.
                #Pacman agent chooses the action with the highest possible minimum values among the actions ghosts can take.
                v=max(v,minValue(successor,depth,ghostLst[0]))
            return v
        
        #In minimax, minValue finds the minimum value of the successors.
        def minValue(gameState,depth,ghost):
            #If game state is winning or losing state or depth is reached, the evaluation function is returned.
            if(gameState.isWin() or gameState.isLose() or depth==self.depth):
                return self.evaluationFunction(gameState)
            v=sys.maxsize
            #To check legal actions of ghost agent.
            for action in gameState.getLegalActions(ghost):
                successor=gameState.generateSuccessor(ghost,action)
                #If the ghost variable is equal to last ghost in ghostLst then maxValue is calculated else minValue is calculated.
                if(ghost==ghostLst[-1]):
                    #Pacman agent chooses the action with the least possible maximun values among the actions.
                    v=min(v,maxValue(successor,depth+1))#Finds maxValue for next depth.
                else:
                    #Pacman agent chooses the action with the least minimum values among the actions ghosts can take.
                    v=min(v,minValue(successor,depth,ghost+1))#Finds minValue for same depth.
            return v

        actionResult=None
        value=-sys.maxsize 
        for action in gameState.getLegalActions(0): #Legal actions of pacman agent.
            successor=gameState.generateSuccessor(0,action)
            #To find min value of successors.
            v=minValue(successor,0,ghostLst[0])
            #This helps in selecting better outcome for Pacman.
            if(v>value):
                value=v
                actionResult=action
        return actionResult
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #List of ghost agents, 0 excluded as it indicates pacman agent.
        ghostLst=list(range(1,gameState.getNumAgents()))

        #In alpha-beta pruning, maxValue finds the maximum value of the successors and if beta<v then it helps to reduce the number of nodes that need to be explored.
        def maxValue(gameState,depth,alpha,beta):
            #If game state is winning or losing state or depth is reached, the evaluation function is returned.
            if(gameState.isWin() or gameState.isLose() or depth==0):
                return self.evaluationFunction(gameState)
            v=-sys.maxsize 
            legalActions=gameState.getLegalActions(0) #Legal Actions of pacman.
            for action in legalActions:
                successor=gameState.generateSuccessor(0,action)
                #Pacman agent chooses the action with the highest possible minimum values among the actions ghosts can take.
                v=max(v,minValue(successor,1,depth,alpha,beta))
                #If beta value is less then v(max value of its successors) then alpha-beta pruning takes place.
                if(v>beta):
                    return v
                #alpha value updated
                alpha=max(alpha,v)
            return v

        #In alpha-beta pruning, minValue finds the minimum value of the successors and if v<alpha then it helps to reduce the number of nodes that need to be explored.
        def minValue(gameState,ghost,depth,alpha,beta):
            #If game state is winning or losing state or depth is reached, the evaluation function is returned.
            if(gameState.isWin() or gameState.isLose() or depth==0):
                return self.evaluationFunction(gameState)
            v=sys.maxsize 
            legalActions=gameState.getLegalActions(ghost) #Legal Actions of ghost.
            for action in legalActions:
                successor=gameState.generateSuccessor(ghost,action)
                #If the ghost variable is equal to last ghost in ghostLst then maxValue is calculated else minValue is calculated.
                if(ghost==ghostLst[-1]):
                    #Pacman agent chooses the action with the least possible maximun values among the actions.
                    v=min(v,maxValue(successor,depth-1,alpha,beta))#Finds maxValue for next depth.
                else:
                    #Pacman agent chooses the action with the least minimum values among the actions ghosts can take.
                    v=min(v,minValue(successor,ghost+1,depth,alpha,beta))#Finds minValue for same depth.
                #If v(min value of its successors) is less then beta value then alpha-beta pruning takes place.
                if(v<alpha):
                    return v
                beta=min(beta,v)
            return v

        actionResult=None
        alpha=-sys.maxsize 
        beta=sys.maxsize 
        for action in gameState.getLegalActions(0): #Legal actions of pacman agent.
            successor=gameState.generateSuccessor(0,action)
            v=minValue(successor,ghostLst[0],self.depth,alpha,beta)
            #This helps in selecting better outcome for Pacman.
            if(v>alpha):
                alpha=v
                actionResult=action
        return actionResult
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #List of ghost agents, 0 excluded as it indicates pacman agent.
        ghostLst=list(range(1,gameState.getNumAgents()))

        #The maxValue function selects the action that maximizes the expected value, considering the actions of the ghosts.
        def maxValue(gameState,depth):
            #If game state is winning or losing state or depth is reached, the evaluation function is returned.
            if(gameState.isWin() or gameState.isLose() or depth == 0):
                return self.evaluationFunction(gameState)
            v=-sys.maxsize
            legalActions=gameState.getLegalActions(0) #Legal Actions of pacman.
            for action in legalActions:
                successor=gameState.generateSuccessor(0,action)
                #Pacman agent chooses the action with the highest expectimax values.
                v=max(v,expValue(successor,1,depth))
            return v

        #The expValue function calculates the expected value for the minimizing player, averaging over all possible outcomes ghost.
        def expValue(gameState,ghost,depth):
            #If game state is winning or losing state or depth is reached, the evaluation function is returned.
            if(gameState.isWin() or gameState.isLose() or depth == 0):
                return self.evaluationFunction(gameState)
            v=0 
            legalActions = gameState.getLegalActions(ghost)
            for action in legalActions:
                successor=gameState.generateSuccessor(ghost,action)
                #If the ghost variable is equal to last ghost in ghostLst then maxValue is calculated else expValue is calculated.
                if(ghost==ghostLst[-1]):
                    #To calculate the average value of the max values.
                    v+=(maxValue(successor,depth-1)/len(legalActions)) 
                else:
                    #To calculate the average value of the expected values.
                    v+=(expValue(successor,ghost+1,depth)/len(legalActions))
            return v

        actionResult=None
        value=-sys.maxsize
        for action in gameState.getLegalActions(0): #Legal actions of pacman agent.
            successor=gameState.generateSuccessor(0,action)
            v=expValue(successor,ghostLst[0],self.depth)
            #This helps in selecting better outcome for Pacman.
            if(v>value):
                value=v
                actionResult=action

        return actionResult
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This is better evaluation function as we even considered the ghost scared time and power pellets.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    capsules=currentGameState.getCapsules()

    #To find the least food distance to pacman
    minFoodDist=[]
    for foodPos in newFood.asList():
        manhattanDistance=util.manhattanDistance(newPos,foodPos)
        minFoodDist.append(manhattanDistance)
    if(len(minFoodDist)>0):
        #Pacman chooses positions which are closer to the remaining food.
        score=score+(1/(manhattanDistance))

    #Power pellets are remaining then penalty is given so that pacman will try to eat power pellets.
    score=score-10*len(capsules)

    #To find the minimum distance between the ghost and pacman
    for ghostLoc in newGhostStates:
        #Pacman chooses positions which are farther to the ghosts.
        minGhostDist=util.manhattanDistance(newPos,ghostLoc.getPosition())
        #If ghosts are scared then score is increaased
        if(ghostLoc.scaredTimer>0): 
            score=score+(1/(minGhostDist))
        else:
            #If ghosts are not scared score is decreased for pacman to avoid ghost
            score=score+((-10)/(minGhostDist+1))
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
