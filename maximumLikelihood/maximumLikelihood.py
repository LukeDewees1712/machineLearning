import numpy as np
from matplotlib import pyplot as plt

def main():
    priors = [0.5,0.5]
    # calculate sample space
    #
    x = np.linspace(-2,2, 10000)
    # generate p1's probability
    #
    p1 = np.where(np.logical_and(x >= 0, x <= 1), 1, 0)
    # generate alpha values from -2 to 2
    #
    alphas = np.linspace(-2,2, 10000)
    oneIndices = np.where(p1==1)[0]
    probError = []
    # loop through alpha values and calculate p2, then find the overlap of p1 and p2 for every alpha
    #
    for alpha in alphas:
        # calculate lower and upper bound of p2's rectangle
        #
        lowerBound = alpha
        upperBound = alpha + 1
        # calculate p2's rectanlge
        #
        p2 = np.where(np.logical_and(x>=lowerBound, x <= upperBound), 1, 0)
        # find smallest prior
        #
        smallestPrior = min(priors)
        # find indices where p2 is 1
        #
        twoIndices = np.where(p2==1)[0]
        # find the number of points where p2 and p1 and both 1 (overlap)
        #
        overlapPoints = len(np.intersect1d(twoIndices, oneIndices))
        # calculate P(E)
        #
        pe = smallestPrior * (overlapPoints)/2500
        # add to array
        #
        probError.append(pe)
    plt.plot(x, probError)
    plt.show()
if __name__ == "__main__":
    main()
