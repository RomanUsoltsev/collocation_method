from solveFun import np
from solveFun import mt
import matplotlib.pyplot as plt


def showData(a, b, X, Y, index, title=None, xlabel=None, ylabel=None,
             xticks=None, yticks=None, typePoint=None):
    splot = plt.subplot(a, b, index)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.rcParams['font.size'] = '14'

    if (typePoint == None):
        plt.plot(X, Y)
    else:
        plt.plot(X, Y, marker=typePoint, label=title)
    plt.grid(True)
    plt.legend()

    if (xticks != None):
        plt.xticks(np.arange(min(X), max(X), xticks))
    if (yticks != None):
        plt.yticks(np.arange(min(Y), max(Y), yticks))

    return splot


def monomPoly(x, N):
    ans = np.zeros(shape=N + 1)
    for i in range(ans.shape[0]):
        ans[i] = mt.pow(x, i)

    return ans


def getValueCollocFun(xArr, coeffArr, N, K, x0, xl):
    length = abs(xl - x0)
    h = length / (2 * K)
    xc = 2 * h
    coeff = np.zeros(shape=(coeffArr.shape[0] + N + 1))
    coeff[0: K * (N + 1)] = np.transpose(coeffArr)[0]
    coeff[K * (N + 1): K * 2 * (N + 1)] = coeff[(K - 1) * (N + 1): K * (N + 1)]

    step = length / K
    stepArr = np.zeros(shape=K + 1)
    stepArr[0] = step
    for i in range(1, stepArr.shape[0] - 1):
        stepArr[i] += stepArr[i - 1] + step
    stepArr[K] = xl

    yVector = np.zeros(shape=xArr.shape)

    for i in range(xArr.shape[0]):
        j = 0
        col = 0
        x = xArr[i]
        while (x > step):
            x -= step
            j += 1
            col += N + 1

        yVal = (x - xc) / h + 1

        monomArr = monomPoly(yVal, N)
        monomWithKoeff = np.multiply(monomArr, coeff[col: col + N + 1])
        yVector[i] = np.sum(monomWithKoeff)

    return yVector


def infNorma(Yans, Yorig):
    absmax = np.max(np.abs(Yans - Yorig))
    infmax = absmax / np.max(Yorig)
    return infmax, absmax
