import solveFun as sf
from solveFun import np
from numpy.ma.core import log2

import complFun as cf
from complFun import plt

import time
import pandas as pd


def main():
    x0 = 0
    xl = 1
    N = 4
    kList = [5, 10, 20, 40, 80, 160]

    ansArr = np.zeros(shape=[6, 7])

    for i in range(len(kList)):

        print("start K = ", kList[i])

        start_time = time.time()
        globMatrix, bVector = sf.collocationMethod(x0, xl, N, kList[i])
        globMatrix2, bVector2 = sf.collocationMethod(x0, xl, N, int(kList[i] / 2))

        if (i == 0):
            ans = pd.DataFrame(globMatrix)

        ans, arrB = sf.transformHouseholder(globMatrix, bVector)
        ans2, arrB2 = sf.transformHouseholder(globMatrix2, bVector2)

        # gauss Reverse
        arrAnsX = sf.gaussReverse(ans, arrB)
        arrAnsX2 = sf.gaussReverse(ans2, arrB2)

        times = time.time() - start_time

        step = 1
        xMax = 1000
        xArr = np.array([i for i in range(0, xMax, step)]) / xMax

        yVector = cf.getValueCollocFun(xArr, arrAnsX, N, kList[i], x0, xl)
        yVector2 = cf.getValueCollocFun(xArr, arrAnsX2, N, int(kList[i] / 2), x0, xl)

        infN, absN = cf.infNorma(yVector, sf.testFun2(xArr))
        infN2, absN2 = cf.infNorma(yVector2, sf.testFun2(xArr))
        R1 = -np.log2(infN / infN2)
        R2 = -np.log2(absN / absN2)

        ansArr[i][0] = kList[i]
        ansArr[i][1] = infN
        if (i != 0):
            ansArr[i][2] = R1
        ansArr[i][3] = absN
        if (i != 0):
            ansArr[i][4] = R2
        ansArr[i][5] = np.linalg.cond(globMatrix)
        ansArr[i][6] = times

        plt.figure(figsize=(15, 25), facecolor='white')
        cf.showData(len(kList), 1, xArr, sf.testFun2(xArr), 1, "exact solution", "X", "Y", 0.05, 0.01, '^')
        cf.showData(len(kList), 1, xArr, yVector, 1, "approximate solution for K = " + str(kList[i]),
                    "X", "Y", 0.05, 0.01, 'o')

        plt.show()

    return ansArr, ans


if __name__ == "__main__":
    arrTab, ans = main()
    arrPD = pd.DataFrame(arrTab, columns=["K", "Ea", "R", "Er", "R", "cond(A)", "tsol"])
    print(arrPD)

