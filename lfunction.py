import yfinance as yf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd


def em_period(x):  # error management: periode
    test_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    exist_count = test_list.count(x)
    if exist_count > 0:
        return x
    else:
        print('unknown period, switch to 3mo')
        return '3mo'


def em_kernel(x):  # error management: kernel
    test_list = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    exist_count = test_list.count(x)
    if exist_count > 0:
        return x
    else:
        print('unknown kernel, switch to gaussian')
        return 'gaussian'


def em_Ticker(data, x):  # error management: Ticker
    if data.empty:
        print(x + ' is an invalid Ticker')
        return False
    else:
        return True


def em_proba(x):  # error management: Proba
    if isinstance(x, list):
        res = True
        for i in x:
            if not em_proba(i):
                res = False
        return res

    else:
        if (x > 1.0) or (x < 0.0):
            return False
        else:
            return True


class yahooData():
    def __init__(self, udlg='META', period='6mo', kernel='gaussian', field='Close'):
        self.udlg = udlg
        self.period = em_period(period)
        self.kernel = em_kernel(kernel)
        self.GetInformation = yf.Ticker(self.udlg)
        self.field = field
        self.dataAll = self.GetInformation.history(period=self.period, interval="1d")
        self.isTicker = em_Ticker(self.dataAll, self.udlg)  # check if ticker is valid and allow any computation if true
        if self.isTicker:
            self.data = self.dataAll[self.field]
            self.data = self.data.to_numpy()
        self.kde = None
        self.bestparams = None
        self.info = None

    def getInfo(self):
        self.info = yf.Ticker(self.udlg)
        print(self.info)

    def calibKernelDensity(self):  # use Sklearn routine to automatically calibrate the bandwitch
        if self.isTicker:
            bandwidths = 10 ** np.linspace(-1, 1, 100)
            grid = GridSearchCV(KernelDensity(kernel=self.kernel), {'bandwidth': bandwidths}, cv=LeaveOneOut())
            grid.fit(self.data[:, None])
            self.bestparams = grid.best_params_['bandwidth']  # load the estimator
            # contructor of the kernel with the optimal parameter
            self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bestparams).fit(self.data[:, None])

    def dKD(self, x): # compute the pdf at a value x
        npt = np.array(x).reshape(-1, 1)  # from scalar to 2D array
        return np.exp(self.kde.score_samples(npt))[0]

    def pKD(self, x): # compute the cdf at a value x
        if self.isTicker:
            temp = quad(self.dKD, 0, x)[0]
            return 1-temp  # P(x>=X)

    def getUdlgtgtPrice(self, x): # imply spot for a cdf value of x
        self.isTicker = em_proba(x)

        if isinstance(x, list):
            res = []
            for i in x:
                res.append(self.getUdlgtgtPrice(i))

            return res
        else:
            def objf(y):
                return (x - self.pKD(y)) ** 2

            if self.isTicker:
                ''' trace back
                output = []
                output.append(['period', self.period])
                output.append(['kernel', self.kernel])
                output.append(['field', self.field])
                output.append(['udlg', self.udlg])
                output.append(['isTicker', self.isTicker])
                output.append(['kde', self.kde])
                df = pd.DataFrame(output)
                df.to_csv('out.csv')
                '''
                x0 = self.data[0]  # initialisation of the solver
                res = minimize(objf, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
                print(str(res['message']) + ' at probe ' + str(x) + ' for ticker: ' + self.udlg)
                return float(res['x'])

            else:
                print(str(x) + ' is not a valid proba for Ticker: ' + self.udlg)

    def storeRawData(self, x=1):
        # generate a csv file for the yahoo raw data for a given folder and file name x
        if x == 1:
            self.dataAll.to_csv('output.csv')
        else:
            self.dataAll.to_csv(x)
