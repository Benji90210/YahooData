import yfinance as yf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def likelyhoodKD(x, kernel, bestparams, data):
    kde = KernelDensity(kernel=kernel, bandwidth=bestparams).fit(data[:, None])
    npt = np.array(x).reshape(-1, 1)
    return np.exp(kde.score_samples(npt))[0]


def em_period(x):
    test_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    exist_count = test_list.count(x)
    if exist_count > 0:
        return x
    else:
        print('unknown period, switch to 3mo')
        return '3mo'


def em_kernel(x):
    test_list = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    exist_count = test_list.count(x)
    if exist_count > 0:
        return x
    else:
        print('unknown kernel, switch to gaussian')
        return 'gaussian'


def em_Ticker(data, x):
    if data.empty:
        print(x + ' is an invalid Ticker')
        return False
    else:
        return True


def em_proba(x):
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
        self.isTicker = em_Ticker(self.dataAll, self.udlg)
        if self.isTicker:
            self.data = self.dataAll[self.field]
            self.data = self.data.to_numpy()
        self.kde = None
        self.bestparams = None
        self.info = None


    def getInfo(self):
        self.info = yf.Ticker(self.udlg)
        print(self.info)

    def calibKernelDensity(self):
        if self.isTicker:
            bandwidths = 10 ** np.linspace(-1, 1, 100)
            grid = GridSearchCV(KernelDensity(kernel=self.kernel), {'bandwidth': bandwidths}, cv=LeaveOneOut())
            grid.fit(self.data[:, None])
            self.bestparams = grid.best_params_['bandwidth']
            self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bestparams).fit(self.data[:, None])

    def dKD(self, x):
        npt = np.array(x).reshape(-1, 1)  # from scalar to 2D array
        return np.exp(self.kde.score_samples(npt))[0]

    def pKD(self, x):
        if self.isTicker:
            temp = quad(self.dKD, 0, x)[0]
            return 1-temp  # P(x>=X)

    def getUdlgtgtPrice(self, x):
        self.isTicker = em_proba(x)
        def objf(y):
            return (x - self.pKD(y)) ** 2

        if self.isTicker:
            res = minimize(objf, 400.0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
            print(str(res['message']) + ' at probe ' + str(x) + ' for ticker: ' + self.udlg)
            return float(res['x'])

        else:
            print(str(x) + ' is not a valid proba for Ticker: ' + self.udlg)

    def storeRawData(self, x=1):
        if x == 1:
            self.dataAll.to_csv()
        else:
            self.dataAll.to_csv(x)
