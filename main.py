
from lfunction import yahooData

if __name__ == "__main__":
    A = yahooData('META')
    A.calibKernelDensity()
    f = A.getUdlgtgtPrice(0.50)
    print(f)



