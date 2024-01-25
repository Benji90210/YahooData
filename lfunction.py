import yfinance as yf
import numpy as np


def mainProcess():
    GetInformation = yf.Ticker("META")
    return GetInformation.info


def dProcess():
    GetInformation = yf.Ticker("META")
    g = GetInformation.history(period="6mo")
    print(g)