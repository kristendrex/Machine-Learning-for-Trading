""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: kdreinger3 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 902738777 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import scipy.optimize as scipyOpt


def compute_daily_returns(df):
    rets = np.log(df/ df.shift(1))
    return rets

def cumulative_returns(df):
    cum_returns = df.iloc[-1] / df.iloc[0] - 1
    return cum_returns

def get_portfolio_value(prices,allocs):
    normed = prices / prices.iloc[0]
    pos_vals = normed * allocs
    port_val = np.sum(pos_vals,axis =1)
    return port_val

def get_portfolio_stats(port_val):
    k = np.sqrt(252)
    daily_ret = compute_daily_returns(port_val)
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    cumulative_ret = cumulative_returns(port_val)
    sharpe_ratio = avg_daily_ret / std_daily_ret * k
    return cumulative_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def apply_minimizer(prices):
    num = len(prices.columns)
    rets = compute_daily_returns(prices)

    def portfolioTests(allocs):
        allocs = np.array(allocs)
        ret_test = np.sum(rets.mean()*allocs) * 252
        std_test = np.sqrt(np.dot(allocs.T,np.dot(rets.cov()*252,allocs)))
        return np.array([ret_test, std_test, ret_test/std_test])

    def min_sharpe(allocs):
        return -portfolioTests(allocs)[2]

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0,1) for x in range(num))

    opts = scipyOpt.minimize(min_sharpe, num * [1. / num,], method = 'SLSQP', bounds = bnds, constraints = cons)

    return opts['x']



def optimize_portfolio(  		  	   		     		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		     		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		     		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		     		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		     		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		     		  		  		    	 		 		   		 		  
    statistics.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		     		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		     		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		     		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		     		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		     		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		     		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		     		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		     		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		     		  		  		    	 		 		   		 		  

    #find optimal allocations
    allocs = apply_minimizer(prices)
    port_val = get_portfolio_value(prices,allocs)
    cr, adr, sddr, sr = get_portfolio_stats(port_val)


    port_df = port_val.to_frame()
    spy_df = prices_SPY.to_frame()
    normSpy_df = spy_df/spy_df.ix[0,:]
    port_df.columns = ['Portfolio']
    if gen_plot == True:
        # add code to plot here
        df= port_df.join(normSpy_df)
        dfplot = df.plot(title='Daily Portfolio Value and SPY', fontsize=12)
        dfplot.set_xlabel("Date")
        dfplot.set_ylabel("Price")
        plt.savefig('plot.png')


        pass  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
  		  	   		     		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		     		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		     		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		     		  		  		    	 		 		   		 		  
    test_code()

