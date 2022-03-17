""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Kristen Drexinger 	  	   		     		  		  		    	 		 		   		 		  
GT User ID: kdrexinger3		  	   		     		  		  		    	 		 		   		 		  
GT ID: 902738777	  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os
import numpy as np
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		     		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		     		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		     		  		  		    	 		 		   		 		  
    # TODO: Your code here  		  	   		     		  		  		    	 		 		   		 		  

    #load and clean dataframe
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df=orders_df.sort_index()
    orders_df.index=pd.to_datetime(orders_df.index)

    #define variables from dataframe
    sd=orders_df.index.values[0]
    ed = orders_df.index.values[-1]
    symbols = list(set(orders_df['Symbol']))
    dates = pd.date_range(sd, ed)

    #dataframe with price info for each symbol
    prices = get_data(symbols, dates, colname = 'Adj Close')  # automatically adds SPY
    prices['Cash']=np.ones(prices.shape[0])

    #intialize dataframe to keep track of units bought/sold
    units_df=prices*0.0
    units_df.iloc[0,-1]=start_val

    #calcualte daily impact
    for i, col in orders_df.iterrows():
        order_price = prices[col[0]].ix[i]
        order_units = col[2]
        if col[1] == "BUY":
            sign = 1
        else:
            sign = -1
        imp = order_units * order_price * impact
        units_df.loc[i, col[0]] += order_units * sign
        units_df.loc[i, "Cash"] += order_units * order_price * sign * -1
        units_df.loc[i, "Cash"] -= commission
        units_df.loc[i, "Cash"] -= imp

    #update table to reflect cumulative daily
    for i in range(1, units_df.shape[0]):
        for j in range(0, units_df.shape[1]):
            new_val = units_df.iloc[i, j] + units_df.iloc[i - 1, j]
            units_df.iloc[i, j] = new_val

    port_vals = prices * units_df
    port_vals["port_val"] = port_vals.sum(axis=1)
    port_vals['SPY'] = prices['SPY']
    cumReturnFund, avgDailyReturnFund, stdDeviationFund, sharpeRatioFund = compute_pf_stats(port_vals)

    #print("Sharpe Ratio: {}".format(sharpeRatioFund))
    #print("Cumulative Return: {}".format(cumReturnFund))
    #print("Standard Deviation of Fund: {}".format(stdDeviationFund))
    #print("Average Daily Return: {}".format(avgDailyReturnFund))

    # update port_vals to only show date/portfolio value
    port_val = port_vals.iloc[:, -2:-1]
    return port_val

def compute_pf_stats(port_vals):

    port_vals["daily_returns"] = (port_vals["port_val"][1:] / port_vals["port_val"][:-1].values) - 1
    port_vals["daily_returns"][0] = 0
    cumReturnFund = (port_vals.ix[-1, -2] - port_vals.ix[0, -2]) / port_vals.ix[0, -2]
    avgDailyReturnFund = port_vals["daily_returns"][1:].mean()
    stdDeviationFund = port_vals["daily_returns"][1:].std()
    sharpeRatioFund = (252 ** (0.5) * (avgDailyReturnFund)) / stdDeviationFund

    return cumReturnFund, avgDailyReturnFund, stdDeviationFund, sharpeRatioFund

def author():
    return "kdrexinger3"