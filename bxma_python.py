import pandas as pd
import numpy as np

def monte_carlo_var(mu, sigma, port_val, n_sims=10000, confidence_level=0.95):
    """
    Calculate Monte Carlo VaR
    """
    #1. Generate random returns
    returns = np.random.normal(mu, sigma, n_sims)
    #2. Calculate portfolio vals
    portfolio_vals = port_val * (1+returns)
    #3. Calculate losses
    losses = port_val - portfolio_vals
    #4. Calculate VaR (the percentile of the losses)
    var = np.percentile(losses, confidence_level*100)
    return var