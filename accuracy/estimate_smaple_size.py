"""
Applying the formula for sample size in simple random sampling 
for estimating a population proportion:
    n = (Z^2 * p * (1-p)) / E^2
    
    Where:
    n = required sample size
    Z = Z-score corresponding to the desired confidence level 
        (e.g., 1.96 for 95% confidence)
    p = estimated overall accuracy 
        (or a conservative estimate like 0.5 if unknown)
    E = desired margin of error 
        (half-width of the confidence interval, e.g., 0.05 for ±5% accuracy)
"""

import math
from scipy.stats import norm


def calculate_z(confidence_level):
    """
    Compute the Z-score corresponding to the given confidence level.

    Parameters:
      confidence_level (float): Confidence level as a decimal (e.g., 0.75 for 75%).

    Returns:
      float: Z-score for the two-tailed test.
    """
    # Determine the total significance level (alpha)
    alpha = 1 - confidence_level
    # For a two-tailed test, use half the significance level
    Z = norm.ppf(1 - alpha/2)
    return Z


def calculate_sample_size(Z, p, E):
    """
    Calculate the required sample size based on the Z-score, estimated proportion, and margin of error.

    Parameters:
      Z (float): Z-score corresponding to the confidence level.
      p (float): Estimated proportion (a value between 0 and 1).
      E (float): Desired margin of error.

    Returns:
      int: Required sample size, rounded up.
    """
    n = (Z ** 2 * p * (1 - p)) / (E ** 2)
    return math.ceil(n)




if __name__ == "__main__":
    confidence_level = 0.95  # 95% confidence level
    Z = calculate_z(confidence_level)
    p = 0.7   # Conservative estimate 
    E = 0.05  # Desired margin of error (5%)
    
    sample_size = calculate_sample_size(Z, p, E)
    
    print("Calculated Z value:", Z)
    print("Required sample size:", sample_size)
