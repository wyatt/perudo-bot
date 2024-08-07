from math import comb

def binomial_pmf(n, p, k):
    """
    Calculate the probability mass function (PMF) for a binomial distribution.

    Parameters:
    n (int): number of trials
    p (float): probability of success on each trial
    k (int): number of successes

    Returns:
    float: the probability P(X = k)
    """
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binomial_probability_gte(n, p, k):
    """
    Calculate the probability P(X >= k) for a binomial distribution.

    Parameters:
    n (int): number of trials
    p (float): probability of success on each trial
    k (int): threshold number of successes

    Returns:
    float: the probability P(X >= k)
    """
    return sum(binomial_pmf(n, p, j) for j in range(k, n + 1))

if __name__ == "__main__":
    # Example usage:
    n = 25
    p = 1/6
    k = 3

    # Calculate P(X >= k) for different k values
    for k_value in range(11):
        probability = binomial_probability_gte(n, p, k_value)
        print(f"The probability P(X >= {k_value}) for n={n}, p={p} is: {probability}")