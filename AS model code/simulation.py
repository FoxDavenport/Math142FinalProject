import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)  # clean

    years = df.columns.astype(int)
    absolute_data = df.values

    # Normalize to proportions (by column)
    proportions = absolute_data / absolute_data.sum(axis=0)

    return df.index.tolist(), years, proportions

# 2. Extended AS model
def extended_as_model(t, x, s, a, beta):
    n = len(x)
    dxdt = np.zeros(n)
    epsilon = 1e-12

    for i in range(n):
        sum_gain = 0
        sum_loss = 0
        xi = max(x[i], epsilon)
        one_minus_xi = max(1 - xi, epsilon)
        for j in range(n):
            if i != j:
                xj = max(x[j], epsilon)
                one_minus_xj = max(1 - xj, epsilon)

                Pji = s[i] * xi**beta * one_minus_xj**(a - beta)
                Pij = s[j] * xj**beta * one_minus_xi**(a - beta)

                sum_gain += x[j] * Pji
                sum_loss += Pij
        dxdt[i] = sum_gain - x[i] * sum_loss
    return dxdt


# 3. Simulate model
def simulate_language_dynamics(x0, s, a, beta, t_span, t_eval):
    sol = solve_ivp(extended_as_model, t_span, x0, args=(s, a, beta), t_eval=t_eval, method='LSODA')
    return sol.y

# 4. Loss function
def loss_function_with_si(params, x_data, t_eval):
    a, beta = params[:2]
    s = params[2:]
    s = np.clip(s, 1e-6, 1.0)  # keep s values within [0, 1] to avoid instability
    s = s / np.sum(s)  # normalize so s behaves like probabilities

    x0 = x_data[:, 0]
    x_model = simulate_language_dynamics(x0, s, a, beta, (t_eval[0], t_eval[-1]), t_eval)
    return np.mean((x_model - x_data) ** 2)


# 5. Fit model
def fit_model_per_language(x_data, t_eval):
    n = x_data.shape[0]
    initial_s = np.ones(n) / n
    initial_params = np.concatenate(([1.0, 0.5], initial_s))  # [a, beta, s1, ..., sn]

    bounds = [(0.01, 3.0), (0.01, 3.0)] + [(1e-6, 1.0)] * n  # bounds for a, beta, s_i

    result = minimize(loss_function_with_si, initial_params, args=(x_data, t_eval),
                      bounds=bounds, method='L-BFGS-B')

    a_fit, beta_fit = result.x[:2]
    s_fit = result.x[2:]
    s_fit /= np.sum(s_fit)  # ensure s is normalized
    print(f"Optimization success: {result.success}, Loss: {result.fun}")
    print(f"a = {a_fit:.4f}, beta = {beta_fit:.4f}")
    print("s =", s_fit)
    return a_fit, beta_fit, s_fit


def main(filename):
    languages, years, proportions = load_data(filename)
    s = np.ones(len(languages)) / len(languages)  # Equal attractiveness
    # Fit parameters
    # In your main function after loading and normalizing the data
    recent_years = years[years >= 2015]
    recent_data = proportions[:, years >= 2015]
    a_fit, beta_fit, s_fit = fit_model_per_language(recent_data, recent_years)
    # a_fit, beta_fit, s_fit = fit_model_per_language(proportions, years)
    print(f"Fitted parameters: a = {a_fit:.4f}, beta = {beta_fit:.4f}")

    # Forecast
    forecast_start = years[-1]
    future_years = np.arange(forecast_start, forecast_start + 50, 5)
    x0 = proportions[:, -1]  # initial condition at 2025
    x_forecast = simulate_language_dynamics(x0, s_fit, a_fit, beta_fit,
                                        (forecast_start, forecast_start + 50),
                                        future_years)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for i, lang in enumerate(languages):
        plt.plot(future_years, x_forecast[i], label=lang)
    plt.title('Projected Speaker Fractions (Extended AS Model)')
    plt.xlabel('Year')
    plt.ylabel('Fraction of Global Language Speakers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

main("language_speakers_3.csv")



