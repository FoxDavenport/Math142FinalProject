import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Load data 
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    years = df.columns.astype(int)
    absolute_data = df.values
    proportions = absolute_data / absolute_data.sum(axis=0)
    return df.index.tolist(), years, proportions

# 2. AS model 
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

# 3. Simulate with tiny yearly shifts
def simulate_language_dynamics(x0, s, a, beta, t_span, t_eval):
    if callable(s):
        s_values = np.array([s(t) for t in t_eval])
    else:
        s_values = np.tile(s, (len(t_eval), 1))

    sol = solve_ivp(extended_as_model, t_span, x0, args=(s_values[0], a, beta),
                    t_eval=[t_span[0]], method='LSODA')

    for i in range(1, len(t_eval)):
        sol_i = solve_ivp(extended_as_model, [t_eval[i-1], t_eval[i]], sol.y[:, -1],
                         args=(s_values[i], a, beta), t_eval=[t_eval[i]], method='LSODA')
        sol.y = np.hstack((sol.y, sol_i.y))
        sol.t = np.hstack((sol.t, sol_i.t))

    return sol.y

# 4. Low-volatility status shifts 
def gradual_status(t, base_s, volatility=0.01, mean_reversion=0.1):
    """Tiny annual shifts with mean reversion to original status"""
    if not hasattr(gradual_status, 'last_s'):
        gradual_status.last_s = base_s.copy()

    # Tiny random changes (volatility=0.01 is 1% max change)
    delta = volatility * (np.random.random(len(base_s)) - 0.5)  # Centered at 0

    # Mean reversion pulls status back toward original values
    gradual_status.last_s += delta + mean_reversion * (base_s - gradual_status.last_s)

    # Ensure valid probabilities
    gradual_status.last_s = np.clip(gradual_status.last_s, 0.001, 1.0)
    gradual_status.last_s /= np.sum(gradual_status.last_s)

    return gradual_status.last_s

# 5. New: Simulate factors (M, I, B)
def simulate_factors(n, t_eval, initial_values=None, growth_rates=None, volatility=0.05):
    """
    Simulate M (migration), I (internet), B (business relations) factors
    Returns: dict of {'M': array, 'I': array, 'B': array} for each time point
    """
    if initial_values is None:
        initial_values = {'M': np.random.rand(n), 
                         'I': np.random.rand(n), 
                         'B': np.random.rand(n)}
    if growth_rates is None:
        growth_rates = {'M': np.random.normal(0, 0.02, n),
                       'I': np.random.normal(0.03, 0.01, n),
                       'B': np.random.normal(0.02, 0.01, n)}
    
    factors = {'M': np.zeros((len(t_eval), n)),
              'I': np.zeros((len(t_eval), n)),
              'B': np.zeros((len(t_eval), n))}
    
    for i, t in enumerate(t_eval):
        if i == 0:
            for key in ['M', 'I', 'B']:
                factors[key][i] = initial_values[key]
        else:
            for key in ['M', 'I', 'B']:
                # Geometric Brownian motion-like simulation
                factors[key][i] = factors[key][i-1] * (1 + growth_rates[key] + 
                                                      np.random.normal(0, volatility, n))
                # Ensure values stay between 0 and 1
                factors[key][i] = np.clip(factors[key][i], 0.01, 1.0)
    
    return factors

# 6. New: Enhanced dynamic status with multiple factors and gradual shifts
def enhanced_dynamic_status(t, s_base, gdp_df, lang_to_countries, factors, weights):
    """
    Calculate status using formula: si = a1*si_base + a2*Mi + a3*Ei + a4*Ii + a5*Bi
    weights = [a1, a2, a3, a4, a5] (should sum to 1)
    """
    # Get economic factor
    Ei = compute_ei_t(gdp_df, lang_to_countries, t)
    
    # Find index for current time
    t_index = np.where(factors['time'] == t)[0][0]
    
    # Get current factors
    Mi = factors['M'][t_index]
    Ii = factors['I'][t_index]
    Bi = factors['B'][t_index]
    
    # Normalize all factors (except Ei which is already normalized)
    Mi = Mi / Mi.sum()
    Ii = Ii / Ii.sum()
    Bi = Bi / Bi.sum()
    
    # Calculate combined status
    s = (weights[0] * s_base + 
         weights[1] * Mi + 
         weights[2] * Ei + 
         weights[3] * Ii + 
         weights[4] * Bi)
    
    # Apply gradual status shifts to the combined status
    s = gradual_status(t, s, volatility=0.005, mean_reversion=0.05)
    
    return s

# 7. Loss function 
def loss_function_with_si(params, x_data, t_eval):
    a, beta = params[:2]
    s = params[2:]
    s = np.clip(s, 1e-6, 1.0)  # keep s values within [0, 1] to avoid instability
    s = s / np.sum(s)  # normalize so s behaves like probabilities

    x0 = x_data[:, 0]
    x_model = simulate_language_dynamics(x0, s, a, beta, (t_eval[0], t_eval[-1]), t_eval)
    return np.mean(np.abs(x_model - x_data))

# 8. Fit model 
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

# 9. Economic factor calculation 
gdp_df = pd.read_csv("gdp_by_country_2025_2075.csv", index_col=0)

#language -> (country, weight)
lang_to_countries = {
  'English': [('United States',1.0), ('United Kingdom',1.0)],
  'Mandarin Chinese': [('China',1.0), ('Taiwan',1.0)],
  'Hindi': [('India',0.4), ('Nepal',1.0)],
  'Spanish': [('Spain',1.0), ('Mexico',1.0)],
  'French': [('France',1.0), ('Canada',0.2)],
  'Modern Arabic': [('Saudi Arabia',1.0), ('Egypt',1.0)],
  'Bengali': [('India',0.1), ('Bangladesh',1.0)],
  'Russian': [('Russia',1.0)],
  'Portuguese': [('Brazil',1.0), ('Portugal',1.0)],
  'Indonesian': [('Indonesia',1.0)],
  'Urdu': [('India',0.05), ('Pakistan',1.0)],
  'Japanese': [('Japan',1.0)],
  'German': [('Germany',1.0), ('Switzerland',0.6)],
}
languages = [
    'English',
    'Mandarin Chinese',
    'Hindi',
    'Spanish',
    'French',
    'Modern Arabic',
    'Bengali',
    'Russian',
    'Portuguese',
    'Indonesian',
    'Urdu',
    'Japanese',
    'German'
]

def compute_ei_t(gdp_df, lang_to_countries, t):
    Ei = []
    for lang in languages:
        total = 0
        for country, weight in lang_to_countries[lang]:
            if country in gdp_df.columns:
                total += weight * gdp_df.loc[t, country]
        Ei.append(total)
    Ei = np.array(Ei)
    return Ei / Ei.sum()  # normalize

native_proportions = np.array([
    0.11915674,  # English
    0.30247479,  # Mandarin Chinese
    0.10540788,  # Hindi
    0.14787657,  # Spanish
    0.02260923,  # French
    0.00000000,  # Modern Arabic (L1 not reported)
    0.07393828,  # Bengali
    0.04430186,  # Russian
    0.07638252,  # Portuguese
    0.02291476,  # Indonesian
    0.02383135,  # Urdu
    0.03788573,  # Japanese
    0.02322029   # German
])


# 10. Updated main function
def main(filename):
    languages, years, proportions = load_data(filename)
    recent_years = years[years >= 2015]
    recent_data = proportions[:, years >= 2015]
    a_fit, beta_fit, s_fit = fit_model_per_language(recent_data, recent_years)

    forecast_start = years[-1]
    future_years = np.arange(forecast_start, forecast_start + 50, 1)
    
    # Simulate factors for future years
    factors = simulate_factors(len(languages), future_years)
    factors['time'] = future_years  # Add time to factors dict for reference
    
    # Define weights for status calculation [a1, a2, a3, a4, a5]
    # These weights should sum to 1. Adjust based on your assumptions.
    status_weights = [0.6, 0.2, 0.2, 0, 0]  # Base, Migration, Economic, Internet, Business
    
    # Create status function with all factors and gradual shifts
    s_enhanced = lambda t: enhanced_dynamic_status(
        t, s_fit, gdp_df, lang_to_countries, factors, status_weights
    )
    
    # Single simulation 
    x0 = native_proportions
    x_forecast = simulate_language_dynamics(
        x0, s_enhanced, a_fit, beta_fit,
        (forecast_start, forecast_start + 50),
        future_years
    )

    plt.figure(figsize=(12, 6))
    for i, lang in enumerate(languages):
        plt.plot(future_years, x_forecast[i], label=lang, alpha=0.8)
    plt.title('Language Projections with Enhanced Dynamic Status and Gradual Shifts')
    plt.xlabel('Year')
    plt.ylabel('Fraction of Speakers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Uncertainty analysis (10 runs)
    n_simulations = 50
    all_simulations = np.zeros((n_simulations, len(languages), len(future_years)))

    for sim in range(n_simulations):
        # Resimulate factors for each run to capture uncertainty
        sim_factors = simulate_factors(len(languages), future_years)
        sim_factors['time'] = future_years
        s_sim = lambda t: enhanced_dynamic_status(
            t, s_fit, gdp_df, lang_to_countries, sim_factors, status_weights
        )
        
        all_simulations[sim] = simulate_language_dynamics(
            x0, s_sim, a_fit, beta_fit,
            (forecast_start, forecast_start + 50),
            future_years
        )

    final_proportions = x_forecast[:, -1]
    ranking = sorted(zip(languages, final_proportions), key=lambda x: x[1], reverse=True)

    print("\nRanking of languages by proportion in 2075:")
    for rank, (lang, proportion) in enumerate(ranking, 1):
        print(f"{rank}. {lang}: {proportion:.4f}")

    # Plot median and 25-75th percentiles
    plt.figure(figsize=(12, 6))
    for i, lang in enumerate(languages[:3]):  # First 3 languages
        median = np.median(all_simulations[:, i, :], axis=0)
        p25 = np.percentile(all_simulations[:, i, :], 25, axis=0)
        p75 = np.percentile(all_simulations[:, i, :], 75, axis=0)
        plt.plot(future_years, median, label=lang, lw=2)
        plt.fill_between(future_years, p25, p75, alpha=0.15)

    plt.title('Projections with 25-75th Percentile Bands (Enhanced Status with Gradual Shifts)')
    plt.xlabel('Year')
    plt.ylabel('Fraction of Speakers')
    plt.legend()
    plt.show()

main("language_speakers_3.csv")