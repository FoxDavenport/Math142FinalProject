import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_languages = 10
time_steps = 1000
dt = 0.01
a = 1.3  # Volatility parameter

# Initial language fractions (must sum to 1)
x = np.zeros((time_steps, num_languages))
x[0] = np.random.dirichlet(np.ones(num_languages), size=1)[0]

# Relative status for each language (must sum to 1)
status = np.random.dirichlet(np.ones(num_languages), size=1)[0]

# Simulation
for t in range(1, time_steps):
    dx = np.zeros(num_languages)
    for i in range(num_languages):
        for j in range(num_languages):
            if i != j:
                dx[i] += status[i] * x[t-1, j]**a - status[j] * x[t-1, i]**a
    x[t] = x[t-1] + dt * dx
    x[t] = np.clip(x[t], 0, 1)  # Prevents negative fractions
    x[t] /= np.sum(x[t])       # Renormalize to keep sum = 1

# Plot results
plt.figure(figsize=(12, 6))
for i in range(num_languages):
    plt.plot(np.arange(time_steps), x[:, i], label=f'Language {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Fraction of Population')
plt.title('Language Competition (Abrams-Strogatz Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
