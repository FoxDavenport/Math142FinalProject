import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
N = 10000                      # Total population
num_languages = 10
time_steps = 100
max_languages_per_person = 3
status = np.random.dirichlet(np.ones(num_languages))  # Relative status

# Initialize: each person starts with 1 language
people = [set([np.random.choice(num_languages)]) for _ in range(N)]

# Store speaker counts over time
language_counts_over_time = np.zeros((time_steps, num_languages), dtype=int)

# Simulation
for t in range(time_steps):
    # Count how many people know each language
    counts = Counter(lang for person in people for lang in person)
    for i in range(num_languages):
        language_counts_over_time[t, i] = counts.get(i, 0)

    # Each person tries to learn one new language
    for person in people:
        if len(person) < max_languages_per_person:
            # Calculate exposure (fraction of population that knows each language)
            exposures = np.array([counts.get(i, 0) / N for i in range(num_languages)])
            probs = status * exposures
            probs /= probs.sum()  # Normalize

            # Choose one language not already known
            possible = [i for i in range(num_languages) if i not in person]
            if possible:
                chosen = np.random.choice(possible, p=probs[possible] / probs[possible].sum())
                person.add(chosen)

# Final histogram plot
final_counts = language_counts_over_time[-1]

plt.figure(figsize=(10, 6))
plt.bar(range(num_languages), final_counts)
plt.xlabel('Language')
plt.ylabel('Number of Speakers')
plt.title(f'Final Number of Speakers per Language (max {max_languages_per_person} languages per person)')
plt.xticks(range(num_languages), [f'Lang {i+1}' for i in range(num_languages)])
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd

# Count native, secondary, and tertiary speakers
native = np.zeros(num_languages, dtype=int)
secondary = np.zeros(num_languages, dtype=int)
tertiary = np.zeros(num_languages, dtype=int)

for person in people:
    for idx, lang in enumerate(person):
        if idx == 0:
            native[lang] += 1
        elif idx == 1:
            secondary[lang] += 1
        elif idx == 2:
            tertiary[lang] += 1

# Create a summary table
lang_summary = pd.DataFrame({
    'Language': [f'Lang {i+1}' for i in range(num_languages)],
    'Native Speakers': native,
    'Secondary Speakers': secondary,
    'Tertiary Speakers': tertiary
})

print(lang_summary.to_string(index=False))
