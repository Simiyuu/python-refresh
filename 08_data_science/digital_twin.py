import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

historical_activities = [
    0,1,2,1,3,1,4,
    0,1,2,1,3,1,4,0
]

activity_names = {
    0: 'Sleep',
    1: 'Work',
    2: 'Exercise',
    3: 'Social',
    4: 'Leisure'
}

def calculate_probabilities(sequence):
    total_samples = len(sequence)
    counts = Counter(sequence)

    probabilities = {
        activity: counts.get(activity, 0) / total_samples
        for activity in activity_names.keys()
    }
    return probabilities

def simulate_twin_day(prob_dict, periods=7):
    states = list(prob_dict.keys())
    probs = list(prob_dict.values())

    simulated_numeric = np.random.choice(states, size=periods, p=probs)
    simulated_labels = [activity_names[idx] for idx in simulated_numeric]
    return simulated_numeric, simulated_labels

if __name__ == '__main__':
    np.random.seed(42)

    probs = calculate_probabilities(historical_activities)

    print("-- Activity Probabilities --")
    for act_id, prob in probs.items():
        name = activity_names[act_id]
        print(f"[{act_id}] {name:<10}: {prob * 100:.1f}%")

sim_ids, sim_names = simulate_twin_day(probs, periods=7)
print("\n-- Simulated Twin Schedule --")
print(" -> ".join(sim_names))

freq_series = pd.Series(
    [activity_names[i] for i in historical_activities]
).value_counts(normalize=True) * 100

plt.figure(figsize=(8, 4))
freq_series.plot(kind='bar', color='#4F46E5', edgecolor='black')
plt.title("Digital Twin: Activity Distribution (%)")
plt.xlabel("Activity")
plt.ylabel("Observed Frequency (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()