import json
import os
import numpy as np

results_dir = 'acn_exp_2x2_results'

# Load all seeds
forecast_data = []
no_forecast_data = []

for seed in range(1, 6):
    with open(os.path.join(results_dir, f'forecast_seed{seed}.json')) as f:
        data = json.load(f)
        forecast_data.append({
            'violation': data['violation_rate'] * 100,
            'unmet': data['unmet_fraction'] * 100,
            'delivered': data['delivered_kwh']
        })
    with open(os.path.join(results_dir, f'no_forecast_seed{seed}.json')) as f:
        data = json.load(f)
        no_forecast_data.append({
            'violation': data['violation_rate'] * 100,
            'unmet': data['unmet_fraction'] * 100,
            'delivered': data['delivered_kwh']
        })

# Aggregate
print("cap_only: 0.00% / 17.46% / 2161.6")
print()
print("no_forecast (5 seeds):")
for i, d in enumerate(no_forecast_data, 1):
    print(f"  seed{i}: {d['violation']:.2f}% / {d['unmet']:.2f}% / {d['delivered']:.1f}")

v_mean = np.mean([d['violation'] for d in no_forecast_data])
v_std = np.std([d['violation'] for d in no_forecast_data])
u_mean = np.mean([d['unmet'] for d in no_forecast_data])
u_std = np.std([d['unmet'] for d in no_forecast_data])
d_mean = np.mean([d['delivered'] for d in no_forecast_data])
d_std = np.std([d['delivered'] for d in no_forecast_data])
print(f"Aggregate: {v_mean:.2f}±{v_std:.2f}% / {u_mean:.2f}±{u_std:.2f}% / {d_mean:.1f}±{d_std:.1f}")

print()
print("forecast (5 seeds):")
for i, d in enumerate(forecast_data, 1):
    print(f"  seed{i}: {d['violation']:.2f}% / {d['unmet']:.2f}% / {d['delivered']:.1f}")

v_mean = np.mean([d['violation'] for d in forecast_data])
v_std = np.std([d['violation'] for d in forecast_data])
u_mean = np.mean([d['unmet'] for d in forecast_data])
u_std = np.std([d['unmet'] for d in forecast_data])
d_mean = np.mean([d['delivered'] for d in forecast_data])
d_std = np.std([d['delivered'] for d in forecast_data])
print(f"Aggregate: {v_mean:.2f}±{v_std:.2f}% / {u_mean:.2f}±{u_std:.2f}% / {d_mean:.1f}±{d_std:.1f}")
