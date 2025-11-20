# simulate_data.py
import numpy as np
import pandas as pd

def gen_normal(n=5000):
    # daily drift + noise
    t = 29 + 5*np.sin(np.linspace(0, 6.28, n)/ (2*3.14))  # slow sinusoidal drift
    h = 50 + 10*np.sin(np.linspace(0, 6.28, n)*0.8)
    t += np.random.normal(0, 0.3, n)
    h += np.random.normal(0, 1.0, n)
    return np.vstack([t, h]).T

def gen_anomalies(n=50):
    # spikes and step anomalies sprinkled
    t = 29 + np.random.normal(0, 0.5, n) + np.random.choice([5, -7, 0], size=n, p=[0.2,0.1,0.7])
    h = 50 + np.random.normal(0, 1.5, n) + np.random.choice([20, -30, 0], size=n, p=[0.1,0.05,0.85])
    return np.vstack([t, h]).T

normal = gen_normal(6000)
anoms = gen_anomalies(200)
data = np.vstack([normal, anoms])
df = pd.DataFrame(data, columns=["temp","hum"])
df.to_csv("sensor_data.csv", index=False)
print("sensor_data.csv created")
