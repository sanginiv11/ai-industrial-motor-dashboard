"""Flask backend: dataset playback -> DSP/FFT -> energy_model (unchanged) -> JSON.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from energy_model import ConveyorEnergyModel
import ml_slot

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "combined_dataset.csv"))
FS, WIN, N = 1000.0, 256, len(df)           # fs / window from models/README.json & simulator config
DT = 1.0 / FS                                # each row = 1 ms
cond = df["operating_condition"].values
vib = df["vibration"].values
model = ConveyorEnergyModel()                # default config, unchanged

# --- energy model applied to every dataset row (duration 1 ms each) ---
# Severity is not in the dataset, so severity=0.0 is passed (no fault penalty).
# severity=1.0 is the model's own maximum, shown only as an upper bound.
keys = ["b_kw", "a_kw", "w_kw", "b_kwh", "a_kwh", "w_kwh", "b_co2", "a_co2", "w_co2"]
E = {k: np.zeros(N) for k in keys}
for k, (r, l, v, c) in enumerate(zip(df.motor_rpm.values, df.load.values, df.voltage.values, cond)):
    a = model.calculate(r, l, v, c, 0.0, DT)
    w = model.calculate(r, l, v, c, 1.0, DT)
    E["b_kw"][k], E["a_kw"][k], E["w_kw"][k] = a["baseline_power_kw"], a["faulty_power_kw"], w["faulty_power_kw"]
    E["b_kwh"][k], E["a_kwh"][k], E["w_kwh"][k] = a["baseline_energy_kwh"], a["faulty_energy_kwh"], w["faulty_energy_kwh"]
    E["b_co2"][k], E["a_co2"][k], E["w_co2"][k] = a["baseline_co2_kg"], a["faulty_co2_kg"], w["faulty_co2_kg"]
CUM = {k: np.cumsum(E[k]) for k in keys if "kwh" in k or "co2" in k}

segs, start = [], 0
for i in range(1, N + 1):
    if i == N or cond[i] != cond[start]:
        segs.append({"start": start, "end": i - 1, "condition": str(cond[start])}); start = i

app = Flask(__name__, static_folder=os.path.join(BASE, "static"))

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/meta")
def meta():
    c = model.cfg
    return jsonify(n=N, fs=FS, window=WIN, segments=segs,
                   energy_config=dict(rated_power_kw=c.rated_power_kw, rated_rpm=c.rated_rpm,
                                      base_efficiency=c.base_efficiency, reference_voltage_v=c.reference_voltage_v,
                                      emission_factor=c.grid_emission_factor_kg_per_kwh,
                                      fault_penalties=c.fault_penalties),
                   dataset_mean_voltage=float(df.voltage.mean()),
                   ml=ml_slot.status())

@app.get("/api/frame")
def frame():
    i = int(np.clip(int(request.args.get("i", 0)), 0, N - 1))
    row = df.iloc[i]
    end = max(i, WIN - 1)
    seg = vib[end - WIN + 1:end + 1]
    w = np.hanning(WIN)
    mag = np.abs(np.fft.rfft((seg - seg.mean()) * w)) * 2 / w.sum()
    pk = int(np.argmax(mag[1:]) + 1)
    ce = {k: float(v[i]) for k, v in CUM.items()}
    return jsonify(
        i=i, timestamp=str(row.timestamp), condition=str(row.operating_condition),
        t={k: float(row[k]) for k in ["voltage", "current", "power_factor", "motor_rpm", "belt_speed", "temperature", "load", "vibration"]},
        vib=dict(wave=np.round(seg, 4).tolist(), fft=np.round(mag, 5).tolist(), rms=float(np.sqrt(np.mean(seg ** 2))),
                 peak=float(np.max(np.abs(seg))), peak_hz=pk * FS / WIN, bin_hz=FS / WIN, shaft_hz=float(row.motor_rpm) / 60.0,
                 warming=bool(i < WIN - 1), mixed=bool(len(set(cond[end - WIN + 1:end + 1])) > 1)),
        energy=dict(baseline_kw=float(E["b_kw"][i]), asrun_kw=float(E["a_kw"][i]), upper_kw=float(E["w_kw"][i]),
                    voltage_effect=float(E["a_kw"][i] / E["b_kw"][i]) if E["b_kw"][i] > 0 else 1.0, cum=ce),
        ml=dict(enabled=ml_slot.ENABLED))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
