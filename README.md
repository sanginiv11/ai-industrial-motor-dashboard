# AI-Powered Energy and Predictive Maintenance for Industrial Motors
Simulation-mode dashboard. Flask backend plays `data/combined_dataset.csv` (from conveyor_monitor_milestone1), computes FFT, and runs `energy_model.py` unchanged.

    pip install -r requirements.txt
    python app.py     # http://127.0.0.1:5000   (needs internet for three.js + Google Fonts CDN)

Honest limits
- Diagnosis = dataset `operating_condition` label (ground truth), NOT an ML prediction.
- ML ensemble (models/) is a disabled slot: the 164-feature extractor was not supplied. See ml_slot.py. models/RESULTS.txt reports 88.5% on its own hard simulator and 45% on original-source windows.
- Severity is not in the dataset: shown as N/A. Energy "model output" uses severity 0; "upper bound" uses severity 1.0.
- Simulator runs ~400 V; energy model reference is 230 V (unchanged) -> constant ~3% voltage-term offset.
- All data and fault penalties are synthetic; nothing is validated on real motors.

PUBLIC DEPLOYED URL: https://ai-industrial-motor-dashboard.onrender.com/
