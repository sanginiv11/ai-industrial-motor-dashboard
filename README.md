# AI-Powered Energy and Predictive Maintenance for Industrial Motors

A simulation-based industrial motor monitoring dashboard that combines **motor telemetry, vibration signal processing, FFT analysis, fault-condition monitoring, energy modelling, and predictive-maintenance-oriented insights** in a web interface.

## 🌐 Live Dashboard

**Public Demo:**  
https://ai-industrial-motor-dashboard.onrender.com/

## Overview

The system operates in simulation mode using a synthetic motor dataset. A Flask backend plays the motor data from `data/combined_dataset.csv`, performs vibration analysis including FFT computation, and runs the supplied `energy_model.py` without modification.

The dashboard provides:

- Live motor telemetry
- Rotating 3D motor visualization
- Vibration monitoring
- FFT/frequency-spectrum analysis
- Operating-condition monitoring
- Energy consumption analysis
- Estimated CO₂ impact
- Maintenance-oriented information
- Simulation playback

## System Flow

```text
Synthetic Motor Dataset
          ↓
   Flask Backend
          ↓
 ┌────────┴─────────┐
 ↓                  ↓
Telemetry       Vibration Signal
 ↓                  ↓
Motor KPIs       FFT Analysis
          ↓
   Condition Monitoring
          ↓
     Energy Model
          ↓
 Energy + CO₂ Analysis
          ↓
      Web Dashboard
```

## Technology Stack

- **Python**
- **Flask**
- **NumPy**
- **Pandas**
- **HTML / CSS / JavaScript**
- **Three.js** for 3D motor visualization
- **XGBoost / Extra Trees model files** retained as an ML integration slot

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The frontend currently loads Three.js and Google Fonts through external CDNs, so an internet connection is required for those resources.

## Project Structure

```text
ai-industrial-motor-dashboard/
│
├── app.py
├── energy_model.py
├── requirements.txt
│
├── data/
│   └── combined_dataset.csv
│
├── models/
│   ├── xgb.pkl
│   └── extra_trees.pkl
│
└── static/
    └── index.html
```

## Data and Model Limitations

This project is currently a **simulation and research prototype**, not a validated industrial monitoring system.

### Diagnosis

The displayed diagnosis uses the dataset's `operating_condition` label.

It is therefore a **ground-truth dataset label and not a live ML prediction**.

### ML Ensemble

The supplied ML models are retained as an integration slot.

The original **164-feature extraction pipeline was not supplied**, so the ensemble is currently disabled rather than being presented as a live prediction system.

The supplied model results report:

- **88.5% accuracy** on its own hard synthetic simulator holdout
- **45% accuracy** on untouched original-source windows

These figures should not be interpreted as real-world industrial validation.

### Severity

Severity is not included in the supplied motor dataset.

Therefore, the dashboard displays severity as:

```text
N/A — not provided by simulator
```

For energy analysis, severity `0` is used for the model output and severity `1.0` is used as an upper-bound scenario.

### Voltage Reference

The simulator operates around **400 V**, while the supplied energy model uses **230 V as its reference voltage**.

The energy model has intentionally been left unchanged. This creates an approximately **3% voltage-term offset** in the model output.

### Synthetic Data

The motor data and fault-energy penalties are **synthetic assumptions**.

They have not been experimentally validated against measurements from physical industrial motors.

Therefore, the dashboard should be considered a **simulation/prototype platform rather than a production-grade predictive-maintenance system**.

## Purpose

The project demonstrates how:

**Motor telemetry + DSP/FFT + condition monitoring + energy modelling + maintenance insights**

can be integrated into a single industrial monitoring interface.

Future development can replace the simulated data with real sensor streams and integrate the complete feature-extraction pipeline required for live ML inference.
