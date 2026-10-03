"""
Conveyor Motor Energy + CO2 Model
---------------------------------
Physics-informed synthetic model intended to plug into an existing
conveyor-motor simulator and fault classifier.

Inputs per simulated window:
    rpm, load_pct, voltage_v, fault, severity, duration_s

Outputs:
    baseline_power_kw
    faulty_power_kw
    power_penalty_pct
    baseline_energy_kwh
    faulty_energy_kwh
    extra_energy_kwh
    baseline_co2_kg
    faulty_co2_kg
    extra_co2_kg

IMPORTANT:
The fault penalties are configurable synthetic assumptions. They are
not experimentally validated motor/fault measurements.
"""

from dataclasses import dataclass
from typing import Dict
import math
import pandas as pd


@dataclass
class EnergyModelConfig:
    # Synthetic motor parameters
    rated_power_kw: float = 5.0
    rated_rpm: float = 1500.0
    base_efficiency: float = 0.88

    # Electrical / operating assumptions
    reference_voltage_v: float = 230.0
    min_efficiency: float = 0.55

    # Configurable grid emission factor.
    # Replace with the factor appropriate for your chosen scenario/source.
    grid_emission_factor_kg_per_kwh: float = 0.70

    # Synthetic fault penalty at severity=1.0.
    # Penalty is interpolated linearly with severity.
    fault_penalties: Dict[str, float] = None

    def __post_init__(self):
        if self.fault_penalties is None:
            self.fault_penalties = {
                "Healthy": 0.00,
                "Belt Misalignment": 0.08,
                "Excessive Load": 0.12,
                "Mechanical Imbalance": 0.07,
                "Belt Slip": 0.15,
                "Bearing Degradation": 0.10,
            }


class ConveyorEnergyModel:
    def __init__(self, config: EnergyModelConfig = None):
        self.cfg = config or EnergyModelConfig()

    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

    def baseline_power(self, rpm: float, load_pct: float) -> float:
        """
        Synthetic healthy electrical power model.

        Mechanical demand scales approximately with load and speed.
        A small speed-dependent idle/friction term is included so that
        the model does not predict zero power at zero load.
        """
        load = self._clip(load_pct / 100.0, 0.0, 1.0)
        speed = max(rpm, 0.0) / self.cfg.rated_rpm

        # Synthetic mechanical demand.
        mechanical_kw = self.cfg.rated_power_kw * (
            0.08 * speed**2 + 0.92 * load * speed**1.15
        )

        # Voltage deviation produces a modest synthetic electrical
        # efficiency effect around the reference voltage.
        voltage_factor = 1.0 + 0.03 * abs(
            1.0 - self.cfg.reference_voltage_v / max(1.0, self.cfg.reference_voltage_v)
        )

        return mechanical_kw / self.cfg.base_efficiency * voltage_factor

    def calculate(
        self,
        rpm: float,
        load_pct: float,
        voltage_v: float = 230.0,
        fault: str = "Healthy",
        severity: float = 0.0,
        duration_s: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate energy and CO2 metrics for one simulation window.

        severity:
            0.0 = no fault effect
            1.0 = maximum configured synthetic fault effect
        """
        severity = self._clip(float(severity), 0.0, 1.0)

        baseline_kw = self.baseline_power(rpm, load_pct)

        max_penalty = self.cfg.fault_penalties.get(fault, 0.0)

        # Fault penalty grows with severity.
        fault_penalty = max_penalty * severity

        # Add a small voltage-deviation effect.
        voltage_ratio = voltage_v / self.cfg.reference_voltage_v
        voltage_effect = 1.0 + 0.04 * abs(voltage_ratio - 1.0)

        faulty_kw = baseline_kw * (1.0 + fault_penalty) * voltage_effect

        duration_h = max(0.0, duration_s) / 3600.0

        baseline_kwh = baseline_kw * duration_h
        faulty_kwh = faulty_kw * duration_h
        extra_kwh = max(0.0, faulty_kwh - baseline_kwh)

        ef = self.cfg.grid_emission_factor_kg_per_kwh

        baseline_co2 = baseline_kwh * ef
        faulty_co2 = faulty_kwh * ef
        extra_co2 = extra_kwh * ef

        power_penalty_pct = (
            (faulty_kw - baseline_kw) / baseline_kw * 100.0
            if baseline_kw > 0
            else 0.0
        )

        return {
            "rpm": float(rpm),
            "load_pct": float(load_pct),
            "voltage_v": float(voltage_v),
            "fault": fault,
            "severity": severity,
            "duration_s": float(duration_s),

            "baseline_power_kw": baseline_kw,
            "faulty_power_kw": faulty_kw,
            "power_penalty_pct": power_penalty_pct,

            "baseline_energy_kwh": baseline_kwh,
            "faulty_energy_kwh": faulty_kwh,
            "extra_energy_kwh": extra_kwh,

            "baseline_co2_kg": baseline_co2,
            "faulty_co2_kg": faulty_co2,
            "extra_co2_kg": extra_co2,
        }

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add energy/CO2 outputs to a simulator DataFrame."""
        required = {"rpm", "load_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        results = []

        for _, row in df.iterrows():
            results.append(
                self.calculate(
                    rpm=row["rpm"],
                    load_pct=row["load_pct"],
                    voltage_v=row.get("voltage_v", 230.0),
                    fault=row.get("fault", "Healthy"),
                    severity=row.get("severity", 0.0),
                    duration_s=row.get("duration_s", 1.0),
                )
            )

        energy_df = pd.DataFrame(results, index=df.index)

        # Avoid duplicate source columns when merging.
        output = df.copy()
        for col in energy_df.columns:
            if col not in output.columns:
                output[col] = energy_df[col]
            else:
                output[f"energy_{col}"] = energy_df[col]

        return output


if __name__ == "__main__":
    model = ConveyorEnergyModel()

    # Example single simulated window
    example = model.calculate(
        rpm=1750,
        load_pct=72,
        voltage_v=228,
        fault="Bearing Degradation",
        severity=0.68,
        duration_s=1.0,
    )

    print("\n=== ENERGY MODEL EXAMPLE ===")
    for key, value in example.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

    # Example batch integration
    sample = pd.DataFrame([
        {
            "rpm": 1500,
            "load_pct": 60,
            "voltage_v": 230,
            "fault": "Healthy",
            "severity": 0.0,
            "duration_s": 60,
        },
        {
            "rpm": 1750,
            "load_pct": 72,
            "voltage_v": 228,
            "fault": "Bearing Degradation",
            "severity": 0.68,
            "duration_s": 60,
        },
        {
            "rpm": 1800,
            "load_pct": 80,
            "voltage_v": 225,
            "fault": "Belt Slip",
            "severity": 0.75,
            "duration_s": 60,
        },
    ])

    results = model.calculate_batch(sample)

    print("\n=== BATCH RESULTS ===")
    print(
        results[
            [
                "fault",
                "severity",
                "baseline_power_kw",
                "faulty_power_kw",
                "power_penalty_pct",
                "baseline_energy_kwh",
                "faulty_energy_kwh",
                "extra_energy_kwh",
                "baseline_co2_kg",
                "faulty_co2_kg",
                "extra_co2_kg",
            ]
        ].to_string(index=False)
    )

    results.to_csv("energy_model_results.csv", index=False)
    print("\nSaved: energy_model_results.csv")
