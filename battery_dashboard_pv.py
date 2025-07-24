import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import locale
import datetime
#import test_daily_load


# Streamlit config
st.set_page_config(page_title="Battery storage simulation dashboard", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1f77b4;}
    .metric {background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .plot-box {border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #fff; margin-bottom: 1rem;}
    .metric-title {font-size: 1.8rem; color: #444;font-weight: bold}
    .metric-value {font-size: 1.7rem; color: #111}
    </style>
""", unsafe_allow_html=True)

# Set German locale for number formatting
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'deu_deu')
    except locale.Error:
        st.write(" ")

# ADDED: This is the key change to prevent plots from opening in new tabs
plotly_config = {
    'displayModeBar': False,
    'showTips': False,
    'displaylogo': False,
    'scrollZoom': True,
    'staticPlot': False
}
#from graphs import demand_charge



# .metric-value {font-size: 1.5rem; color: #111; font-weight: bold;}
st.title("🔋 ecoplanet Battery Storage Simulation Dashboard")

#--------------------- Helper Definitions -------------------------
# Initial battery configuration
battery_efficiency = 0.9
discharge_percentage = 0.001
demand_charge = 200
template_url = "https://docs.google.com/spreadsheets/d/1xJ3Lk8uy3X8piSt-IUxZgeVYqeOuRG9N/edit?usp=sharing&ouid=114799245841423325825&rtpof=true&sd=true"
battery_configs = {
    "Small": {"capacity": 90, "power": 92},
    "Medium": {"capacity": 215, "power": 100},
    "Large": {"capacity": 500, "power": 250}
}

# -------------------- Helper functions --------------------------

### Peak Shaving Simulation
def peak_shaving(load_data, threshold):
    peak_threshold = max(load_data) * (threshold / 100)
    optimized_load = np.where(load_data > peak_threshold, peak_threshold, load_data)
    return optimized_load



### New battery simulation
def battery_simulation_v02(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 20%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = 0.25  # 15-minute intervals
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    charge = []
    discharge = []
    soc_state = []

    for load in df["load"]:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # soc = max(soc - energy_used, reserve_energy)
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours
            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            charge.append(actual_charge_power)
            discharge.append(0)

        else:
            charge.append(0)
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df




### New battery simulation
def battery_simulation_vpv(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1):
    # Idee: Peak shifting: Ab ~Juni: 50 % der Kapazität für Load shifting reservieren
        # Zähler n=0, if n>15*24*30*5 ~~20.000:
            # total_capacity = battery_capacity if or(n<20.000, n>40000) else total_capacity =  capactiy_sommer = 0.5* total capacity
            # sommer_capacity = 0.5*total_capacity
    # Which columns do I need to properly simulate the battery - for peakshaving discharge & charge, peak shifting discharge & charge, total charge & discharge
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = 0.25  # 15-minute intervals
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            charge_pv.append(0)
            charge_grid.append(0)

        # --- CHARGING ---
        else:
            # 1. Charge from PV first (always allowed, up to available PV and battery limits)
            max_charge_power = (total_capacity - soc) / interval_hours  # what battery can take
            # FALLS es PV gibt - also pv_neg <0 ist - dann mit pv laden, maximal so viel wie geht. Sonst is pv_chaege_power 0
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            soc_available = total_capacity - soc
            if soc_available > 0 and load + pv_charge_power < threshold_kw:
                # How much more can be charged without exceeding threshold?
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(power_rating - pv_charge_power, max_grid_charge_power, soc_available / interval_hours)
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(0)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    return df

def battery_simulation_vpv_selfconsumption_oldf(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = 0.25  # 15-minute intervals
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(6, 10)

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            # Peak shaving as before
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # Default all PV charging to self-consumption, unless peak shaving is triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            soc_available = total_capacity - soc
            if soc_available > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(power_rating - pv_charge_power, max_grid_charge_power, soc_available / interval_hours)
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    battery_charge_grid_self = grid_charge_power  # Default all grid charging to self-consumption, unless peak shaving is triggered
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power

            # --- PV-rich period: Discharge for self-consumption even if below threshold ---
            if row['pv_rich'] and load > 0 and soc > reserve_energy:
                discharge_power = min(load, power_rating, (soc - reserve_energy) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, reserve_energy), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    return df


def optimize_battery_params(df, battery_capacity, power_rating, demand_charge_low, demand_charge_high, energy_charge_low, energy_charge_high):
    best_roi = -float('inf')
    best_params = {}

    # Define parameter ranges
    threshold_range = range(75, 95, 5)  # 60-90% in 5% steps
    base_reserve_range = [0.2, 0.3, 0.4]
    extra_reserve_range = [0.4, 0.5]

    for threshold in threshold_range:
        for base_reserve in base_reserve_range:
            for extra_reserve in extra_reserve_range:
                # Run simulation
                result_df = battery_simulation_vpv_selfconsumption(
                    df.copy(), battery_capacity, power_rating,
                    80, threshold, base_reserve=base_reserve,
                    extra_reserve=extra_reserve
                )

                vlh = df["grid_load_pv_bt"].sum() /4 / df["grid_load_pv_bt"].max()
                demand_charge_calc = demand_charge_low if vlh <2500 else demand_charge_high
                energy_charge_calc = energy_charge_high if vlh < 2500 else energy_charge_low

                # Calculate ROI
                roi = calculate_roi(result_df, demand_charge_calc, energy_charge_calc)

                if roi > best_roi:
                    best_roi = roi
                    best_params = {
                        'threshold_pct': threshold,
                        'base_reserve': base_reserve,
                        'extra_reserve': extra_reserve,
                        'roi': roi
                    }

    return best_params

def calculate_roi(df, demand_charge, energy_charge):
    savings_ps = (df["load_pv"].max() - df["grid_load_pv_bt"].max()) * demand_charge
    pv_selfcons_kwh = min(
        df['battery_charge_pv_selfcons'].sum() / 4,
        df['battery_discharge_selfcons'].sum() / 4
    )
    annual_savings_selfcons = pv_selfcons_kwh * energy_charge
    savings_total = savings_ps + annual_savings_selfcons

    return savings_total


def optimize_battery_params_working(df, battery_capacity, power_rating, demand_charge_low, demand_charge_high, energy_charge_low, energy_charge_high):
    best_roi = -float('inf')
    best_params = {}

    # Define parameter ranges
    threshold_range = range(75, 95, 5)  # 60-90% in 5% steps
    reserve_fraction_range = [0.2, 0.3, 0.4]

    for threshold in threshold_range:
        for reserve_fraction in reserve_fraction_range:
            # Run simulation
            result_df = battery_simulation_vpv_selfconsumption_working(
                df.copy(), battery_capacity, power_rating,
                80, threshold,
                reserve_fraction = reserve_fraction
            )

            vlh = df["grid_load_pv_bt"].sum() / df["grid_load_pv_bt"].max()
            demand_charge_calc = demand_charge_low if vlh <2500 else demand_charge_high
            energy_charge_calc = energy_charge_high if vlh < 2500 else energy_charge_low

            # Calculate ROI
            roi = calculate_roi(result_df, demand_charge_calc, energy_charge_calc)

            if roi > best_roi:
                best_roi = roi
                best_params = {
                    'threshold_pct': threshold,
                    'reserve_fraction': reserve_fraction,
                    'roi': roi
                }

    return best_params


def battery_simulation_vpv_selfconsumption_working(
    df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1, reserve_fraction=0.3):
    ################### ELAS CHANGES ###############
    max_load_jump = df["load"].diff().clip(lower=0).max()
    #reserve_fraction = max_load_jump / df["load"].max()
    #reserve_fraction = max_load_jump /4 / battery_capacity
    reserve_amount = max_load_jump /4

    ##########################
    total_capacity = battery_capacity  # kWh
    true_min_soc = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    reserve_soc = true_min_soc + (total_capacity - true_min_soc) * reserve_fraction
#    reserve_soc = true_min_soc + reserve_amount

    soc = total_capacity  # start fully charged
    interval_hours = 0.25  # 15-minute intervals
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(start_month_pv, end_month_pv)
    #df['pv_rich'] = df['month'].isin([1,2,3,4, 5, 6, 8, 9,12]) // LB Profil

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []
    soc_reserve_state = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > true_min_soc:
            # Peak shaving: allow discharge down to technical minimum
            power_needed = load - threshold_kw
            max_discharge_power = (soc - true_min_soc) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first (up to available space)
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # All PV charging is for self-consumption unless peak shaving triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            # --- CHANGED LOGIC HERE ---
            if row['pv_rich']:
                # In PV-rich (summer): only fill grid up to reserve
                soc_available_for_grid = reserve_soc - soc
            else:
                # In winter: allow grid charging up to 100%
                soc_available_for_grid = total_capacity - soc
            # -------------------------
            if soc_available_for_grid > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(
                    power_rating - pv_charge_power,
                    max_grid_charge_power,
                    soc_available_for_grid / interval_hours
                )
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    if row['pv_rich']:
                        battery_charge_grid_self = 0  # grid charging is only for reserve (peak shaving), not self-consumption
                        battery_charge_grid_peak = grid_charge_power
                    else:
                        battery_charge_grid_self = 0  # in winter, all grid charging is for peak shaving
                        battery_charge_grid_peak = grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
                    battery_charge_grid_self = 0
                    battery_charge_grid_peak = 0
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power
                battery_charge_grid_self = 0
                battery_charge_grid_peak = 0

            # --- PV-rich period: Discharge for self-consumption down to reserve_soc (not technical min) ---
            if row['pv_rich'] and load > 0 and soc > reserve_soc:
                # Only discharge self-consumption down to reserve_soc
                discharge_power = min(load, power_rating, (soc - reserve_soc) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, true_min_soc), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(battery_charge_grid_peak)

        optimized.append(grid_load)
        soc_state.append(soc)
        soc_reserve_state.append(reserve_soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    df["soc_reserve"] = soc_reserve_state
    return df


def battery_simulation_vpv_selfconsumption(
    df,
    battery_capacity,
    power_rating,
    depth_of_discharge,
    threshold_pct,
    battery_efficiency=1,
    base_reserve=0.3,
    extra_reserve=0.,
    reserve_window_hours=24
):
    total_capacity = battery_capacity  # kWh
    true_min_soc = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = 0.25  # 15-minute intervals

    # Calculate threshold for peak shaving
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(2, 10)

    # Calculate the largest positive jump in net load (after PV)
    max_load_jump = df["load"].diff().clip(lower=0).max()
    jump_reserve = max_load_jump * interval_hours
    min_reserve_fraction = 0.2

    # Calculate rolling peak and dynamic reserve fraction
    window_intervals = int(reserve_window_hours / interval_hours)
    df['rolling_peak'] = df['load'].rolling(window=window_intervals, min_periods=1).max()
    max_peak = df['load'].max()
    df['reserve_fraction_dynamic'] = base_reserve + extra_reserve * (df['rolling_peak'] / max_peak)
    df['reserve_fraction_dynamic'] = df['reserve_fraction_dynamic'].clip(upper=0.9)

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []
    soc_reserve_state = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv_neg = row["load_pv_neg"]
        pv = row["pv"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # Dynamic reserve for this timestep
        reserve_fraction = row['reserve_fraction_dynamic']
        #reserve_soc = true_min_soc + (total_capacity - true_min_soc) * reserve_fraction

        reserve_soc = max(true_min_soc + jump_reserve,
                          true_min_soc + (total_capacity - true_min_soc) * min_reserve_fraction,
                          true_min_soc + (total_capacity - true_min_soc) * reserve_fraction)


        # --- DISCHARGING: Peak shaving ---
        if load > threshold_kw and soc > true_min_soc:
            # Peak shaving: allow discharge down to technical minimum
            power_needed = load - threshold_kw
            max_discharge_power = (soc - true_min_soc) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first (up to available space)
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # All PV charging is for self-consumption unless peak shaving triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            if row['pv_rich']:
                soc_available_for_grid = reserve_soc - soc
            else:
                soc_available_for_grid = total_capacity - soc

            if soc_available_for_grid > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(
                    power_rating - pv_charge_power,
                    max_grid_charge_power,
                    soc_available_for_grid / interval_hours
                )
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    if row['pv_rich']:
                        battery_charge_grid_self = 0  # grid charging is only for reserve (peak shaving), not self-consumption
                        battery_charge_grid_peak = grid_charge_power
                    else:
                        battery_charge_grid_self = 0  # in winter, all grid charging is for peak shaving
                        battery_charge_grid_peak = grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
                    battery_charge_grid_self = 0
                    battery_charge_grid_peak = 0
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power
                battery_charge_grid_self = 0
                battery_charge_grid_peak = 0

            # --- PV-rich period: Discharge for self-consumption down to reserve_soc (not technical min) ---
            if row['pv_rich'] and load > 0 and soc > reserve_soc:
                # Only discharge self-consumption down to reserve_soc
                discharge_power = min(load, power_rating, (soc - reserve_soc) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, true_min_soc), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(battery_charge_grid_peak)

        optimized.append(grid_load)
        soc_state.append(soc)
        soc_reserve_state.append(reserve_soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    df["soc_reserve"] = soc_reserve_state
    df["reserve_fraction_dynamic"] = df['reserve_fraction_dynamic']

    return df


def battery_simulation_ps_with_pv(df, load_series, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 20%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = 0.25  # 15-minute intervals
    peak = load_series.max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    discharge = []
    soc_state = []
    battery_load_from_pv = []

    for load in load_series:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # soc = max(soc - energy_used, reserve_energy)
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours

            #if

            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            discharge.append(0)

        else:
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load_pv_bt"] = optimized
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df


#--------------------------------------------- FILE PROCESSING ------------------------------------------------------

############################################# File Processing #############################################
######### Uploader #########
with st.sidebar:
    st.subheader("File upload")
    st.write("Find a template [here](%s)" % template_url)
    uploaded_file = st.file_uploader("📁 Upload your load profile data (XLSX). Please make sure the file has two columns 'timestamp' and 'load'.", type=["xlsx"])

    pv_total = st.number_input("☀️ PV (kw peak)", 0)
    st.write(f"**Netznutzungsentgelte**")
    st.write("**<2500 VLH**")
    demand_charge_low = st.number_input("💰⬇️ Leistungspreis <2500 VLH (in €/kW)", min_value=None, value=20.00)
    energy_charge_high_input = st.number_input("💲 Arbeitspreis <2500 VLH (in ct/kWh)",min_value=None, value=10.00)
    st.write(f"**\>=2500 VLH**")
    demand_charge_high = st.number_input("💰️⬆️ Leistungspreis >=2500 VLH (in €/kW)", min_value=None, value=200.00)
    energy_charge_low_input = st.number_input("💲 Arbeitspreis >=2500 VLH (in ct/kWh)", min_value=None, value= 1.00)

    energy_charge_high = energy_charge_high_input /100
    energy_charge_low = energy_charge_low_input /100

    ## Handling
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = df[~df['timestamp'].duplicated(keep=False)]
        # Try to auto-detect datetime columns
        col_map = {col.lower(): col for col in df.columns}
        date_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["date", "day", "tag"])), None)
        time_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["time", "hour", "timestamp", "zeit","uhrzeit", "timestamps", "datum"])), None)
        load_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["kw", "load", "value", "value_kw", "power", "entnahme", "last", "netzbezug"])), None)

        if date_col and time_col:
            df["timestamp"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), format='%d.%m.%Y %H:%M')
        elif time_col:
            try:
                df["timestamp"] = pd.to_datetime(df[time_col], dayfirst=True)
            except Exception as e1:
                try:
                    df["timestamp"] = pd.to_datetime(df[time_col], format="mixed", dayfirst=True)
                except Exception as e2:
                    try:
                        df["timestamp"] = pd.to_datetime(df[time_col], format="ISO8601")
                    except Exception as e3:
                        raise ValueError(f"Failed to parse datetime. Errors:\n1. {e1}\n2. {e2}\n3. {e3}")
        elif date_col:
            df["timestamp"] = pd.to_datetime(df[date_col], dayfirst = True)
        else:
            df["timestamp"] = pd.to_datetime(df.iloc[:, 0], dayfirst = True)  # fallback

        if load_col:
            df["load"] = df[load_col]

        # Create date columns
        df["datetime"] = df["timestamp"]
        df["Date"] = df["timestamp"].dt.date
    #    df["week"] = df["timestamp"].dt.strftime("%G-W%V-%u")
        df["Week"] = df["timestamp"].dt.isocalendar().week
        df["Month"] = df["timestamp"].dt.strftime("%Y-%m")
        df["Year"] = df["timestamp"].dt.strftime("%Y")
        df["Weekday"] = df["timestamp"].dt.day_name()
        df["time"] = df["timestamp"].dt.time



        with st.expander ("### 📊 Raw data preview"):
            st.write(df.head())




################################################################################ LOAD PROFILE INFORMATION ######################################################

tab_analyse, tabsimulation, tabsolar = st.tabs(["📈 Lastgang Analyse", "🔋 Batterie-Wizzard", "☀️ PV Simulation"])

with ((tab_analyse)):
    if not uploaded_file:
        with st.container(border = True):
            st.subheader("⚠️ Please upload your load profile in the left sidebar.")
            st.write("**Ensure that the file contains two columns with a timestamp and the load in kW.**")
            st.write("**In case you encounter an error, ensure that the naming of the columns is 'timestamp' and 'load'.**")
            st.write("A template can be downloaded [here](%s)" % template_url)
            st.write("Please note: The ideal format for the timestamp is dd.mm.yyyy hh:mm")

    if uploaded_file:
        ### -------------------------------------------------- PV ------------------------------------------------------------ ####
        MAGIC_YEARLY_PV_MULTIPLIER = 1000
        INTERVAL_HOURS = 0.25

        df_org = df.copy()
        df_pv = pd.read_csv("C:/Users/mgr/Documents/python/battery-dashboard/battery-system-analysis/Data/solar_data_de_small.csv")

        # Dataframe to store PV distribution
        df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], format="%d.%m.%y %H:%M")
        df_pv["yearly_production_kw"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER / INTERVAL_HOURS  # result in kW
        df_pv["yearly_production_kwh"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER # result in kwh

        # Create load with PV in "load_pv"
        df["timestamp_index"] = df["timestamp"]
        df_pv["timestamp_index"] = df_pv["timestamp"]
        df = df.set_index('timestamp_index')
        df_pv = df_pv.set_index('timestamp_index')
        df["pv"] = df_pv["yearly_production_kw"]
        df["load_org"] = df["load"]
        df["load_pv"] = df["load"].sub(df_pv["yearly_production_kw"], fill_value=0)

        # Positive Werte (Netzbezug) und negative Werte (Einspeisung) berücksichtigen
        df["load_pv_pos"] = df["load_pv"].clip(lower=0)
        df["load_pv_neg"] = df["load_pv"].clip(upper=0)
        positive_load_pv = df["load_pv"][df["load_pv"] > 0]
        negative_load_pv = df["load_pv"][df["load_pv"] < 0]

        # df ist original - df_with_pv ist mit PV
        df_with_pv = df.copy()
        df_with_pv["load"] = df["load_pv"]

        ## ------------------------------------------    VARIABLEN    --------------------------------------------
        total_entries = len(df)
        avg_load = df["load"].mean()
        min_load = df["load"].min()
        peak_load = df["load"].max()
        total_energy_kwh = df["load"].sum() /4  # assuming 15-min intervals
        total_energy_Mwh = df["load"].sum() / 4 / 1000  # assuming 15-min intervals
        volllaststunden = df["load"].sum() / 4 / peak_load

        # PV METRICS
        peak_load_pv = df_with_pv["load"].max()
        energy_consumed_pv = positive_load_pv.sum() / 4 / 1000
        vollaststunden_pv = positive_load_pv.sum() / 4 / positive_load_pv.max()
        pv_energy_exported = -df["load_pv_neg"].sum() * INTERVAL_HOURS  # Eingespeiste Energie (PV-Überschuss)
        pv_generated = df_pv["yearly_production_kw"].sum() * INTERVAL_HOURS  # PV-Daten (gesamte PV-Erzeugung)
        pv_self_consumed = pv_generated - pv_energy_exported  # Eigenverbrauch PV (direkt genutzte PV-Energie)
        pv_self_consumption_ratio = pv_self_consumed / pv_generated * 100  # Eigenverbrauchsquote
        autarky_ratio = pv_self_consumed / total_energy_kwh * 100  # Autarkiegrad

        ## -----------------------------------------------------------------------------------------------------

        st.header("Lastgang Übersicht")

        st.subheader("📈 Statistik")
        st.write(f"Von **{df["timestamp"].min()}** bis **{df["timestamp"].max()}**")

        col1, col2, col3, col4, col5 = st.columns(5)


        #col1.markdown("<div class='metric'><div class='metric-title'> 📅 Total datapoints </div><div class='metric-value'>" + f"{total_entries:,.0f}" + "</div></div>",unsafe_allow_html=True)
        col1.markdown(
            "<div class='metric'><div class='metric-title'> 🔌 Gesamtverbrauch </div><div class='metric-value'>" + f"{total_energy_Mwh:,.0f} MWh" + "</div></div>",
            unsafe_allow_html=True)
        col2.markdown(
            "<div class='metric'><div class='metric-title'> ⚡  Durschn. Last </div><div class='metric-value'>" + f"{avg_load:,.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col3.markdown(
            "<div class='metric'><div class='metric-title'> 🔺 Max. Last </div><div class='metric-value'>" + f"{peak_load:,.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col4.markdown(
            "<div class='metric'><div class='metric-title'> 🔻 Min. Last </div><div class='metric-value'>" + f"{min_load:.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col5.markdown(
            "<div class='metric'><div class='metric-title'> ⏺️ Volllaststunden </div><div class='metric-value'>" + f"{(total_energy_kwh / peak_load ):.0f} h" + "</div></div>",
            unsafe_allow_html=True)


        st.write("\n")

        tab_param, tab_agg, tab_solar, tab_toppeaks, tab_loadduration  = st.tabs(["Parameter", "Aggregierte Ansicht", "PV Analyse", "Lastspitzen-Analyse", "Lastdauerkurve"])

        with tab_param:
            col1, col2 = st.columns([1,2])
            with col1:
                with st.container(border=True):
                    st.subheader("💶 Netznutzungsentgelte")
                    st.write("**<2500 Vollnutzungsstunden**")
                    col3, col4 = st.columns(2)
                    col3.metric("Leistungspreis", f"{demand_charge_low:,.2f} €/kW")
                    col4.metric("Arbeitspreis", f"{energy_charge_high_input:,.2f} ct/kW")

                    st.write("**\>=2500 Vollnutzungsstunden**")
                    col3,col4 = st.columns(2)
                    col3.metric("Leistungspreis", f"{demand_charge_high:,.2f} €/kW")
                    col4.metric("Arbeitspreis", f"{energy_charge_low_input:,.2f} ct/kW")
                    st.subheader("☀️ PV Anlage")
                    st.metric("Leistung ", f"{pv_total:,.0f} kWp")


        with tab_agg:

            fig_overview_gesamt = px.line(df, x="timestamp", y="load",
                                   title=f"Lastprofil in {df["timestamp"].dt.year.min()}",
                                   labels={"timestamp": "Zeit", "load": "Last (kW)"}
                                         )
            fig_overview_gesamt.update_layout(height=400, xaxis_title="Zeit")
            st.plotly_chart(fig_overview_gesamt, use_container_width=True, config=plotly_config)


        with tab_solar:

            fig_pv = go.Figure()

            # Add PV load  in yellow
            fig_pv.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df_pv["yearly_production_kw"],
                mode='lines',
                name='PV Erzeugung',
                line=dict(color='yellow')
            ))

            fig_pv.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['load'],
                mode='lines',
                name='Lastgang',
                line=dict(color='black')
            ))

            # Add grid load  in green
            fig_pv.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['load_pv'],
                mode='lines',
                name='Lastgang mit PV',
                line=dict(color='green')
            ))

            # Add Einspeisung
            fig_pv.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df["load_pv_neg"],
                mode='lines',
                name='Eingespeiste Energie',
                line=dict(color='orange')
            ))

            # Add Netzbezug
            fig_pv.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df["load_pv_pos"],
                mode='lines',
                name='Netzbezug',
                line=dict(color='blue')
            ))


            # Optional layout tweaks
            fig_pv.update_layout(
                title='Lastgang mit und ohne PV',
                xaxis_title='Zeit',
                yaxis_title='Last (kW)',
                template='simple_white'
            )

            with st.container(border=True):
                col1_pv, col2_pv, col3_pv, col4_pv, col5_pv, col6_pv = st.columns(6)

                with col1_pv:
                    st.header("Ohne PV ")
                    st.header("Mit PV ", help="Nur Netzbezug")
                with col2_pv:
                    st.metric("Energieverbrauch", f"{total_energy_Mwh:,.0f} MWh")
                    st.metric("Netto Energieverbrauch", f"{positive_load_pv.sum() / 4 / 1000:,.0f} MWh", f"{-(total_energy_Mwh-(positive_load_pv.sum() / 4 / 1000)):,.0f} MWh", delta_color="inverse")
                with col3_pv:
                    st.metric("Spitzenlast", f"{df["load"].max():,.0f} kW")
                    st.metric("Spitzenlast", f"{positive_load_pv.max():,.0f} kW", f"{-(peak_load-positive_load_pv.max()):,.0f} kW", delta_color="inverse")
                with col4_pv:
                    st.metric("Volllaststunden", f"{(volllaststunden):,.0f} h")
                    st.metric("Volllaststunden", f"{vollaststunden_pv:,.0f} h")
                with col5_pv:
                    st.metric(f" ","")
                    st.metric(f" ","")
                    st.metric("Bezug aus PV", f"{(total_energy_Mwh - positive_load_pv.sum() / 4 / 1000):,.0f} MWh")
                with col6_pv:
                    st.metric(f" ","")
                    st.metric(f" ","")
                    st.metric("Netzeinspeisung", f"{-(negative_load_pv.sum() / 4 / 1000):,.0f} MWh")

            with st.container(border=True):
                col1_pv, col2_pv, col3_pv, col4_pv, col5_pv, col6_pv = st.columns(6)

                with col1_pv:
                    st.header("PV Daten")
                with col2_pv:
                    st.metric("Leistung", f"{pv_total:,.0f} kWp")
                with col3_pv:
                    st.metric("Erzeugte Energie", f"{df_pv["yearly_production_kw"].sum() / 4 / 1000:,.0f} MWh")
                with col4_pv:
                    st.metric("Eigenverbrauchsquote PV", f"{pv_self_consumption_ratio:,.0f} %")
                with col5_pv:
                    st.metric("Autarkiegrad", f"{(df["load"].sum() - (positive_load_pv.sum()))/(df["load"].sum() )*100:,.0f} %", help="Anteil des Gesamtverbrauchs, der durch PV gedeckt wurde")

            st.plotly_chart(fig_pv, use_container_width=True, config=plotly_config)


        with tab_toppeaks:
            st.subheader("🚩  Spitzenlasten im Lastgangprofil")
            df_peaks = df.copy()

            col1, col2, col3 = st.columns([2, 3, 6], vertical_alignment="top")
            with col1:
                with st.container(border = True):
                    # Let user select number of entries shown
                    n_peaks = st.number_input("Anzahl der Spitzen", min_value=1, value=30)

                    top_peaks = df_peaks.nlargest(n_peaks, "load").reset_index(drop=True)[["timestamp", "Date", "time",  "Weekday", "load"]]
                    st.metric("Höchster Wert", f"{top_peaks['load'].max():,.0f} kW")
                    st.metric("Niedrigster Wert", f"{top_peaks['load'].min():,.0f} kW")

            with col2:
                st.write(top_peaks[["Date", "time", "Weekday","load"]])

            with col3:
                year = top_peaks["timestamp"].dt.year.min()
                # Define start and end of the year
                start = datetime.datetime(year, 1, 1)
                end = datetime.datetime(year, 12, 31, 23, 00, 00)

                fig_top20 = px.bar(top_peaks.sort_values("load"), x="timestamp", y="load",
                                   title=f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten",
                                   labels={"load": "Last (kW)", "timestamp": "Zeit"})
                fig_top20.update_layout(xaxis_tickformat="%b",xaxis_title=f"{year}", xaxis_tickangle=-45, xaxis=dict(nticks=20, range=[start, end]))
                st.plotly_chart(fig_top20, use_container_width=True)


            ################### +++++++++++++++++++++++++++++SUN+++++++++++++++++++++++++++++ ###################
            st.subheader("☀️ Spitzenlasten mit PV")
            df_pv_peaks = df.copy()
            col1, col2, col3 = st.columns([2, 3, 6], vertical_alignment="top")
            with col1:
                with st.container(border = True):
                    # Let user select number of entries shown
                    #n_peaks = st.number_input("Anzahl der Spitzen", min_value=1, value=30)

                    top_peaks_pv = df_pv_peaks.nlargest(n_peaks, "load_pv").reset_index()[["timestamp", "Date", "time", "Weekday", "load_pv"]]
                    st.metric("Höchster Wert", f"{top_peaks_pv['load_pv'].max():,.0f} kW")
                    st.metric("Niedrigster Wert", f"{top_peaks_pv['load_pv'].min():,.0f} kW")

            with col2:
                st.write(top_peaks_pv[["Date","time","Weekday","load_pv"]])

            with col3:
                fig_top20_pv = px.bar(top_peaks_pv.sort_values("load_pv"), x="timestamp", y="load_pv",
                                   title=f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten",
                                   labels={"load_pv": "Last (kW)", "Timestamp": "Time"})
                fig_top20_pv.update_layout(xaxis_tickformat="%b",xaxis_title=f"{year}", xaxis_tickangle=-45, xaxis=dict(nticks=20, range=[start, end]))
                st.plotly_chart(fig_top20_pv, use_container_width=True)


            st.header("")
            col1, col3, col_leer = st.columns([3 , 8,1], vertical_alignment="top")
            with col1:
                st.subheader(f"🚩☀️ {n_peaks} Spitzenlasten im Vergleich mit/ohne PV")
                with st.container(border = True):
                    col_a, col_b = st.columns([2, 2])
                    #col_aa.subheader(" ")
                    #col_aa.write(" ")
                    #col_aa.subheader("Höchster Wert")
                    #col_aa.write(" ")
                    #col_aa.write(" ")
                    #col_aa.subheader("Niedrigster Wert")
                    col_a.subheader("Ohne PV")
                    #col_a.write("---")
                    col_a.metric("Spitzenlast ohne PV", f"{df['load'].max():,.0f} kW")
                    col_a.metric("Niedrigster Wert", f"{top_peaks['load'].min():,.0f} kW")

                    col_b.subheader("Mit PV")
                    #col_b.write("---")
                    col_b.metric("Spitzenlast mit PV", f"{top_peaks_pv['load_pv'].max():,.0f} kW")
                    col_b.metric("Niedrigster Wert", f"{top_peaks_pv['load_pv'].min():,.0f} kW")

            with col3:
                st.subheader(f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten unter Berücksichtigung einer PV-Anlage")

                # Assume top_peaks and top_peaks_pv as before
                timestamps_pv = set(top_peaks_pv['timestamp'])

                # Assign colors and labels
                top_peaks['color'] = top_peaks['timestamp'].apply(
                    lambda ts: 'red' if ts in timestamps_pv else 'green'
                )
                top_peaks['label'] = top_peaks['timestamp'].apply(
                    lambda ts: 'Nicht vermiedene Spitzenlasten' if ts in timestamps_pv else 'Vermiedene Spitzenlasten'
                )

                # Color map for German labels
                color_discrete_map = {
                    'Nicht vermiedene Spitzenlasten': 'red',
                    'Vermiedene Spitzenlasten': 'green'
                }

                fig_combined = px.bar(
                    top_peaks.sort_values("load"),
                    x="timestamp",
                    y="load",
                    color="label",  # Use the German label column for legend
                    color_discrete_map=color_discrete_map,
                    title=f"",
                    labels={"load": "Power (kW)", "timestamp": "Zeit", "label": "Legende"}
                )

                fig_combined.update_layout(
                    xaxis_tickformat="%b",
                    xaxis_title="Zeitpunkt",
                    xaxis_tickangle=-45,
                    xaxis=dict(nticks=20, range=[start, end])
                )

                st.plotly_chart(fig_combined, use_container_width=True)


            st.header("🔍 Detail-Analyse der Spitzenlasten")

            col_context_left, col_context_right, col_context_buffer = st.columns([1, 3, 5])
            with col_context_left:
#                st.subheader(f"\n")
                st.subheader("Spitzenlast\n\n")
                selected_timestamp = st.selectbox("Bitte timestamp auswählen", top_peaks["timestamp"].astype(str))
                # Convert back to datetime if needed
                selected_timestamp = pd.to_datetime(selected_timestamp)
                # Sort full df by timestamp
                df_sorted = df_peaks.copy()
                df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

                # Find index of selected timestamp in full sorted DataFrame
                selected_index = df_sorted[df_sorted["timestamp"] == selected_timestamp].index

                if not selected_index.empty:
                    idx = selected_index[0]
                    # Get 10 rows before and after
                    context_df = df_sorted.iloc[max(0, idx - 10): idx + 11]

                    with col_context_right:
                        st.subheader(f"📊 10 Lasten vor und nach Spitzenlast \n\n **(am {selected_timestamp})**")
                       # st.dataframe(context_df[["date","Weekday","load"]])
                    #st.dataframe(context_df)
                        def highlight_selected_row(row):
                            if row["timestamp"] == selected_timestamp:
                                return ["background-color: #fdd835"] * len(row)  # yellow highlight
                            else:
                                return [""] * len(row)

                        # Apply the style
                        styled_context_df = context_df[["timestamp","Weekday","load"]]
                        styled_context_df = styled_context_df.style.apply(highlight_selected_row, axis=1)

                        # Show it in Streamlit
                        st.dataframe(styled_context_df)
                else:
                    st.warning("Selected timestamp not found in full data.")

            with col_context_buffer:
                st.subheader("Lastkurve\n\n")
                selected_day = selected_timestamp.date()
                df_day = df_peaks[df_peaks["timestamp"].dt.date == selected_day]
                fig_day = px.line(df_day, x="timestamp", y="load", title=f"🔋 Lastkurve am ausgewählten Tag ({selected_day})")
                fig_day.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_day, use_container_width=True)

        with tab_loadduration:
            st.subheader("📉 Lastdauerkurve")
            sorted_loads = df_peaks["load"].sort_values(ascending=False).reset_index(drop=True)
            fig_load_duration = px.line(sorted_loads, title="🔺 Lastdauerkurve (ohne PV)")
            fig_load_duration.update_layout(yaxis_title="Last (kW)", xaxis_title="Anzahl Stunden (sorriert nach Last)")
            st.plotly_chart(fig_load_duration, use_container_width=True)
            st.write("\n \n")

            st.subheader("📉☀️  Lastdauerkurve mit PV ️️☀️")
            sorted_loads_pv = df_peaks["load_pv"].sort_values(ascending=False).reset_index(drop=True)
            fig_load_duration_pv = px.line(sorted_loads_pv, title="🔺 Lastdauerkurve (mit PV)")
            fig_load_duration_pv.update_layout(yaxis_title="Last (kW)", xaxis_title="Anzahl Stunden (sorriert nach Last)")
            st.plotly_chart(fig_load_duration_pv, use_container_width=True)
            st.write("\n \n")

            st.subheader("Lastdauerkurve mit und ohne PV ")
            df_duration = pd.DataFrame({
                "Ohne PV": sorted_loads,
                "Mit PV": sorted_loads_pv
            })

            # Create figure
            fig = go.Figure()

            # Add Ohne PV (black)
            fig.add_trace(go.Scatter(
                y=df_duration["Ohne PV"],
                x=df_duration.index,
                mode='lines',
                name='Ohne PV',
                line=dict(color='black')
            ))

            # Add Mit PV (orange)
            fig.add_trace(go.Scatter(
                y=df_duration["Mit PV"],
                x=df_duration.index,
                mode='lines',
                name='Mit PV',
                line=dict(color='orange')
            ))

            fig.update_layout(
                title="🔺 Lastdauerkurve mit und ohne PV",
                yaxis_title="Last (kW)",
                xaxis_title="Stunden (sortiert nach Last)",
                legend_title="Legende"
            )

            st.plotly_chart(fig, use_container_width=True)
            ##########################################



        # Add additional information about the load profile
        with st.expander("Additional load profile statistics"):
            st.write(f"**Data Points:** {len(df)}")
            st.write(f"**Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            st.write(f"**Median Load:** {df['load'].median():.2f} kW")
            st.write(f"**Standard Deviation:** {df['load'].std():.2f} kW")
            st.write(f"**Load Factor:** {(avg_load / peak_load * 100):.2f}%")


#################################################################################################################################################################

with tabsimulation:
    # ----------------------------------- PARAMETER EINGABE LINKS ----------------------------------------
    if uploaded_file:
        # Create test for battery
        #df_bt_org = df.copy()
        df_exp_2 = df.copy()
        df_exp_pv = df_with_pv.copy()

        st.header("🔋 📉 Battery Simulation Dashboard")
       # st.warning("**Disclaimer**: This dashboard is intended for **indicative purposes only**. "
       #            "All calculations and results are based on **simplified assumptions** and **should not be interpreted as precise or reliable forecasts.** "
       #            "The tool provides no guarantee of accuracy, performance, or financial outcomes. Users must validate all assumptions independently and conduct a **thorough technical and financial analysis** before making any decisions based on the results shown. "
       #            "The creators of this dashboard accept no liability for any actions taken based on its outputs.")

        # ----------- Battery configurations -----------------------
        battery_lifetime = 10

        if 'battery_capacity' not in st.session_state:
            st.session_state.battery_capacity = 215

        if 'power_rating' not in st.session_state:
            st.session_state.power_rating = 100

        if 'battery_cost_per_kwh' not in st.session_state:
            st.session_state.battery_cost_per_kwh = 220

        battery_capacity = 0
        power_rating = 0
        battery_cost_per_kwh = 220

        # Functions to update battery values
        def set_small():
            st.session_state.battery_capacity = 90
            st.session_state.power_rating = 92
            st.session_state.battery_cost_per_kwh = 400

        def set_medium():
            st.session_state.battery_capacity = 215
            st.session_state.power_rating = 100
            st.session_state.battery_cost_per_kwh = 220

        def set_large():
            st.session_state.battery_capacity = 430
            st.session_state.power_rating = 200
            st.session_state.battery_cost_per_kwh = 220

        def set_xlarge():
            st.session_state.battery_capacity = 1000
            st.session_state.power_rating = 500
            st.session_state.battery_cost_per_kwh = 280


        def set_custom():
            # This just activates the custom input section
            st.session_state.show_custom = True


        col_battery_specs, col_battery_simulation = st.columns([1, 5])

        with col_battery_specs:

            with st.container(border=True):

                st.markdown("### 🔧🔋 Batterie")
                with st.expander("Auswahl"):
                    # Create simple buttons stacked vertically
                    st.button("Voltfang (90 kWh / 92 kW)", on_click=set_small)
                    st.button("Fox (215 kWh / 100 kW)", on_click=set_medium)
                    st.button("2 Fox (430 kWh / 200 kW)", on_click=set_large)
                    #st.button("Extra large (1MWh / 500 kW)", on_click=set_xlarge)

                    if 'show_custom' not in st.session_state:
                        st.session_state.show_custom = False
                    st.button("Custom Configuration", on_click=set_custom)

                    if st.session_state.show_custom:
                        st.session_state.battery_capacity = st.number_input("Custom Battery Capacity (kWh)", min_value=1, value=st.session_state.battery_capacity)

                        st.session_state.power_rating = st.number_input(
                            "Custom Power Rating (kW)",
                            min_value=1,
                            value=st.session_state.power_rating
                        )

                    # Display the current values
                st.write(f"Gewählte Kapazität: **{st.session_state.battery_capacity} kWh**")
                st.write(f"Gewählte Leistung: **{st.session_state.power_rating} kW**")
                st.write("---")
                st.write("**Kostenannahmen**")
                st.session_state.battery_cost_per_kwh = st.number_input("Batterie Kosten pro kWh (€/kWh)", min_value=100, value=st.session_state.battery_cost_per_kwh)
                system_cost_multiplier = st.number_input("Systemkosten Zusatzfaktor", min_value=1.0, value=1.2)

            ## --------------------------------- PEAK REDUCTION --------------------------------------------------------------------
            with st.container(border=True):
                st.subheader("🔻 Spitzenreduktion")
                st.write(f"Max. Reduktion mit gewählter Batterie: **{st.session_state.power_rating}kW**")

                value_peak_reduction = st.number_input("Reduktion um (kW) ",
                                                       0,
                                                       int(peak_load) * 1000000,
                                                       st.session_state.power_rating if st.session_state.power_rating < int(0.3 * peak_load) else int(0.3 * peak_load))

                calculated_peakshaving_threshold = (peak_load - value_peak_reduction) / peak_load *100
                if pv_total > 0:
                    st.write(f"Ziel-Spitzenlast ohne PV: {(peak_load - value_peak_reduction):.0f}kW")
                    st.write(f"Ziel-Spitzenlast mit PV: {(peak_load_pv - value_peak_reduction):.0f}kW")
                st.write(f"Anteil Spitzenlast : {100-calculated_peakshaving_threshold:.0f}%")

            ## -------------------------------------------  FINANCIAL ASSUMPTIONS ----------------------------------------------------------------------------


            with st.container(border=True):
                st.markdown("### 💰 Netzentgelte", help="Bitte in Seitenleiste links eingeben")
                with st.expander("Details"):
                    st.write(f"**\<= 2500 VLH**")
                    st.write(f"Arbeitspreis: {energy_charge_high} ct/kWh, Leistungspreis: {demand_charge_low} €/kw")
                    st.write(f"**\> 2500 VLH**")
                    st.write(f"Arbeitspreis: {energy_charge_low} ct/kWh, Leistungspreis > 2500 VLH: {demand_charge_high} €/kw")

            ## -------------------------------------------  MODIFICATON MONTHS ----------------------------------------------------------------------------
            with st.container(border=True):
                st.markdown("### 🌻 Einstellung PV Nutzung")
                start_month_pv = st.number_input("Start (Monat)", value=3, min_value=1, max_value=13)
                end_month_pv = st.number_input("Ende (Monat)", value=10, min_value=0, max_value=13)


            ###############################################################################################################################

        # ----------------------------------- MAIN WINDOW RECHTS  ----------------------------------------
        with col_battery_simulation:
            # -----------------------------        PEAKSHAVING METRICS CALCULATION   -----------------------------------------------------

            df_bt_ps = battery_simulation_v02(df.copy(), st.session_state.battery_capacity, st.session_state.power_rating, 90, calculated_peakshaving_threshold)

            peak_before = peak_load
            peak_after_bt_ps = df_bt_ps["grid_load"].max()
            peak_reduction_bt_ps = peak_before - peak_after_bt_ps

            volllaststunden_bt_ps = df_bt_ps["load"].sum() / 4 / df_bt_ps["grid_load"].max()

            demand_charge = demand_charge_high if volllaststunden_bt_ps > 2500 else demand_charge_low
            annual_savings_actual = peak_reduction_bt_ps * demand_charge




            # --------------------------------   OTHER METRICS CALCULATION------------------------------------------------

            total_battery_cost = st.session_state.battery_capacity * st.session_state.battery_cost_per_kwh * system_cost_multiplier
            payback_period = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
            payback_period_target = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
            roi = (annual_savings_actual * battery_lifetime - total_battery_cost) / total_battery_cost * 100




            b_without_pv, b_with_pv_simple, b_with_pv_ebo = st.tabs(["Ohne PV", "Mit PV (PS)", "Mit PV & Eigenbedarf"])

            with b_without_pv:
                if (st.session_state.battery_capacity == 0 and st.session_state.power_rating == 0):
                    with st.container(border=True):
                        st.subheader("Bitte Batteriegröße auswählen")
                else:
                    if (peak_reduction_bt_ps) == 0 :
                        st.warning("⚠️ The selected battery capacity is too low for succesful peakshaving with your load profile. \n "
                                   "Try using a battery with a higher capacity or adjusting the peakshaving threshold to a lower value.⚠️")
                    elif (peak_reduction_bt_ps) < 0.98*value_peak_reduction:
                        st.warning(f"Amount shaved: {peak_reduction_bt_ps:,.0f}kW"
                                    "⚠️ The selected battery specifications are not sufficient for your load profile. \n "
                                   "Try using a battery with a higher capacity or adjusting the peakshaving threshold to a lower value.⚠️")


                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>🔺 Alte Spitzenlast</div><div class='metric-value'>" + f"{peak_before:,.1f} kW" + "</div></div>",
                            unsafe_allow_html=True)

                    with col2:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>🔻 Neue Spitzenlast</div><div class='metric-value'>" + f"{peak_after_bt_ps:,.1f} kW" + "</div></div>",
                            unsafe_allow_html=True)
                       # st.markdown(
                       #     "<div class='metric'><div class='metric-title'>🔻 Neue Spitzenlast</div><div class='metric-value'>"
                       #     + f"{df_bt_org["grid_load"].max():,.1f} kW" + "</div></div>",
                       #     unsafe_allow_html=True)

                    with col3:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>📉 Reduktion</div><div class='metric-value'>" +
                            f"{(peak_reduction_bt_ps / peak_before * 100):.1f}%" + "</div></div>",
                            unsafe_allow_html=True)
                    with col4:
                        if ((df["load"].sum() / 4 / peak_after_bt_ps) > 2500):
                            demand_charge = demand_charge_high
                        else:
                            demand_charge = demand_charge_low

                        st.markdown(
                            "<div class='metric'><div class='metric-title'>💸 Ersparnis (p.a.)</div><div class='metric-value'>" + f"€{demand_charge * (peak_before - df_bt_ps["grid_load"].max()):,.1f}" + "</div></div>",
                            unsafe_allow_html=True,
                            help="Kalkuliert durch Spitzenlastkappung mit der Annahme einer Reduktion von " + f"{((df_bt_ps["load"].max() - df_bt_ps["grid_load"].max())):,.1f} kW" + " und einem Leistungspreis von  " + f"€/kW{demand_charge:,.1f}" + " pro kW pro Jahr.")

                    st.write("---")

                    # ---------------------------------------    BATTERIE CHART OHNE PV           ---------------------------------------
                    fig_batt_exp = px.line(df_bt_ps.reset_index(), x="timestamp",
                                           y=["load", "grid_load", "battery_discharge", "battery_soc"],
                                           title="⚡ Load, Optimized Load, and Battery Discharge with selected Battery",
                                           labels={"value": "Power (kW)", "timestamp": "Time",
                                                   "battery_discharge": "Battery discharge",
                                                   "soc_state": "Battery state of charge", "variable": "Legend"},
                                           line_shape='spline',
                                           color_discrete_map={"load": "#eb1b17", "grid_load": "#11a64c",
                                                               "battery_discharge": "#030ca8", "battery_soc": "#e64e02"})
                    fig_batt_exp.add_hline(y=df_bt_ps["grid_load"].max(),
                                           line_dash="dot", line_color="orange",
                                           annotation_text=f"Achieved peak load ({df_bt_ps["grid_load"].max():.0f}kW)",
                                           annotation_position="bottom right",
                                           name="Achieved peak load",  # Add this
                                           showlegend=True  # And this
                                           )

                    fig_batt_exp.add_hline(y=calculated_peakshaving_threshold*0.01* peak_before,
                                           line_dash="dot",
                                           line_color="blue",
                                           annotation_text=f"Target peak load ({calculated_peakshaving_threshold*0.01* peak_before:.1f}kW)",
                                           annotation_position="bottom left",
                                           name="Target peak load",
                                           showlegend=True
                                           )

                    fig_batt_exp.add_hline(y=df_bt_ps["load"].max(),
                                           line_dash="dot", line_color="red",
                                           annotation_text="Original overall peak",
                                           annotation_position="top right",
                                           name="Original overall peak",
                                           showlegend=True
                                           )

                    fig_batt_exp.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_batt_exp, use_container_width=True)

                    ####################################asddddddasssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

                    with st.container(border=True):
                        st.subheader("📈 Lastgang Analyse")
                        col0, col1, col2, col3, col4 = st.columns(5)
                        with col0:
                            st.subheader("Verbrauch")
                            st.metric("⚡ Gesamtverbrauch", f"{(df_bt_ps["load"].sum() / 1000 / 4):,.0f} MWh")

                        with col1:
                            st.subheader("Spitzenlast")
                            st.metric("🔺 Spitzenlast ohne Batterie", f"{df["load"].max():,.0f} kW")
                            st.metric("🔸 Spitzenlast mit Batterie", f"{df_bt_ps["grid_load"].max():,.0f} kW",
                                      f"{df_bt_ps["grid_load"].max() - df["load"].max():,.0f} kW", delta_color="inverse")
                        with col2:
                            col2.subheader("Volllaststunden")
                            st.metric("Ohne Batterie", f"{df_org["load"].sum()/4/peak_before:,.0f} h")
                            st.metric("Mit Batterie", f"{df_org["load"].sum() / 4 / peak_after_bt_ps:,.0f} h")
                        with col3:
                            st.subheader("Finanzen")
                            st.metric(f"💰 Jährl. Einsparungen (Peak Shaving)",
                                      f"~€ {annual_savings_actual:,.0f}", help = f"(Leistungspreis: {demand_charge}€/kW)")
                            st.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")
                        with col4:
                            st.subheader("ROI")
                            st.metric("✅ Payback Zeitraum", f"{payback_period:.1f} years")
                            st.metric(f"📈 ROI über {battery_lifetime} Jahre (Batterie Lebenszeit)", f"{roi:.1f}%")


                    with st.container(border=True):
                        st.subheader("📈 Financial Analyse")
                        col1, col2, col3, col4 = st.columns(4)

                        col1.subheader("Batteriekosten")
                        col1.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")

                        col2.subheader("Ersparnis")
                        col2.metric(f"💰 Geschätzte jährl. Einsparungen durch Peak Shaving",
                                      f"~€ {annual_savings_actual:,.1f}", help = f"(Leistungspreis: {demand_charge}€/kW)")

                        col3.subheader("ROI")
                        col3.metric("✅ Payback Zeitraum", f"{payback_period:.1f} years")

                        col4.subheader("")
                        col4.metric(f"📈 ROI über {battery_lifetime} Jahre (Batterie Lebenszeit)", f"{roi:.1f}%")

                    with st.container(border=True):
                        st.subheader("🔋 Batterie Analyse")
                        col_b_1, col_b_2, col_b_3 = st.columns(3)

                        col_b_1.subheader("🔋 Geladene Energie")
                        charge_energy = df_bt_ps["battery_charge"].sum()
                        col_b_1.metric("Aus Netz", f"{charge_energy:,.1f}")

                        col_b_2.subheader("🪫 Entladene Energie")
                        discharge_energy = df_bt_ps["battery_discharge"].sum()
                        col_b_2.metric("Für Peakshaving", f"{discharge_energy:,.1f}")

            with b_with_pv_simple:

                # ---------------------------------------    BATTERIE CHART MIT PV           ---------------------------------------
                # Peakshaving algorithm that considers only the positive net load (original load - pv impact)
                # No consumption optimization, savings only from reducing the peak

                df_exp_pv = battery_simulation_ps_with_pv(df_exp_pv, df_exp_pv["load_pv_pos"], st.session_state.battery_capacity, st.session_state.power_rating, 90, calculated_peakshaving_threshold)


                volllaststunden_bt_pv = df_exp_pv["grid_load_pv_bt"].clip(lower=0).sum() / 4 / df_exp_pv["grid_load_pv_bt"].max()
                col_0, col_1, col_2, col_3, col_4  = st.columns(5)

                with col_0:
                    with st.container(border=True):
                        st.subheader("⚡ Netzbezug")
                        st.metric("Netzbezug mit PV", f"{energy_consumed_pv:,.0f} MWh")

                with col_1:
                    with st.container(border=True):
                        st.subheader("🔺 Spitzenlast")
                        st.metric("Alt", f"{peak_load_pv:,.0f} kW")
                        st.metric("Neu", f"{df_exp_pv["grid_load_pv_bt"].max():,.0f} kW")

                with col_2:
                    with st.container(border=True):
                        st.subheader("⏺️ Volllaststunden")
                        st.metric("Alt", f"{vollaststunden_pv:,.0f} VLH")
                        st.metric("Neu", f"{volllaststunden_bt_pv:,.0f} VLH")

                with col_3:
                    with st.container(border=True):
                        st.subheader("💵 Leistungspreis")
                        vollstring = ">= 2500 VLH" if volllaststunden_bt_ps > 2500 else "< 2500 VLH"
                        demand_charge = demand_charge_high if volllaststunden_bt_pv > 2500 else demand_charge_low
                        st.metric(f"Da Volllaststunden {vollstring}", f"{demand_charge:,.0f} €/kW")

                with col_4:
                    with st.container(border=True):
                        st.subheader("💰 Ersparnis durch Peakshaving")
                        savings_ps = (peak_load_pv - df_exp_pv["grid_load_pv_bt"].max()) * demand_charge
                        st.metric(f" Durch Reduktion von {(peak_load_pv - df_exp_pv["grid_load_pv_bt"].max()):,.1f}kW", f"{savings_ps:,.0f} €/kW")


#################################################################################################################################################################
                fig_batt_exp_pv = px.line(df_exp_pv.reset_index(), x="timestamp",
                                       y=["load_pv_pos", "grid_load_pv_bt", "battery_discharge", "battery_soc"],
                                       title="⚡ Load, Optimized Load, and Battery Discharge with selected Battery - with PV ☀️",
                                       labels={"value": "Power (kW)", "timestamp": "Time",
                                               "battery_discharge": "Battery discharge",
                                               "soc_state": "Battery state of charge", "variable": "Legend"},
                                       line_shape='spline',
                                       color_discrete_map={"load": "#eb1b17", "grid_load_pv_bt": "#11a64c",
                                                           "battery_discharge": "#030ca8", "battery_soc": "#e64e02"})

                fig_batt_exp_pv.add_hline(y=df_exp_pv["grid_load_pv_bt"].max(),
                                          line_dash="dot", line_color="orange",
                                          annotation_text=f"Achieved peak load ({df_exp_pv["grid_load_pv_bt"].max():.0f}kW)",
                                          annotation_position="bottom right",
                                          name="Achieved peak load",  # Add this
                                          showlegend=True  # And this
                                          )

                fig_batt_exp_pv.add_hline(y=calculated_peakshaving_threshold*0.01* peak_before,
                                       line_dash="dot",
                                       line_color="blue",
                                       annotation_text=f"Target peak load ({calculated_peakshaving_threshold*0.01* peak_before:.1f}kW)",
                                       annotation_position="bottom left",
                                       name="Target peak load",
                                       showlegend=True
                                       )

                fig_batt_exp_pv.add_hline(y=df_exp_pv["load"].max(),
                                       line_dash="dot", line_color="red",
                                       annotation_text="Original overall peak",
                                       annotation_position="top right",
                                       name="Ursprüngliche Spitzenlast",
                                       showlegend=True
                                       )

                fig_batt_exp_pv.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_batt_exp_pv, use_container_width=True)

                with st.container(border=True):
                    st.subheader("📈 Lastgang Analyse")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Energieverbrauch")
                        st.metric("Netzbezug ohne PV", f"{total_energy_kwh:,.0f} kWh")
                        st.metric("Netzbezug mit PV", f"{df_exp_pv["load_pv_pos"].sum()/4:,.0f} kWh")
                        # // Change this to "drawn from battery instead
                        st.metric("Netzbezug mit PV & Batterie", f"{df_exp_pv["grid_load_pv_bt"].sum()/4:,.0f} kWh")

                    with col2:
                        st.subheader("Spitzenlast")
                        st.metric("🔺 Spitzenlast ohne Batterie", f"{peak_load_pv:,.0f} kW")
                        st.metric("🔸 Spitzenlast mit Batterie", f"{df_exp_pv["grid_load_pv_bt"].max():,.0f} kW",
                                  f"{df_exp_pv["grid_load_pv_bt"].max() - peak_load_pv:,.0f} kW",
                                  delta_color="inverse")
                    with col3:
                        col3.subheader("Volllaststunden")
                        st.metric("Ohne Batterie & ohne PV", f"{df_org["load"].sum() / 4 / peak_before:,.0f} h")
                        st.metric("Mit PV",f"{df_exp_pv["load"].sum() / 4 / df_exp_pv["load"].max():,.0f} h" )
                        st.metric("Mit PV & Batterie", f"{df_exp_pv["grid_load_pv_bt"].sum() / 4 / df_exp_pv["grid_load_pv_bt"].max():,.0f} h")

                with st.container(border=True):
                    st.subheader("📈 Financial Analyse")
                    col1, col2, col3, col4 = st.columns(4)

                    col1.subheader("Batteriekosten")
                    col1.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")

                    col2.subheader("Ersparnis")
                    col2.metric(f"💰 Geschätzte jährl. Einsparungen durch Peak Shaving",
                                f"~€ {annual_savings_actual:,.1f}", help=f"(Leistungspreis: {demand_charge}€/kW)")

                    col3.subheader("ROI")
                    col3.metric("✅ Payback Zeitraum", f"{payback_period:.1f} years")

                    col4.subheader("")
                    col4.metric(f"📈 ROI über {battery_lifetime} Jahre (Batterie Lebenszeit)", f"{roi:.1f}%")

                with st.container(border=True):
                    st.subheader("🔋 Batterie Analyse")
                    col_b_1, col_b_2, col_b_3 = st.columns(3)

                    col_b_1.subheader("🔋 Geladene Energie")
                    charge_energy = df_bt_ps["battery_charge"].sum()
                    col_b_1.metric("Aus Netz", f"{charge_energy:,.1f}")

                    col_b_2.subheader("🪫 Entladene Energie")
                    discharge_energy = df_bt_ps["battery_discharge"].sum()
                    col_b_2.metric("Für Peakshaving", f"{discharge_energy:,.1f}")

#################################################################################################################################################################
#################################################################################################################################################################

            with b_with_pv_ebo:

               # df_exp_2 = battery_simulation_vpv(df_with_pv, st.session_state.battery_capacity,
               #                                   st.session_state.power_rating, 90,
               #                                   calculated_peakshaving_threshold)

                calculated_peakshaving_threshold_pv = (peak_load_pv - value_peak_reduction) / peak_load_pv * 100

                df_exp_2 = battery_simulation_vpv_selfconsumption_working(df_with_pv, st.session_state.battery_capacity,
                                                                          st.session_state.power_rating, 90,
                                                                          calculated_peakshaving_threshold_pv)

               ########################        VARIABLEN FÜR BATTERIE SIMILATION           ###########################

                volllaststunden_bt_pv_2 = df_exp_2["grid_load_pv_bt"].clip(lower=0).sum() / 4 / df_exp_2["grid_load_pv_bt"].max()

                demand_charge = demand_charge_high if volllaststunden_bt_pv_2 > 2500 else demand_charge_low
                energy_charge = energy_charge_low if volllaststunden_bt_pv_2 > 2500 else energy_charge_high

                energy_exported_mwh = -df_exp_2["grid_load_pv_bt"].clip(upper=0).sum() / 4 / 1000
                pv_selfcons_kwh = min(
                    df_exp_2['battery_charge_pv_selfcons'].sum() / 4,
                    df_exp_2['battery_discharge_selfcons'].sum() / 4
                )
                annual_savings_selfcons = pv_selfcons_kwh * energy_charge
                annual_savings_peakshaving = (peak_load_pv - df_exp_2["grid_load_pv_bt"].max()) * demand_charge



                annual_savings_total = annual_savings_peakshaving + annual_savings_selfcons
                payback_period = total_battery_cost / annual_savings_total
                roi = (annual_savings_total * battery_lifetime - total_battery_cost) / total_battery_cost * 100

                max_load_jump = df["load"].diff().clip(lower=0).max()

               ##############         KPI Übersicht über Graph           ####################################################
                col_0, col_1, col_2, col_3, col_4 = st.columns([3,3,4,4,3])
                with col_0:
                    with st.container(border=True):
                        st.subheader("⚡ Netzbezug")
                        #st.metric("Netzbezug mit PV", f"{energy_consumed_pv:,.0f} MWh")
                        st.metric("Netzbezug mit PV & Batterie",
                                  f"{df_exp_2["grid_load_pv_bt"].clip(lower=0).sum() / 4 / 1000:,.0f} MWh",
                                  f"{df_exp_2["grid_load_pv_bt"].clip(lower=0).sum() / 4 / 1000 - energy_consumed_pv:,.0f} MWh",delta_color="inverse")
                        st.write(f"(Netzbezug ohne Batterie: **{energy_consumed_pv:,.0f}** MWh)\n\n")

                with col_1:
                    with st.container(border=True):
                        st.subheader("🔺 Spitzenlast")
                        #st.metric("Spitzenlast mit PV", f"{peak_load_pv:,.0f} kW")
                        st.metric("Erreichte Spitzenlast mit PV & Batterie", f"{df_exp_2["grid_load_pv_bt"].max():,.0f} kW",
                                  f"{df_exp_2["grid_load_pv_bt"].max()-peak_load_pv:,.0f} kW",delta_color="inverse")
                        st.write(f"(Spitzenlast ohne Batterie: **{peak_load_pv:,.0f}** kW)\n\n")

                with col_2:
                    with st.container(border=True):
                        vollstring2 = ">= 2500 h" if volllaststunden_bt_pv_2 > 2500 else "< 2500 h"
                        demand_charge2 = demand_charge_high if volllaststunden_bt_pv_2 > 2500 else demand_charge_low

                        st.subheader("⏺️ Volllaststunden")
                        #st.metric("Mit PV", f"{vollaststunden_pv:,.0f} h")
                        st.metric("Mit PV & Batterie", f"{volllaststunden_bt_pv_2:,.0f} h", f"{(volllaststunden_bt_pv_2-vollaststunden_pv):,.0f} h", delta_color="off", help=f"Resuliert in Leistungspreis von {demand_charge2:,.0f} €/kW ")
                        st.write(f"(VLH ohne Batterie: **{vollaststunden_pv:,.0f}** h)\n\n")

                with col_3:
                    with st.container(border=True):
                        st.subheader("💰 Einsparpotential")
                        savings_ps = (peak_load_pv - df_exp_2["grid_load_pv_bt"].max()) * demand_charge2
                        st.metric("Järhliche Einsparungen", f"~{(savings_ps + annual_savings_selfcons):,.0f}€")
#                                f"~{savings_ps:,.0f} €"
#                                f" Durch Reduktion von {(peak_load_pv - df_exp_2["grid_load_pv_bt"].max()):,.0f}kW",
#                                f"~{savings_ps:,.0f} €")
                        st.write(f" **{savings_ps:,.0f}€** : Lastreduktion um {(peak_load_pv - df_exp_2["grid_load_pv_bt"].max()):,.0f}kW \n\n  **{annual_savings_selfcons:,.0f}€** : Eigenbedarfsoptimierung" )

                with col_4:
                    with st.container(border=True):
                        st.subheader("✅ ROI")
                        st.metric("Amortisationszeit", f"{payback_period:.1f} Jahre")


#############################       OPTIMIZATION ################################

#                    st.write("---")

#                    if st.button("🎯 Auto-Optimize Parameters"):
#                        with st.spinner("Optimizing battery parameters..."):
#                            optimal_params = optimize_battery_params_working(
#                                df_exp_2, st.session_state.battery_capacity, st.session_state.power_rating,
#                                demand_charge_low, demand_charge_high, energy_charge_low, energy_charge_high
#                            )

#                        st.success(f"Optimal ROI: {optimal_params['roi']:.1f} years")

#                        st.write(f"{optimal_params['threshold_pct']} and reduction of {(optimal_params['threshold_pct']/100 * peak_load_pv)} ")
#                        st.write(optimal_params['reserve_fraction'])

                    #st.write(optimal_params['base_reserve'])
                    #st.write(optimal_params['extra_reserve'])

                    # Update sliders with optimal values
                    #st.session_state.threshold_pct = optimal_params['threshold_pct']
                    #st.session_state.base_reserve = optimal_params['base_reserve']
                    #st.session_state.extra_reserve = optimal_params['extra_reserve']



                st.write("---")

                df_exp_2["battery_discharge"] = -df_exp_2["battery_discharge"]

                column_map = {
                    "load": "Netzlast (mit PV) ohne Batterie",
                    "grid_load_pv_bt": "Netzlast mit Batterie",
                    "load_pv_neg": "PV-Überschuss",
                    "battery_discharge": "Batterie-Entladung",
                   # "battery_soc": "Batterie Ladezustand (SoC)",
                    "battery_charge": "Batterie-Ladung (gesamt)",
                   # "battery_charge_pv": "Batterie-Ladung (PV)",
                   # "battery_charge_grid": "Batterie-Ladung (Netz)"
                }

                df_plot = df_exp_2.rename(columns=column_map)

                fig_batt_exp_pv_2 = px.line(df_plot.reset_index(), x="timestamp",
                                            y=list(column_map.values()),
                                            title="⚡ Netzlast, Batterieeinsatz und Spitzenlasten mit PV & Batterie ☀️",
                                            labels={"timestamp": "Zeit", "value": "Leistung (kW)",
                                                    "variable": "Legende"},
                                            line_shape='spline',
                                            color_discrete_map={"Netzlast (mit PV) ohne Batterie": "crimson",
                                                                "Netzlast mit Batterie": "#43A047",
                                                                "Batterie-Ladung (gesamt)":"#64B5F6",
                                                                "Batterie-Entladung": "#1976D2",
                                                               # "Batterie Ladezustand (SoC)": "#e64e02",
                                                                "PV-Überschuss": "gold"}
                                            )

                fig_batt_exp_pv_2.add_hline(y=df_exp_2["load_org"].max(),
                                        line_dash="dot", line_color="DarkSlateGray",
                                        annotation_text="Spitzenlast ohne PV",
                                        annotation_position="top right",
                                        name="Spitzenlast ohne PV",
                                        showlegend=True
                                        )

                fig_batt_exp_pv_2.add_hline(y=df_exp_2["load"].max(),
                                       line_dash="dot", line_color="red",
                                       annotation_text="Spitzenlast mit PV",
                                       annotation_position="top right",
                                       name="Ursprüngliche Spitzenlast",
                                       showlegend=True
                                       )

                fig_batt_exp_pv_2.add_hline(y=calculated_peakshaving_threshold_pv*0.01* peak_load_pv,
                                       line_dash="dot",
                                       line_color="MediumVioletRed",
                                       annotation_text=f"Ziel-Spitzenlast ({calculated_peakshaving_threshold_pv*0.01* peak_load_pv:.0f}kW)",
                                       annotation_position="bottom left",
                                       name="Ziel Spitzenlast",
                                       showlegend=True
                                       )

                fig_batt_exp_pv_2.add_hline(y=df_exp_2["grid_load_pv_bt"].max(),
                                          line_dash="dot", line_color="chartreuse",
                                          annotation_text=f"Erreichte Spitzenlast ({df_exp_2["grid_load_pv_bt"].max():.0f}kW)",
                                          annotation_position="bottom right",
                                          name="Erreichte Spitzenlast",  # Add this
                                          showlegend=True  # And this
                                          )


                fig_batt_exp_pv_2.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_batt_exp_pv_2, use_container_width=True)

                with st.expander("Detail-Ansicht"):
                    fig_detailed_pv_bt = go.Figure()

                    # Netzlast ohne PV und Batterie
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load_org'],
                        mode='lines', name=' Netzlast ohne PV und Batterie',
                        line=dict(color='darkgray', width=1),
                        hovertemplate=' Netzlast ohne PV und Batterie: %{y:.1f} kW'
                    ))

                    # Netzlast mit PV ohne Batterie (dunkelrot)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load'].clip(lower=0),
                        mode='lines', name='Netzlast mit PV',
                        line=dict(color='darkred', width=1),
                        hovertemplate='Netzlast (mit PV): %{y:.1f} kW'
                    ))

                    # Netzeinspeisung mit PV ohne Batterie (rot)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load'].clip(upper=0),
                        mode='lines', name='Netzeinspeisung mit PV',
                        line=dict(color='crimson', width=1),
                        hovertemplate='Netzeinspeisung (mit PV): %{y:.1f} kW'
                    ))

                    # Netzlast mit Batterie (grün)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['grid_load_pv_bt'].clip(lower=0),
                        mode='lines', name='Netzlast (mit PV & Batterie)',
                        line=dict(color='darkgreen', width=2),
                        hovertemplate='Netzlast mit Batterie: %{y:.1f} kW'
                    ))
                    # Netzeinspeisung mit Batterie (grün)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['grid_load_pv_bt'].clip(upper=0),
                        mode='lines', name='Netzeinspeisung (mit PV & Batterie)',
                        line=dict(color='springgreen', width=2),
                        hovertemplate='Netzeinspeisung mit Batterie: %{y:.1f} kW'
                    ))

                    # Batterie-Ladung mit PV (gelb)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_charge_pv'],
                        mode='lines', name='Batterie-Ladung aus PV',
                        line=dict(color='mediumblue', width=1),
                        hovertemplate='Batterie-Ladung aus PV: %{y:.1f} kW'
                    ))

                    # Batterie-Ladung von Grid )
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_charge_grid'],
                        mode='lines', name='Batterie-Ladung aus Netz',
                        line=dict(color='purple', width=1),
                        hovertemplate='Batterie-Ladung aus Netz: %{y:.1f} kW'
                    ))

                    # Batterie-Discharge
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_discharge'],
                        mode='lines', name='Batterie-Entladung',
                        line=dict(color='midnightblue', width=1),
                        hovertemplate='Batterie-Entladung: %{y:.1f} kW'
                    ))


                    # Batterie-Ladung (orange, gestrichelt, sekundäre Y-Achse)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_soc'],
                        mode='lines', name='Batterie-Ladung (SoC)',
                        line=dict(color='#ff780a', width=2),
                        yaxis='y2',
                        hovertemplate='SoC: %{y:.1f} kWh'
                    ))


                    # Horizontale Linien für Spitzenlasten
                    fig_detailed_pv_bt.add_hline(
                        y=df_exp_2["grid_load_pv_bt"].max(), line_dash="dot", line_color="green",
                        annotation_text=f"Erreichte Spitzenlast ({df_exp_2['grid_load_pv_bt'].max():.0f} kW)",
                        annotation_position="bottom right"
                    )

                    if (df_exp_2["grid_load_pv_bt"].max() != (calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv)):
                        fig_detailed_pv_bt.add_hline(
                            y=calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv, line_dash="dot", line_color="blue",
                            annotation_text=f"Ziel-Spitzenlast ({calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv:.0f} kW)",
                            annotation_position="bottom left"
                        )

                    fig_detailed_pv_bt.add_hline(
                        y=df_exp_2["load"].max(), line_dash="dot", line_color="red",
                        annotation_text=f"Spitzenlast ohne Batterie ({df_exp_2['load'].max():.0f} kW)",
                        annotation_position="top right"
                    )

                    # Layout-Optimierung
                    fig_detailed_pv_bt.update_layout(
                        title="⚡☀️🔋🔎 Detail-Ansicht: Netzlast, Batterieeinsatz und Spitzenlasten mit PV & Batterie",
                        xaxis_title="Zeit",
                        yaxis_title="Leistung (kW)",
                        yaxis2=dict(
                            title="Batterie-Ladung (kWh)",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        legend_title="Legende",
                        height=550,
                        margin=dict(l=20, r=20, t=60, b=20),
                        font=dict(size=14)
                    )

                    st.plotly_chart(fig_detailed_pv_bt, use_container_width=True)





                with st.expander("Netzbezug & Spitzenlasten"):
                    fig_load = go.Figure()
                    fig_load.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["load_pv_pos"],
                        mode='lines', name="Netzlast mit PV", line=dict(color="#5bc0eb")
                    ))
                    fig_load.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["grid_load_pv_bt"].clip(lower=0),
                        mode='lines', name="Netzlast mit PV & Batterie", line=dict(color="#11a64c")
                    ))

                    fig_load.add_hline(y=df_exp_2["grid_load_pv_bt"].max(),
                                       line_dash="dot", line_color="orange",
                                       annotation_text=f"Erreichte Spitzenlast ({df_exp_2['grid_load_pv_bt'].max():.0f}kW)",
                                       annotation_position="bottom right")
                    fig_load.add_hline(y=calculated_peakshaving_threshold_pv * 0.01 * peak_before,
                                       line_dash="dot", line_color="blue",
                                       annotation_text=f"Ziel-Spitzenlast ({calculated_peakshaving_threshold_pv * 0.01 * peak_before:.1f}kW)",
                                       annotation_position="bottom left")
                    fig_load.add_hline(y=df_exp_2["load"].max(),
                                       line_dash="dot", line_color="red",
                                       annotation_text="Ursprüngliche Spitzenlast",
                                       annotation_position="top right")

                    fig_load.update_layout(
                        title="Netzbezug und Spitzenlasten",
                        xaxis_title="Zeit", yaxis_title="Leistung (kW)",
                        legend_title="Legende", height=400, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_load, use_container_width=True)

                with st.expander("Monatliche Batterie-Nutzung"):
                    monthly_summary = df_exp_2.groupby('month').agg({
                        'battery_charge_pv': 'sum',
                        'battery_charge_grid': 'sum',
                        'battery_discharge_peakshave': 'sum',
                        'battery_discharge_selfcons': 'sum'
                    }) / 4

                    fig_month = go.Figure()
                    fig_month.add_trace(go.Bar(name='Laden aus PV', x=monthly_summary.index,
                                               y=monthly_summary['battery_charge_pv']))
                    fig_month.add_trace(go.Bar(name='Laden aus Netz', x=monthly_summary.index,
                                               y=monthly_summary['battery_charge_grid']))
                    fig_month.add_trace(go.Bar(name='Entladen für Peak Shaving', x=monthly_summary.index,
                                               y=monthly_summary['battery_discharge_peakshave']))
                    fig_month.add_trace(go.Bar(name='Entladen für Eigenverbrauch', x=monthly_summary.index,
                                               y=monthly_summary['battery_discharge_selfcons']))

                    fig_month.update_layout(
                        barmode='group',
                        title="Monatliche Batterie-Nutzung nach Quelle und Zweck",
                        xaxis_title="Monat", yaxis_title="Energie (kWh)",
                        height=400, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_month, use_container_width=True)

                with st.expander("Batterie Ladezyklen"):
                    import plotly.subplots as sp
                    fig_batt = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                subplot_titles=("Batterie Laden", "Batterie Entladen"))

                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_charge_pv"],
                        mode='lines', name="Laden aus PV", line=dict(color="#f6d55c")
                    ), row=1, col=1)
                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_charge_grid"],
                        mode='lines', name="Laden aus Netz", line=dict(color="#3caea3")
                    ), row=1, col=1)

                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_discharge_peakshave"],
                        mode='lines', name="Entladen für Peak Shaving", line=dict(color="#ed553b")
                    ), row=2, col=1)
                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_discharge_selfcons"],
                        mode='lines', name="Entladen für Eigenverbrauch", line=dict(color="#20639b")
                    ), row=2, col=1)

                    fig_batt.update_layout(
                        title="Batterie Lade- und Entladeleistung",
                        height=600, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_batt, use_container_width=True)
########################################################################################################################################################################################################
                with st.container(border=True):
                    st.header("📈 Finanzanalyse")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.subheader("Batteriekosten")
                    col1.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")

                    col2.subheader("Ersparnis")
                    col2.metric(
                        "💰 Peak Shaving",
                        f"~€ {annual_savings_peakshaving:,.0f}",
                        help=f"Einsparung durch Reduktion der Spitzenlast ({demand_charge} €/kW)"
                    )
                    col2.metric(
                        "💡 Eigenverbrauch",
                        f"~€ {annual_savings_selfcons:,.0f}",
                        help=f"Einsparung durch PV-Eigenverbrauch via Batterie ({energy_charge_high} €/kWh)"
                    )
                    col2.metric(
                        "💰 Gesamteinsparung",
                        f"~€ {annual_savings_total:,.0f}",
                        help="Summe aus Peak Shaving und Eigenverbrauch"
                    )

                    col3.subheader("ROI & Payback")
                    col3.metric("✅ Payback Zeitraum", f"{payback_period:.1f} years")
                    col3.metric("📈 ROI über Lebenszeit", f"{roi:.1f} %")

                    #                        col4.subheader("Parameter")
                    with col4.expander("Parameter", expanded=False):
                        st.metric("Strompreis", f"{energy_charge_high:.2f} €/kWh")
                        st.metric("Leistungspreis", f"{demand_charge:.2f} €/kW")
                        st.metric("Batterielebensdauer", f"{battery_lifetime} Jahre")

                with st.container(border=True):
                    st.header("📈 Lastgang Details")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.subheader("Energieverbrauch")
                        st.metric("Netzbezug ohne PV", f"{total_energy_kwh/1000:,.0f} MWh")
                        st.metric("Netzbezug mit PV", f"{df_exp_2["load_pv_pos"].sum()/4/1000:,.0f} MWh")
                        # // Change this to "drawn from battery instead
                        st.metric("Netzbezug mit PV & Batterie", f"{df_exp_2["grid_load_pv_bt"].clip(lower=0).sum()/4/1000:,.0f} MWh")

                    with col2:
                        st.subheader("Spitzenlast")
                        st.metric("🔺🔺Spitzenlast ohne PV", f"{peak_load:,.0f} kW")
                        st.metric("🔺 Spitzenlast mit PV", f"{peak_load_pv:,.0f} kW", f"{peak_load_pv - peak_load:,.0f} kW",
                                  delta_color="inverse")
                        st.metric("🔸 Spitzenlast mit PV & Batterie", f"{df_exp_2["grid_load_pv_bt"].max():,.0f} kW",
                                  f"{df_exp_2["grid_load_pv_bt"].max() - peak_load:,.0f} kW",
                                  delta_color="inverse")

                    with col3:
                        st.subheader("Eingespeiste Energie")
                        st.metric("Einspeisung ins Netz ohne Batterie", f"{-df_exp_2["load_pv_neg"].sum()/4/1000:,.0f} MWh",
                                 help="Summe aller Zeitpunkte mit negativer Netzlast mit PV (PV-Überschuss)")
                        st.metric("Einspeisung ins Netz nach Optimierung", f"{energy_exported_mwh:,.0f} MWh",
                                 help="Summe aller Zeitpunkte mit negativer Netzlast nach Batterie-Optimierung (PV-Überschuss nach Eigenverbrauch und Batterieladung)")
                    with col4:
                        st.subheader("Volllaststunden")
                        st.metric("Ohne Batterie & ohne PV", f"{df_org["load"].sum() / 4 / peak_before:,.0f} h")
                        st.metric("Mit PV",f"{df_exp_pv["load"].sum() / 4 / df_exp_pv["load"].max():,.0f} h" )
                        st.metric("Mit PV & Batterie", f"{df_exp_2["grid_load_pv_bt"].clip(lower=0).sum() / 4 / df_exp_2["grid_load_pv_bt"].max():,.0f} h")


                with st.container(border=True):
                    st.header("🔋 Batterie Analyse")
                    col_b_1, col_b_2, col_b_3, col_b_4 = st.columns(4)

                with col_b_1:
                    st.subheader("🔋 Geladene Energie")
                    st.metric("Gesamt", f"{df_exp_2['battery_charge'].sum() / 4 /1000:,.0f} MWh")
                    st.metric("Aus Netz", f"{df_exp_2['battery_charge_grid'].sum() / 4 /1000:,.0f} MWh")
                    st.metric("Aus PV", f"{df_exp_2['battery_charge_pv'].sum() / 4 /1000:,.0f} MWh")


                with col_b_2:
                    st.subheader("")
                    with st.expander("Details"):
                        st.metric("Aus PV (Selbstverbrauch)",
                                  f"{df_exp_2['battery_charge_pv_selfcons'].sum() / 4 /1000:,.0f} MWh")
                        st.metric("Aus PV (Peakshaving)",
                                  f"{df_exp_2['battery_charge_pv_peakshave'].sum() / 4 /1000:,.0f} MWh")
                        st.metric("Aus Netz (Selbstverbrauch)",
                                  f"{df_exp_2['battery_charge_grid_selfcons'].sum() / 4 /1000:,.0f} MWh")
                        st.metric("Aus Netz (Peakshaving)",
                                  f"{df_exp_2['battery_charge_grid_peakshave'].sum() / 4 /1000:,.0f} MWh")
                with col_b_3:
                    st.subheader("🪫 Entladene Energie")
                    st.metric("Gesamt", f"{df_exp_2['battery_discharge'].sum() / 4 /1000:,.0f} MWh")
                with col_b_4:
                    st.subheader("")
                    with st.expander("Details"):
                        st.metric("Für Peakshaving", f"{df_exp_2['battery_discharge_peakshave'].sum() / 4 /1000:,.0f} kWh")
                        st.metric("Für Eigenverbrauch", f"{df_exp_2['battery_discharge_selfcons'].sum() / 4 /1000:,.0f} kWh")



