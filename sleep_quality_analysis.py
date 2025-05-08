"""
Sleep Quality Analysis Module

This module provides functions to generate separate visualizations for one specific night:
  - Heart rate trend
  - HRV trend
  - SpO2 trend
  - Wrist temperature trend
  - Sleep stage pie chart
Additionally, it includes a simple 1D-CNN to predict sleep quality score and compute channel importance.
Each time-based chart now uses consistent x-axis from start_time to end_time with hourly ticks for clear comparison across charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from typing import Dict
import io
import base64
from keras import layers, models


# --- Helper to convert figures to base64 ---

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- Individual Charts for Single Night ---

def generate_hr_chart(sleep_data: Dict) -> str:
    """Return a base64 PNG of heart rate over time with hourly x-axis ticks."""
    start = datetime.datetime.fromisoformat(sleep_data['start_time'])
    end = datetime.datetime.fromisoformat(sleep_data['end_time'])
    hr = np.array(sleep_data['heart_rate'], dtype=float)
    n = len(hr)
    times = [start + (end - start) * i/(n-1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, hr, color='tab:red', linewidth=2)
    ax.set_title('Heart Rate (BPM)')
    ax.set_xlabel('Time')
    ax.set_ylabel('BPM')
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def generate_hrv_chart(sleep_data: Dict) -> str:
    """Return a base64 PNG of HRV over time with hourly x-axis ticks."""
    start = datetime.datetime.fromisoformat(sleep_data['start_time'])
    end = datetime.datetime.fromisoformat(sleep_data['end_time'])
    hrv = np.array(sleep_data['hrv'], dtype=float)
    n = len(hrv)
    times = [start + (end - start) * i/(n-1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, hrv, color='tab:blue', linewidth=2)
    ax.set_title('HRV')
    ax.set_xlabel('Time')
    ax.set_ylabel('HRV Metric')
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def generate_spo2_chart(sleep_data: Dict) -> str:
    """Return a base64 PNG of SpO2 over time with hourly x-axis ticks."""
    start = datetime.datetime.fromisoformat(sleep_data['start_time'])
    end = datetime.datetime.fromisoformat(sleep_data['end_time'])
    spo2 = np.array(sleep_data['spo2'], dtype=float)
    n = len(spo2)
    times = [start + (end - start) * i/(n-1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, spo2, color='tab:green', linewidth=2)
    ax.set_title('SpO2 (%)')
    ax.set_xlabel('Time')
    ax.set_ylabel('SpO2 (%)')
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylim(80,100)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def generate_temp_chart(sleep_data: Dict) -> str:
    """Return a base64 PNG of wrist temperature over time with hourly x-axis ticks."""
    start = datetime.datetime.fromisoformat(sleep_data['start_time'])
    end = datetime.datetime.fromisoformat(sleep_data['end_time'])
    temp = np.array(sleep_data['wrist_temp'], dtype=float)
    n = len(temp)
    times = [start + (end - start) * i/(n-1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, temp, color='tab:orange', linewidth=2)
    ax.set_title('Wrist Temperature (°C)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def generate_sleep_stage_pie_chart(sleep_data: Dict) -> str:
    """Return a base64 PNG pie chart of sleep stage durations."""
    stages = sleep_data['sleep_stages']
    labels = []
    sizes = []
    total = sum(stages.values())
    for stage in ['deep','rem','light','awake']:
        dur = stages.get(stage,0)
        labels.append(f"{stage.title()}: {dur}m ({dur/total*100:.1f}%)")
        sizes.append(dur)
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ['navy','purple','skyblue','lightgray']
    ax.pie(sizes, labels=labels, colors=colors, startangle=90)
    ax.set_title(f"Sleep Stages Breakdown - {sleep_data.get('date','')}")
    return _fig_to_base64(fig)

# --- CNN Model and Importance ---

def build_sleep_cnn(input_length: int, channels: int = 4) -> models.Model:
    """Build and compile a 1D-CNN for sleep score prediction."""
    inp = layers.Input(shape=(input_length, channels), name='sleep_input')
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation='linear', name='score_output')(x)
    model = models.Model(inputs=inp, outputs=out, name='SleepQualityCNN')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def compute_channel_importance(cnn_model: models.Model) -> Dict[str, float]:
    """Return normalized sum of abs weights of first Conv1D per channel."""
    w = cnn_model.layers[1].get_weights()[0]
    abs_sum = np.sum(np.abs(w), axis=(0,2))
    norms = abs_sum / np.sum(abs_sum)
    names = ['heart_rate','hrv','spo2','wrist_temp']
    return {names[i]: float(norms[i]) for i in range(len(names))}