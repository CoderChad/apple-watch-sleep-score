"""
Sleep Quality Visualization Module

This module provides visualization functions for the sleep quality assessment system,
including graphic displays of individual metrics, overall score, and trend analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List
import io
import base64

class SleepQualityVisualizer:
    """
    Class for generating visualizations of sleep quality data, with the ability
    to display the numerical sleep quality score alongside detailed diagrams.
    """

    def __init__(self, theme: str = 'dark'):
        """
        Initialize the visualizer with a specified theme.
        Args:
            theme: Visual theme ('dark' or 'light')
        """
        self.theme = theme
        self._set_style()

    def _set_style(self) -> None:
        """Set the visual style based on the theme."""
        if self.theme == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#4287f5',
                'secondary': '#42c6ff',
                'accent': '#f542a7',
                'deep': '#5d36f4',
                'rem': '#3fa6ff',
                'light': '#a1c7ff',
                'awake': '#f55142',
                'text': '#ffffff',
                'background': '#1a1a1a'
            }
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = {
                'primary': '#2c6bf2',
                'secondary': '#0a9df5',
                'accent': '#f542a7',
                'deep': '#3624b3',
                'rem': '#3fa6ff',
                'light': '#a1c7ff',
                'awake': '#f55142',
                'text': '#333333',
                'background': '#ffffff'
            }

    def _figure_to_base64(self) -> str:
        """
        Convert the current matplotlib figure to a base64-encoded PNG and close it.
        Returns:
            Base64 string
        """
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
        plt.close()
        buf.seek(0)
        img_bytes = buf.read()
        return base64.b64encode(img_bytes).decode('utf-8')

    def generate_score_card(self, score: int) -> str:
        """
        Generate a simple card showing the sleep quality score 0-100.
        Args:
            score: Sleep quality score
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, f"{score}", fontsize=72, ha='center', va='center', color=self.colors['primary'])
        plt.text(0.5, 0.3, "Sleep Score", fontsize=14, ha='center', va='center', color=self.colors['text'])
        plt.axis('off')
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_sleep_quality_trend(self, data: List[Dict], days: int = 14) -> str:
        """
        Generate a trend chart showing sleep quality scores over time,
        with the latest numeric score annotated.
        Args:
            data: List of sleep quality assessment results
            days: Number of days to display
        Returns:
            Base64 encoded PNG image
        """
        if not data:
            return ""
        df = pd.DataFrame([
            {'date': datetime.datetime.fromisoformat(item['date']),
             'score': item['sleep_quality_score']}
            for item in data
        ])
        df = df.sort_values('date')
        if len(df) > days:
            df = df.iloc[-days:]

        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['score'], 'o-', color=self.colors['primary'], linewidth=2)
        # trend line
        z = np.polyfit(range(len(df)), df['score'], 1)
        p = np.poly1d(z)
        plt.plot(df['date'], p(range(len(df))), '--', color=self.colors['secondary'], linewidth=1.5, alpha=0.8)

        # annotate latest score
        latest_date = df['date'].iloc[-1]
        latest_score = df['score'].iloc[-1]
        plt.scatter([latest_date], [latest_score], color=self.colors['accent'], s=100)
        plt.text(latest_date, latest_score + 2, f"{int(latest_score)}", fontsize=12,
                 ha='center', va='bottom', color=self.colors['text'])

        plt.title('Sleep Quality Score Trend', fontsize=16)
        plt.ylabel('Sleep Quality Score', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_sleep_structure_chart(self, sleep_data: Dict) -> str:
        """
        Generate a chart showing the sleep structure (stages) for a single night.
        Args:
            sleep_data: Dictionary containing processed sleep data
        Returns:
            Base64 encoded PNG image
        """
        stages = sleep_data['sleep_stages']
        total = sum(stages.values())
        plt.figure(figsize=(8, 6))
        colors = [self.colors['deep'], self.colors['rem'], self.colors['light'], self.colors['awake']]
        labels = [
            f"Deep: {stages['deep']:.0f}m ({stages['deep']/total*100:.1f}%)",
            f"REM: {stages['rem']:.0f}m ({stages['rem']/total*100:.1f}%)",
            f"Light: {stages['light']:.0f}m ({stages['light']/total*100:.1f}%)",
            f"Awake: {stages['awake']:.0f}m ({stages['awake']/total*100:.1f}%)"
        ]
        plt.pie([stages['deep'], stages['rem'], stages['light'], stages['awake']],
                labels=labels, colors=colors, autopct='', startangle=90, wedgeprops={'alpha':0.8})
        date_str = sleep_data.get('date', '')
        plt.title(f"Sleep Structure - {date_str}", fontsize=16)
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_heart_rate_chart(self, sleep_data: Dict) -> str:
        """
        Generate a chart showing heart rate trend during sleep.
        Args:
            sleep_data: Dictionary containing processed sleep data
        Returns:
            Base64 encoded PNG image
        """
        hr = sleep_data['heart_rate']
        start = datetime.datetime.fromisoformat(sleep_data['start_time'])
        end   = datetime.datetime.fromisoformat(sleep_data['end_time'])
        duration = (end - start).total_seconds() / 3600
        times = [start + datetime.timedelta(hours=duration * i / len(hr)) for i in range(len(hr))]
        plt.figure(figsize=(10,5))
        plt.plot(times, hr, '-', color=self.colors['primary'], linewidth=2, alpha=0.8)
        window = max(5, len(hr)//20)
        if len(hr) > window:
            sr = pd.Series(hr).rolling(window=window, center=True).mean()
            plt.plot(times, sr, '-', color=self.colors['accent'], linewidth=2.5)
        plt.title('Heart Rate During Sleep', fontsize=16)
        plt.ylabel('BPM', fontsize=12)
        plt.grid(alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_hrv_trend(self, sleep_data: Dict) -> str:
        """
        Generate a chart showing HRV trend during sleep.
        Args:
            sleep_data: Dictionary containing processed sleep data
        Returns:
            Base64 encoded PNG image
        """
        hrv = np.array(sleep_data['hrv'], dtype=float)
        start = datetime.datetime.fromisoformat(sleep_data['start_time'])
        end   = datetime.datetime.fromisoformat(sleep_data['end_time'])
        # compute frequency string safely
        total_secs = (end - start).total_seconds()
        freq_secs = int(total_secs / len(hrv)) if len(hrv) > 0 else 1
        times = pd.date_range(start=start, periods=len(hrv), freq=f"{freq_secs}S")
        plt.figure(figsize=(10,4))
        plt.plot(times, hrv, '-', linewidth=2, color=self.colors['deep'])
        plt.title('HRV During Sleep', fontsize=16)
        plt.ylabel('HRV Metric', fontsize=12)
        plt.grid(alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_spo2_trend(self, sleep_data: Dict) -> str:
        """
        Generate a chart showing blood oxygen (SpO2) trend during sleep.
        """
        spo2 = np.array(sleep_data['spo2'], dtype=float)
        plt.figure(figsize=(10,4))
        plt.plot(spo2, '-', linewidth=2, color=self.colors['rem'])
        plt.title('SpO2 During Sleep', fontsize=16)
        plt.ylabel('SpO2 (%)', fontsize=12)
        plt.ylim(80, 100)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return self._figure_to_base64()

    def generate_temperature_trend(self, sleep_data: Dict) -> str:
        """
        Generate a chart showing wrist temperature trend during sleep.
        """
        temp = np.array(sleep_data['wrist_temp'], dtype=float)
        plt.figure(figsize=(10,4))
        plt.plot(temp, '-', linewidth=2, color=self.colors['light'])
        plt.title('Wrist Temperature During Sleep', fontsize=16)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return self._figure_to_base64()
