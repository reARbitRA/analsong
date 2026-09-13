"""
AudioIntelligence Pro - Comprehensive Music Analysis Suite

Run: pip install -r requirements.txt && python app.py
Then open http://localhost:7860 in your browser

The application performs local, rule-based audio analysis. No audio or analysis
results are sent to an external service.
"""

from __future__ import annotations

import html
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyloudnorm as pyln
from scipy.signal import butter, find_peaks, resample_poly, sosfilt, welch


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

APP_NAME = "AudioIntelligence Pro"
APP_VERSION = "2.0"
MAX_FILE_BYTES = 50 * 1024 * 1024
LONG_TRACK_SECONDS = 10 * 60
ANALYSIS_SAMPLE_RATE = 22_050
DEFAULT_HOP_LENGTH = 512
STRUCTURE_HOP_LENGTH = 2_048
STFT_SIZE = 2_048
MAX_PLOT_POINTS = 5_000
MAX_SPECTROGRAM_FRAMES = 600
MAX_SPECTROGRAM_BINS = 256
SPECTRUM_SMOOTHING_BINS = 7
MAX_STRUCTURE_FEATURE_FRAMES = 800
MIN_SECTION_SECONDS = 8.0
MIN_PITCH_HZ = 30.0
MAX_PITCH_HZ = 300.0
PITCH_CONFIDENCE_THRESHOLD = 0.5
TRUE_PEAK_OVERSAMPLE = 4
LUFS_FLOOR = -60.0
EPSILON = 1.0e-12

SUPPORTED_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KEY_NAMES = NOTE_NAMES
TIME_SIGNATURES = ("4/4", "3/4", "6/8")
FREQUENCY_BANDS: Mapping[str, Tuple[float, float, str]] = {
    "sub_bass": (20.0, 60.0, "Sub-Bass"),
    "bass": (60.0, 250.0, "Bass"),
    "low_mids": (250.0, 500.0, "Low-Mid"),
    "mids": (500.0, 2_000.0, "Mid"),
    "high_mids": (2_000.0, 4_000.0, "Hi-Mid"),
    "presence": (4_000.0, 8_000.0, "Presence"),
    "brilliance": (8_000.0, 20_000.0, "Brilliance"),
}
PLATFORM_TARGETS: Mapping[str, float] = {
    "spotify": -14.0,
    "apple": -16.0,
    "youtube": -14.0,
    "amazon": -14.0,
    "tidal": -14.0,
    "cd_reference": -9.0,
}
ANALYSIS_STEPS: Tuple[str, ...] = (
    "Loading audio waveform...",
    "Detecting tempo & beat grid...",
    "Analyzing key & harmony...",
    "Extracting chord progressions...",
    "Computing spectral features...",
    "Measuring loudness & dynamics...",
    "Analyzing stereo field...",
    "Generating AI production report...",
)
PLOT_CONFIG: Mapping[str, Any] = {"responsive": True, "displayModeBar": True}
CUSTOM_COLORSCALE: List[List[Any]] = [
    [0.0, "#06060a"],
    [0.1, "#0c0c2e"],
    [0.25, "#1a1060"],
    [0.4, "#4a1a8a"],
    [0.55, "#7c5cfc"],
    [0.7, "#00b4d8"],
    [0.85, "#00e5a0"],
    [1.0, "#ffffff"],
]

LOGGER = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Complete self-contained design system
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
  --bg-primary: #06060a;
  --bg-secondary: #0c0c14;
  --bg-tertiary: #11111b;
  --bg-card: rgba(255, 255, 255, 0.02);
  --bg-card-hover: rgba(255, 255, 255, 0.04);
  --bg-elevated: rgba(255, 255, 255, 0.06);
  --accent-1: #7c5cfc;
  --accent-2: #00e5a0;
  --accent-3: #00b4d8;
  --accent-gradient: linear-gradient(135deg, #7c5cfc 0%, #00e5a0 50%, #00b4d8 100%);
  --accent-gradient-subtle: linear-gradient(135deg, rgba(124,92,252,0.15) 0%, rgba(0,229,160,0.08) 50%, rgba(0,180,216,0.15) 100%);
  --text-primary: #f0f0f5;
  --text-secondary: #8892b0;
  --text-tertiary: #4a5568;
  --text-accent: #7c5cfc;
  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-medium: rgba(255, 255, 255, 0.1);
  --border-accent: rgba(124, 92, 252, 0.3);
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
  --shadow-md: 0 8px 32px rgba(0,0,0,0.4);
  --shadow-lg: 0 16px 64px rgba(0,0,0,0.5);
  --shadow-glow: 0 0 40px rgba(124,92,252,0.15);
  --shadow-glow-green: 0 0 40px rgba(0,229,160,0.1);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 24px;
  --radius-full: 9999px;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --duration-fast: 150ms;
  --duration-normal: 300ms;
  --duration-slow: 500ms;
}

.gradio-container {
  background: var(--bg-primary) !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 0 !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
  color: var(--text-primary) !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

footer { display: none !important; }
.svelte-1ld8xk2 { display: none !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
::selection { background: rgba(124,92,252,0.3); color: var(--text-primary); }

.aurora-bg { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; overflow: hidden; pointer-events: none; }
.aurora-bg::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(ellipse 600px 400px at 20% 20%, rgba(124,92,252,0.08) 0%, transparent 70%), radial-gradient(ellipse 500px 500px at 80% 30%, rgba(0,229,160,0.05) 0%, transparent 70%), radial-gradient(ellipse 400px 600px at 50% 80%, rgba(0,180,216,0.06) 0%, transparent 70%);
  animation: aurora-drift 20s ease-in-out infinite alternate;
}
@keyframes aurora-drift { 0% { transform: translate(0, 0) rotate(0deg); } 33% { transform: translate(30px, -20px) rotate(1deg); } 66% { transform: translate(-20px, 30px) rotate(-1deg); } 100% { transform: translate(10px, 10px) rotate(0.5deg); } }

.app-header { padding: 32px 40px 24px; border-bottom: 1px solid var(--border-subtle); background: linear-gradient(180deg, rgba(124,92,252,0.03) 0%, transparent 100%); position: relative; }
.app-header h1 { font-size: 28px !important; font-weight: 700 !important; letter-spacing: -0.5px !important; background: var(--accent-gradient) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; margin: 0 !important; line-height: 1.2 !important; }
.app-header .tagline { color: var(--text-secondary); font-size: 14px; margin-top: 6px; font-weight: 400; }
.version-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: var(--accent-gradient-subtle); border: 1px solid var(--border-accent); border-radius: var(--radius-full); font-size: 11px; font-weight: 600; color: var(--accent-1); letter-spacing: 0.5px; text-transform: uppercase; }
.version-badge::before { content: ''; width: 6px; height: 6px; background: var(--accent-2); border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }

.glass-card { background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-lg) !important; padding: 24px !important; backdrop-filter: blur(20px) saturate(1.2) !important; -webkit-backdrop-filter: blur(20px) saturate(1.2) !important; transition: all var(--duration-normal) var(--ease-out) !important; position: relative; overflow: hidden; }
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%); }
.glass-card:hover { background: var(--bg-card-hover) !important; border-color: var(--border-medium) !important; box-shadow: var(--shadow-glow) !important; transform: translateY(-1px); }

.kpi-grid { display: grid !important; grid-template-columns: repeat(4, 1fr) !important; gap: 16px !important; margin: 24px 0 !important; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr) !important; } }
@media (max-width: 500px) { .kpi-grid { grid-template-columns: 1fr !important; } }
.kpi-card { background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-lg) !important; padding: 20px 24px !important; position: relative; overflow: hidden; transition: all var(--duration-normal) var(--ease-out); }
.kpi-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: var(--accent-gradient); opacity: 0; transition: opacity var(--duration-normal); }
.kpi-card:hover::after { opacity: 1; }
.kpi-card:hover { border-color: var(--border-accent) !important; box-shadow: var(--shadow-glow); }
.kpi-icon { font-size: 20px; margin-bottom: 8px; display: block; }
.kpi-label { font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; color: var(--text-tertiary) !important; margin-bottom: 4px !important; }
.kpi-value { font-size: 32px !important; font-weight: 700 !important; letter-spacing: -1px !important; color: var(--text-primary) !important; line-height: 1.1 !important; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 12px !important; color: var(--text-secondary) !important; margin-top: 4px !important; }
.kpi-value.accent-purple { color: var(--accent-1) !important; }
.kpi-value.accent-green { color: var(--accent-2) !important; }
.kpi-value.accent-blue { color: var(--accent-3) !important; }

.tab-nav { border-bottom: 1px solid var(--border-subtle) !important; gap: 4px !important; padding: 0 24px !important; margin-bottom: 0 !important; overflow-x: auto; }
.tab-nav button { background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; color: var(--text-tertiary) !important; font-size: 13px !important; font-weight: 500 !important; padding: 12px 20px !important; cursor: pointer !important; transition: all var(--duration-fast) var(--ease-out) !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important; letter-spacing: 0.2px !important; white-space: nowrap; }
.tab-nav button:hover { color: var(--text-secondary) !important; background: rgba(255,255,255,0.02) !important; }
.tab-nav button.selected { color: var(--text-primary) !important; border-bottom-color: var(--accent-1) !important; background: rgba(124,92,252,0.05) !important; }
.tabitem { background: transparent !important; border: none !important; padding: 24px !important; }

.upload-zone { border: 2px dashed var(--border-medium) !important; border-radius: var(--radius-xl) !important; padding: 48px 32px !important; text-align: center !important; background: var(--bg-card) !important; transition: all var(--duration-normal) var(--ease-out) !important; cursor: pointer !important; position: relative; overflow: hidden; }
.upload-zone::before { content: ''; position: absolute; inset: 0; background: var(--accent-gradient-subtle); opacity: 0; transition: opacity var(--duration-normal); pointer-events: none; }
.upload-zone button, .upload-zone input, .upload-zone label, .upload-zone [role="button"] { position: relative; z-index: 2; }
.upload-zone button { min-height: 44px !important; touch-action: manipulation; }
.upload-zone input[type="file"] { cursor: pointer; }
.upload-zone .file-preview { position: relative; z-index: 2; }
.audio-preview { margin-top: 14px !important; }
@media (pointer: coarse) {
  .upload-zone { min-height: 150px !important; }
  .upload-zone button, .upload-zone [role="button"] { min-width: 160px; padding: 12px 18px !important; }
}
.upload-zone:hover { border-color: var(--accent-1) !important; box-shadow: var(--shadow-glow) !important; }
.upload-zone:hover::before { opacity: 1; }
.upload-zone .upload-icon { font-size: 48px; margin-bottom: 16px; display: block; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
.upload-zone .upload-text { font-size: 16px; color: var(--text-secondary); font-weight: 500; }
.upload-zone .upload-hint { font-size: 12px; color: var(--text-tertiary); margin-top: 8px; }
.format-badges { display: flex; gap: 8px; justify-content: center; margin-top: 16px; flex-wrap: wrap; }
.format-badge { padding: 4px 10px; background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-full); font-size: 11px; font-weight: 600; color: var(--text-secondary); letter-spacing: 0.5px; text-transform: uppercase; }

.analyze-btn, .analyze-btn button { background: var(--accent-gradient) !important; color: #fff !important; border: none !important; border-radius: var(--radius-md) !important; padding: 16px 48px !important; font-size: 15px !important; font-weight: 700 !important; letter-spacing: 0.5px !important; cursor: pointer !important; transition: all var(--duration-normal) var(--ease-out) !important; box-shadow: 0 4px 24px rgba(124,92,252,0.3) !important; text-transform: uppercase !important; position: relative; overflow: hidden; }
.analyze-btn:hover, .analyze-btn button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 40px rgba(124,92,252,0.4) !important; }
.analyze-btn:active, .analyze-btn button:active { transform: translateY(0) !important; }
.analyze-btn::after { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%); animation: shimmer 3s ease-in-out infinite; }
@keyframes shimmer { 0% { transform: translateX(-100%) rotate(45deg); } 100% { transform: translateX(100%) rotate(45deg); } }

.progress-container { background: var(--bg-tertiary) !important; border-radius: var(--radius-full) !important; height: 6px !important; overflow: hidden !important; margin: 16px 0 !important; }
.progress-bar { height: 100% !important; background: var(--accent-gradient) !important; border-radius: var(--radius-full) !important; transition: width 0.5s var(--ease-out) !important; position: relative; }
.progress-bar::after { content: ''; position: absolute; right: 0; top: 0; bottom: 0; width: 40px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3)); animation: progress-glow 1.5s ease-in-out infinite; }
@keyframes progress-glow { 0%, 100% { opacity: 0; } 50% { opacity: 1; } }
.progress-step { font-size: 13px; color: var(--text-secondary); font-weight: 500; display: flex; align-items: center; gap: 8px; }
.progress-step .spinner { width: 14px; height: 14px; border: 2px solid var(--border-medium); border-top-color: var(--accent-1); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.data-table { width: 100% !important; border-collapse: separate !important; border-spacing: 0 !important; font-size: 13px !important; }
.data-table th { background: var(--bg-elevated) !important; color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1px !important; padding: 12px 16px !important; text-align: left !important; border-bottom: 1px solid var(--border-subtle) !important; }
.data-table td { padding: 12px 16px !important; border-bottom: 1px solid var(--border-subtle) !important; color: var(--text-primary) !important; font-variant-numeric: tabular-nums; }
.data-table tr:hover td { background: rgba(255,255,255,0.02) !important; }
.data-table tr:last-child td { border-bottom: none !important; }

.section-title { font-size: 18px !important; font-weight: 700 !important; color: var(--text-primary) !important; margin-bottom: 16px !important; display: flex !important; align-items: center !important; gap: 10px !important; }
.section-title .icon { width: 32px; height: 32px; background: var(--accent-gradient-subtle); border: 1px solid var(--border-accent); border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-size: 16px; }

.meter-gauge { background: var(--bg-tertiary); border-radius: var(--radius-md); padding: 24px; position: relative; }
.meter-track { height: 12px; background: var(--bg-primary); border-radius: var(--radius-full); position: relative; overflow: hidden; }
.meter-fill { height: 100%; border-radius: var(--radius-full); transition: width 1s var(--ease-out); }
.meter-fill.optimal { background: linear-gradient(90deg, var(--accent-2), #00ff88); }
.meter-fill.warning { background: linear-gradient(90deg, #ffa500, #ff6b6b); }
.meter-fill.danger { background: linear-gradient(90deg, #ff4444, #ff0000); }
.meter-markers { display: flex; justify-content: space-between; margin-top: 8px; font-size: 10px; color: var(--text-tertiary); font-weight: 600; letter-spacing: 0.5px; }

.report-content { background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-lg) !important; padding: 32px !important; line-height: 1.7 !important; }
.report-content h2 { font-size: 20px !important; font-weight: 700 !important; color: var(--text-primary) !important; border-bottom: 1px solid var(--border-subtle) !important; padding-bottom: 8px !important; margin-top: 32px !important; }
.report-content h3 { font-size: 16px !important; font-weight: 600 !important; color: var(--accent-1) !important; margin-top: 24px !important; }
.report-content ul li { color: var(--text-secondary) !important; margin-bottom: 8px !important; }
.report-content strong { color: var(--text-primary) !important; }
.report-content code { background: var(--bg-elevated) !important; padding: 2px 8px !important; border-radius: 4px !important; font-size: 12px !important; color: var(--accent-2) !important; }

.chart-container { background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-lg) !important; padding: 20px !important; margin-bottom: 16px !important; }
.chart-container .chart-title { font-size: 13px !important; font-weight: 600 !important; color: var(--text-secondary) !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; margin-bottom: 12px !important; }

@keyframes fade-up { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.animate-in { animation: fade-up 0.6s var(--ease-out) both; }
.animate-in:nth-child(1) { animation-delay: 0.05s; }
.animate-in:nth-child(2) { animation-delay: 0.1s; }
.animate-in:nth-child(3) { animation-delay: 0.15s; }
.animate-in:nth-child(4) { animation-delay: 0.2s; }
.animate-in:nth-child(5) { animation-delay: 0.25s; }
.animate-in:nth-child(6) { animation-delay: 0.3s; }

@media (max-width: 768px) {
  .app-header { padding: 20px 16px 16px !important; }
  .app-header h1 { font-size: 22px !important; }
  .glass-card { padding: 16px !important; }
  .kpi-value { font-size: 24px !important; }
  .tab-nav button { padding: 10px 12px !important; font-size: 12px !important; }
  .upload-zone { padding: 32px 16px !important; }
}

.gr-form, .gr-blocks { background: transparent !important; }
.gr-textbox textarea, .gr-textbox input { background: var(--bg-tertiary) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-sm) !important; color: var(--text-primary) !important; font-family: inherit !important; }
.gr-textbox textarea:focus, .gr-textbox input:focus { border-color: var(--accent-1) !important; box-shadow: 0 0 0 3px rgba(124,92,252,0.1) !important; outline: none !important; }
.gr-dropdown select { background: var(--bg-tertiary) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-sm) !important; color: var(--text-primary) !important; }
.gr-audio { border-radius: var(--radius-md) !important; }
.gr-info, .wrap.svelte-1p9xokt { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Shared helpers and data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioData:
    """Loaded audio and immutable metadata used by the analysis pipeline."""

    y_mono: np.ndarray
    y_stereo: np.ndarray
    sample_rate: int
    analysis_y: np.ndarray
    analysis_sr: int
    metadata: Mapping[str, Any]


def get_plotly_layout(title: str = "", xaxis_title: str = "", yaxis_title: str = "") -> Dict[str, Any]:
    """Return the shared dark Plotly layout used by every visualization.

    Args:
        title: Figure title.
        xaxis_title: Label for the horizontal axis.
        yaxis_title: Label for the vertical axis.

    Returns:
        A Plotly-compatible layout dictionary.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,6,10,0.6)",
        font=dict(color="#8892b0", family="Inter, system-ui, sans-serif", size=12),
        title=dict(text=title, font=dict(color="#f0f0f5", size=14, family="Inter"), x=0.02, xanchor="left"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11)),
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#4a5568"),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11)),
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="#4a5568"),
        ),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#11111b", bordercolor="rgba(124,92,252,0.3)", font=dict(color="#f0f0f5", size=12, family="Inter")),
        colorway=["#7c5cfc", "#00e5a0", "#00b4d8", "#ff6b6b", "#ffa500", "#ff69b4", "#9b59b6", "#1abc9c"],
        hovermode="x unified",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892b0", size=11), bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
    )


def _finalize_figure(figure: go.Figure) -> go.Figure:
    """Apply responsive Plotly configuration metadata consistently."""
    figure.update_layout(meta={"config": dict(PLOT_CONFIG)})
    # Gradio reads the figure, while this attribute also keeps the intended
    # config discoverable for integrations that serialize Figure objects.
    setattr(figure, "_config", dict(PLOT_CONFIG))
    return figure


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert scalar-like values to finite floats."""
    try:
        result = float(np.asarray(value).reshape(-1)[0])
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, IndexError):
        return default


def _clip01(value: Any) -> float:
    """Clamp a value to the unit interval."""
    return float(np.clip(_safe_float(value), 0.0, 1.0))


def _db(value: float, floor: float = -120.0) -> float:
    """Convert a positive linear value to decibels with a finite floor."""
    return max(floor, 20.0 * math.log10(max(abs(float(value)), EPSILON)))


def _note_from_midi(midi: float) -> str:
    """Convert a MIDI number to a conventional note name."""
    if not math.isfinite(float(midi)):
        return "—"
    rounded = int(round(float(midi)))
    octave = rounded // 12 - 1
    return f"{NOTE_NAMES[rounded % 12]}{octave}"


def _midi_from_hz(frequency: Any) -> np.ndarray:
    """Convert frequencies to MIDI, preserving NaN values."""
    values = np.asarray(frequency, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(np.maximum(values, EPSILON) / 440.0)


def _as_mono(y: np.ndarray) -> np.ndarray:
    """Return a finite mono float32 signal from common audio array layouts."""
    signal = np.asarray(y, dtype=np.float32)
    if signal.ndim == 1:
        mono = signal
    elif signal.ndim == 2 and signal.shape[0] <= signal.shape[1]:
        mono = np.mean(signal, axis=0)
    else:
        mono = np.mean(signal, axis=1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(mono), initial=0.0))
    if peak > 1.0:
        mono = mono / peak
    return mono.astype(np.float32, copy=False)


def _resample_for_analysis(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Downsample a signal above the analysis rate without upsampling it."""
    if sr <= ANALYSIS_SAMPLE_RATE:
        return np.asarray(y, dtype=np.float32), int(sr)
    try:
        resampled = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=ANALYSIS_SAMPLE_RATE, res_type="kaiser_fast")
        return np.asarray(resampled, dtype=np.float32), ANALYSIS_SAMPLE_RATE
    except Exception as exc:  # pragma: no cover - backend-specific fallback
        LOGGER.warning("Analysis-rate resampling failed: %s", exc)
        return np.asarray(y, dtype=np.float32), int(sr)


def _empty_array(dtype: Any = float) -> np.ndarray:
    """Create a consistently shaped empty numeric array."""
    return np.asarray([], dtype=dtype)


def _error_result(message: str) -> Dict[str, Any]:
    """Return the standard graceful-degradation result shape."""
    return {"error": str(message)}


# ---------------------------------------------------------------------------
# Audio analysis engine
# ---------------------------------------------------------------------------


class AudioAnalyzer:
    """Modular musicological and audio-engineering analysis engine."""

    def __init__(self, max_file_bytes: int = MAX_FILE_BYTES) -> None:
        """Initialize an analyzer with a file-size guard.

        Args:
            max_file_bytes: Maximum accepted input size in bytes.
        """
        self.max_file_bytes = int(max_file_bytes)

    def load_audio(self, file_path: str) -> AudioData:
        """Load an audio file, preserving stereo and preparing mono analysis data.

        Args:
            file_path: Path to an MP3, WAV, FLAC, OGG, or M4A file.

        Returns:
            Loaded audio with native-rate stereo and analysis-rate mono arrays.

        Raises:
            ValueError: If the path, extension, or size is invalid.
            RuntimeError: If the decoding backend cannot read the file.
        """
        if not file_path:
            raise ValueError("Choose an audio file before starting analysis.")
        path = Path(str(file_path))
        if not path.exists() or not path.is_file():
            raise ValueError("The selected audio file could not be found.")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported file type. Use MP3, WAV, FLAC, OGG, or M4A.")
        file_size = path.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(f"This file is {file_size / (1024 * 1024):.1f} MB. The maximum size is 50 MB.")

        try:
            loaded, sr = librosa.load(str(path), sr=None, mono=False)
        except Exception as exc:
            raise RuntimeError(f"Audio decoding failed: {exc}. Try exporting the track as WAV or FLAC.") from exc

        raw = np.asarray(loaded, dtype=np.float32)
        if raw.ndim == 1:
            stereo = np.vstack((raw, raw))
        elif raw.shape[0] <= raw.shape[1]:
            stereo = raw[:2]
            if stereo.shape[0] == 1:
                stereo = np.vstack((stereo, stereo))
        else:
            stereo = raw.T[:2]
            if stereo.shape[0] == 1:
                stereo = np.vstack((stereo, stereo))
        stereo = np.nan_to_num(stereo, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(stereo), initial=0.0))
        if peak > 1.0:
            stereo = stereo / peak
        mono = _as_mono(stereo)
        analysis_y, analysis_sr = _resample_for_analysis(mono, int(sr))

        bit_depth: Any = "Unknown"
        subtype: Any = "Unknown"
        try:
            import soundfile as sf

            info = sf.info(str(path))
            subtype = info.subtype or "Unknown"
            match = re.search(r"(\d+)", str(subtype))
            bit_depth = int(match.group(1)) if match else subtype
        except Exception as exc:
            LOGGER.warning("Metadata inspection failed for %s: %s", path.name, exc)

        metadata: Dict[str, Any] = {
            "filename": path.name,
            "file_size_bytes": file_size,
            "duration": float(mono.size / max(int(sr), 1)),
            "sample_rate": int(sr),
            "analysis_sample_rate": int(analysis_sr),
            "bit_depth": bit_depth,
            "subtype": subtype,
            "channel_count": 1 if raw.ndim == 1 else int(raw.shape[0] if raw.shape[0] <= raw.shape[1] else raw.shape[1]),
        }
        if metadata["duration"] > LONG_TRACK_SECONDS:
            LOGGER.warning("Track %s is longer than ten minutes; analysis may take longer.", path.name)
        return AudioData(mono, stereo, int(sr), analysis_y, int(analysis_sr), metadata)

    def analyze_tempo_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Estimate tempo, beats, time signature, syncopation, and groove.

        Args:
            y: Mono floating-point audio.
            sr: Sample rate of ``y``.

        Returns:
            Tempo and rhythm feature dictionary, or an ``error`` dictionary.
        """
        try:
            signal = _as_mono(y)
            onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
            tempo, beat_frames = librosa.beat.beat_track(
                y=signal,
                sr=sr,
                onset_envelope=onset,
                hop_length=DEFAULT_HOP_LENGTH,
                start_bpm=120.0,
                units="frames",
            )
            bpm = _safe_float(tempo, 120.0)
            if bpm <= 0.0:
                bpm = 120.0
            beat_frames_array = np.asarray(beat_frames, dtype=int)
            beat_times = librosa.frames_to_time(beat_frames_array, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
            intervals = np.diff(beat_times)
            if intervals.size:
                interval_cv = float(np.std(intervals) / max(np.mean(intervals), EPSILON))
                groove = _clip01(1.0 - interval_cv * 2.0)
            else:
                groove = 0.0
            confidence = _clip01(1.0 / (1.0 + (float(np.std(intervals) / max(np.mean(intervals), EPSILON)) if intervals.size else 1.0)))
            time_signature, signature_confidence = self._estimate_time_signature(onset, beat_frames_array, sr)
            syncopation = self._estimate_syncopation(onset, beat_frames_array)
            return {
                "bpm": float(bpm),
                "bpm_confidence": confidence,
                "beat_frames": beat_frames_array,
                "beat_times": np.asarray(beat_times),
                "time_signature": time_signature,
                "time_signature_confidence": signature_confidence,
                "onset_strength": np.asarray(onset),
                "onset_times": librosa.times_like(onset, sr=sr, hop_length=DEFAULT_HOP_LENGTH),
                "syncopation_index": syncopation,
                "groove_consistency": groove,
            }
        except Exception as exc:
            LOGGER.warning("Tempo/rhythm analysis unavailable: %s", exc)
            return _error_result(f"Tempo and rhythm analysis unavailable: {exc}")

    def _estimate_time_signature(self, onset: np.ndarray, beat_frames: np.ndarray, sr: int) -> Tuple[str, float]:
        """Estimate a common time signature using beat-group energy regularity."""
        if beat_frames.size < 4:
            return "4/4", 0.35
        valid = beat_frames[(beat_frames >= 0) & (beat_frames < onset.size)]
        strengths = onset[valid] if valid.size else np.ones(4)
        strengths = np.asarray(strengths, dtype=float)
        scores: Dict[str, float] = {}
        for signature, beats_per_bar in (("4/4", 4), ("3/4", 3), ("6/8", 6)):
            groups = strengths[: (strengths.size // beats_per_bar) * beats_per_bar]
            if groups.size < beats_per_bar:
                scores[signature] = 0.0
                continue
            matrix = groups.reshape(-1, beats_per_bar)
            downbeat_strength = np.mean(matrix[:, 0])
            regularity = 1.0 - min(1.0, float(np.std(np.mean(matrix, axis=0)) / (np.mean(matrix) + EPSILON)))
            scores[signature] = max(0.0, float(downbeat_strength / (np.mean(matrix) + EPSILON))) * regularity
        selected = max(scores, key=scores.get)
        all_scores = np.asarray(list(scores.values()), dtype=float)
        confidence = _clip01((scores[selected] - float(np.mean(all_scores))) / (float(np.std(all_scores)) + 0.25) + 0.5)
        return selected, confidence

    def _estimate_syncopation(self, onset: np.ndarray, beat_frames: np.ndarray) -> float:
        """Compare off-beat onset energy against on-beat energy."""
        if beat_frames.size < 2 or onset.size == 0:
            return 0.0
        valid_beats = beat_frames[(beat_frames >= 0) & (beat_frames < onset.size)]
        if valid_beats.size < 2:
            return 0.0
        onbeat = onset[valid_beats]
        midpoints = ((valid_beats[:-1] + valid_beats[1:]) / 2.0).astype(int)
        midpoints = midpoints[(midpoints >= 0) & (midpoints < onset.size)]
        offbeat = onset[midpoints] if midpoints.size else np.zeros(1)
        ratio = float(np.mean(offbeat) / (np.mean(onbeat) + np.mean(offbeat) + EPSILON))
        return _clip01(ratio * 2.0)

    def analyze_key_scale(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Find the most likely major or minor key with Krumhansl profiles."""
        try:
            signal = _as_mono(y)
            chroma = librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=DEFAULT_HOP_LENGTH, n_fft=STFT_SIZE)
            mean_chroma = np.mean(chroma, axis=1)
            major_profile = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            scores: Dict[str, float] = {}
            for index, note in enumerate(KEY_NAMES):
                major = np.roll(major_profile, index)
                minor = np.roll(minor_profile, index)
                scores[f"{note} Major"] = self._correlation(mean_chroma, major)
                scores[f"{note} Minor"] = self._correlation(mean_chroma, minor)
            best = max(scores, key=scores.get)
            key, mode = best.rsplit(" ", 1)
            sorted_scores = sorted(scores.values(), reverse=True)
            separation = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            confidence = _clip01(0.5 + scores[best] / 2.0 + separation)
            return {
                "key": key,
                "mode": mode,
                "key_full": best,
                "confidence": confidence,
                "chroma_energy": np.asarray(mean_chroma),
                "chroma_matrix": np.asarray(chroma),
                "key_profile_correlation": scores,
            }
        except Exception as exc:
            LOGGER.warning("Key analysis unavailable: %s", exc)
            return _error_result(f"Key and scale analysis unavailable: {exc}")

    @staticmethod
    def _correlation(left: np.ndarray, right: np.ndarray) -> float:
        """Calculate a finite normalized correlation for two feature vectors."""
        left_centered = np.asarray(left, dtype=float) - float(np.mean(left))
        right_centered = np.asarray(right, dtype=float) - float(np.mean(right))
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denominator <= EPSILON:
            return 0.0
        return float(np.dot(left_centered, right_centered) / denominator)

    def analyze_chords(self, y: np.ndarray, sr: int, hop_length: int = DEFAULT_HOP_LENGTH) -> Dict[str, Any]:
        """Match beat-synchronous chroma against 84 chord templates."""
        try:
            signal = _as_mono(y)
            chroma = librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=hop_length, n_fft=STFT_SIZE)
            tempo_data = self.analyze_tempo_rhythm(signal, sr)
            beats = np.asarray(tempo_data.get("beat_frames", []), dtype=int) if "error" not in tempo_data else np.asarray([], dtype=int)
            if beats.size:
                boundaries = np.unique(np.concatenate(([0], beats, [chroma.shape[1]])))
            else:
                frames_per_segment = max(1, int(round(2.0 * sr / hop_length)))
                boundaries = np.arange(0, chroma.shape[1] + frames_per_segment, frames_per_segment, dtype=int)
                boundaries[-1] = min(boundaries[-1], chroma.shape[1])
                boundaries = np.unique(boundaries)
            templates = self._chord_templates()
            candidates: List[Tuple[int, int, str]] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end <= start:
                    continue
                vector = np.mean(chroma[:, start:end], axis=1)
                best_name = "N.C."
                best_score = -1.0
                if np.linalg.norm(vector) > EPSILON:
                    for name, template in templates.items():
                        similarity = float(np.dot(vector, template) / (np.linalg.norm(vector) * np.linalg.norm(template) + EPSILON))
                        if similarity > best_score:
                            best_name, best_score = name, similarity
                candidates.append((int(start), int(end), best_name))
            merged: List[Dict[str, Any]] = []
            for start_frame, end_frame, chord in candidates:
                start_time = float(librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length))
                end_time = float(librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length))
                duration = max(0.05, end_time - start_time)
                if merged and merged[-1]["chord"] == chord:
                    merged[-1]["duration"] += duration
                else:
                    merged.append({"time": start_time, "duration": duration, "chord": chord})
            sequence = [str(item["chord"]) for item in merged]
            unique = list(dict.fromkeys(sequence))
            extended_count = sum(1 for name in unique if any(token in name for token in ("7", "aug", "dim")))
            complexity = _clip01((len(unique) / max(len(sequence), 1)) * 0.75 + (extended_count / max(len(unique), 1)) * 0.25)
            return {"chords": merged, "chord_sequence": sequence, "unique_chords": unique, "harmonic_complexity": complexity}
        except Exception as exc:
            LOGGER.warning("Chord analysis unavailable: %s", exc)
            return _error_result(f"Chord progression analysis unavailable: {exc}")

    @staticmethod
    def _chord_templates() -> Dict[str, np.ndarray]:
        """Build normalized templates for the seven supported chord qualities."""
        qualities: Mapping[str, Tuple[int, ...]] = {
            "": (0, 4, 7),
            "m": (0, 3, 7),
            "dim": (0, 3, 6),
            "aug": (0, 4, 8),
            "7": (0, 4, 7, 10),
            "maj7": (0, 4, 7, 11),
            "m7": (0, 3, 7, 10),
        }
        templates: Dict[str, np.ndarray] = {}
        for root_index, root in enumerate(NOTE_NAMES):
            for quality, intervals in qualities.items():
                vector = np.zeros(12, dtype=float)
                vector[(root_index + np.asarray(intervals)) % 12] = 1.0
                templates[f"{root}{quality}"] = vector
        return templates

    def analyze_structure(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect novelty boundaries and heuristically label song sections."""
        try:
            signal = _as_mono(y)
            duration = float(signal.size / max(sr, 1))
            if duration <= MIN_SECTION_SECONDS * 1.25:
                return {"segments": [{"start": 0.0, "end": max(duration, 0.1), "label": "Intro"}], "segment_labels": ["Intro"], "num_segments": 1}
            features = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=STRUCTURE_HOP_LENGTH)
            step = max(1, int(math.ceil(features.shape[1] / MAX_STRUCTURE_FEATURE_FRAMES)))
            sampled = features[:, ::step]
            # Recurrence is intentionally part of the structural signal; its
            # diagonal affinity captures repeated verse/chorus material.
            recurrence = librosa.segment.recurrence_matrix(sampled, metric="cosine", mode="affinity", sym=True)
            recurrence_novelty = np.mean(np.abs(np.diff(recurrence, axis=1)), axis=0) if recurrence.shape[1] > 1 else np.zeros(1)
            feature_novelty = np.mean(np.abs(np.diff(sampled, axis=1)), axis=0) if sampled.shape[1] > 1 else np.zeros(1)
            novelty = np.asarray(feature_novelty, dtype=float)
            if recurrence_novelty.size == novelty.size:
                novelty = novelty + recurrence_novelty
            novelty = novelty / (float(np.max(novelty)) + EPSILON)
            min_distance = max(1, int(round(MIN_SECTION_SECONDS * sr / STRUCTURE_HOP_LENGTH / step)))
            peaks, properties = find_peaks(novelty, distance=min_distance, prominence=0.10)
            if peaks.size:
                prominence = properties.get("prominences", np.ones(peaks.size))
                order = np.argsort(prominence)[::-1][: max(1, min(11, int(duration // MIN_SECTION_SECONDS)))]
                selected = np.sort(peaks[order])
            else:
                selected = np.asarray([], dtype=int)
            boundary_frames = np.concatenate(([0], selected * step, [features.shape[1] - 1]))
            boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=STRUCTURE_HOP_LENGTH)
            boundary_times = np.unique(np.clip(boundary_times, 0.0, duration))
            boundary_times[0] = 0.0
            boundary_times[-1] = duration
            if boundary_times.size < 2:
                boundary_times = np.asarray([0.0, duration])
            segment_energies: List[float] = []
            segments: List[Dict[str, Any]] = []
            for start, end in zip(boundary_times[:-1], boundary_times[1:]):
                start_index = int(max(0, start * sr))
                end_index = int(min(signal.size, max(start_index + 1, end * sr)))
                segment_energies.append(float(np.sqrt(np.mean(signal[start_index:end_index] ** 2))))
                segments.append({"start": float(start), "end": float(end), "label": "Section"})
            energies = np.asarray(segment_energies)
            chorus_index = int(np.argmax(energies)) if energies.size else -1
            for index, segment in enumerate(segments):
                length = segment["end"] - segment["start"]
                if index == 0 and len(segments) > 1:
                    label = "Intro"
                elif index == len(segments) - 1 and len(segments) > 1:
                    label = "Outro"
                elif index == chorus_index:
                    label = "Chorus"
                elif length < MIN_SECTION_SECONDS * 1.2:
                    label = "Bridge"
                else:
                    label = "Verse"
                segment["label"] = label
            counts: Dict[str, int] = {}
            for segment in segments:
                label = str(segment["label"])
                counts[label] = counts.get(label, 0) + 1
                if counts[label] > 1 and label in {"Verse", "Chorus", "Bridge"}:
                    segment["label"] = f"{label} {counts[label]}"
            labels = [str(item["label"]) for item in segments]
            return {"segments": segments, "segment_labels": labels, "num_segments": len(segments)}
        except Exception as exc:
            LOGGER.warning("Structure analysis unavailable: %s", exc)
            return _error_result(f"Song structure analysis unavailable: {exc}")

    def analyze_melody(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract a confidence-filtered melodic pitch contour with pYIN."""
        try:
            signal = _as_mono(y)
            fmin = float(librosa.note_to_hz("C2"))
            fmax = float(librosa.note_to_hz("C7"))
            f0, voiced_flag, voiced_prob = librosa.pyin(
                signal,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=STFT_SIZE,
                hop_length=DEFAULT_HOP_LENGTH,
                fill_na=np.nan,
            )
            confidence = np.nan_to_num(np.asarray(voiced_prob, dtype=float), nan=0.0)
            contour = np.asarray(f0, dtype=float)
            contour[confidence < PITCH_CONFIDENCE_THRESHOLD] = np.nan
            times = librosa.times_like(contour, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
            midi = _midi_from_hz(contour)
            valid = np.isfinite(contour) & (confidence >= PITCH_CONFIDENCE_THRESHOLD)
            pitch_notes = [_note_from_midi(value) if valid[index] else "—" for index, value in enumerate(midi)]
            valid_midi = midi[valid]
            intervals = np.diff(np.round(valid_midi).astype(int)).astype(int).tolist() if valid_midi.size > 1 else []
            return {
                "pitch_contour": contour,
                "pitch_times": np.asarray(times),
                "pitch_notes": pitch_notes,
                "pitch_confidence": confidence,
                "vocal_range_low": _note_from_midi(float(np.min(valid_midi))) if valid_midi.size else "—",
                "vocal_range_high": _note_from_midi(float(np.max(valid_midi))) if valid_midi.size else "—",
                "melodic_intervals": intervals,
            }
        except Exception as exc:
            LOGGER.warning("Melody analysis unavailable: %s", exc)
            return _error_result(f"Melody analysis unavailable: {exc}")

    def analyze_bassline(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Isolate 30–300 Hz, track bass pitch, and build a note histogram."""
        try:
            signal = _as_mono(y)
            nyquist = sr / 2.0
            high = min(MAX_PITCH_HZ, nyquist * 0.92)
            low = min(MIN_PITCH_HZ, high * 0.4)
            if high <= low:
                raise ValueError("Sample rate is too low for bass isolation")
            sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
            filtered = sosfilt(sos, signal).astype(np.float32)
            energy = librosa.feature.rms(y=filtered, frame_length=STFT_SIZE, hop_length=DEFAULT_HOP_LENGTH)[0]
            try:
                f0, _, voiced_prob = librosa.pyin(
                    filtered,
                    fmin=max(20.0, low),
                    fmax=high,
                    sr=sr,
                    frame_length=STFT_SIZE,
                    hop_length=DEFAULT_HOP_LENGTH,
                    fill_na=np.nan,
                )
                confidence = np.nan_to_num(np.asarray(voiced_prob, dtype=float), nan=0.0)
                contour = np.asarray(f0, dtype=float)
                contour[confidence < PITCH_CONFIDENCE_THRESHOLD] = np.nan
            except Exception:
                contour = np.full(energy.shape, np.nan, dtype=float)
            midi = _midi_from_hz(contour)
            notes = [_note_from_midi(value) if math.isfinite(float(value)) else "—" for value in midi]
            valid_notes = [note for note in notes if note != "—"]
            histogram: Dict[str, int] = {}
            for note in valid_notes:
                histogram[note] = histogram.get(note, 0) + 1
            roots: List[str] = []
            for note in valid_notes:
                if not roots or roots[-1] != note:
                    roots.append(note)
            times = librosa.times_like(energy, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
            return {
                "bass_pitch_contour": contour,
                "bass_times": np.asarray(times),
                "bass_notes": notes,
                "bass_root_notes": roots,
                "bass_frequency_distribution": histogram,
                "bass_energy_over_time": np.asarray(energy),
                "bass_energy_times": np.asarray(times),
            }
        except Exception as exc:
            LOGGER.warning("Bassline analysis unavailable: %s", exc)
            return _error_result(f"Bassline analysis unavailable: {exc}")

    def analyze_loudness_dynamics(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Measure LUFS, true peak, RMS, crest factor, and dynamic range."""
        try:
            signal = _as_mono(y)
            try:
                meter = pyln.Meter(sr)
                integrated = float(meter.integrated_loudness(signal))
            except Exception:
                integrated = _db(float(np.sqrt(np.mean(signal**2)) + EPSILON)) - 0.691
            if not math.isfinite(integrated):
                integrated = LUFS_FLOOR
            window_short = max(1, int(round(3.0 * sr)))
            window_momentary = max(1, int(round(0.4 * sr)))
            short = self._window_loudness(signal, sr, window_short)
            momentary = self._window_loudness(signal, sr, window_momentary)
            loudness_over_time = short if short.size else momentary
            lra = float(np.percentile(short, 95) - np.percentile(short, 10)) if short.size > 1 else 0.0
            try:
                peak_signal = resample_poly(signal, TRUE_PEAK_OVERSAMPLE, 1)
            except Exception:
                peak_signal = signal
            true_peak_linear = float(np.max(np.abs(peak_signal), initial=0.0))
            true_peak = _db(true_peak_linear)
            rms = float(np.sqrt(np.mean(signal**2) + EPSILON))
            rms_db = _db(rms)
            crest_factor = true_peak - rms_db
            dynamic_range = float(np.percentile(loudness_over_time, 95) - np.percentile(loudness_over_time, 10)) if loudness_over_time.size > 1 else 0.0
            platforms = {name: {"target": target, "delta": float(target - integrated)} for name, target in PLATFORM_TARGETS.items()}
            return {
                "integrated_lufs": float(integrated),
                "loudness_range_lra": max(0.0, lra),
                "true_peak_dbtp": float(true_peak),
                "rms_db": float(rms_db),
                "crest_factor_db": float(crest_factor),
                "dynamic_range_db": max(0.0, dynamic_range),
                "short_term_lufs": short,
                "momentary_lufs": momentary,
                "loudness_over_time": loudness_over_time,
                "loudness_times": np.arange(loudness_over_time.size) * (len(signal) / max(sr, 1) / max(loudness_over_time.size, 1)),
                "platform_comparison": platforms,
            }
        except Exception as exc:
            LOGGER.warning("Loudness analysis unavailable: %s", exc)
            return _error_result(f"Loudness and dynamics analysis unavailable: {exc}")

    @staticmethod
    def _window_loudness(signal: np.ndarray, sr: int, window: int) -> np.ndarray:
        """Calculate an approximate LUFS timeline from overlapping RMS windows."""
        if signal.size == 0:
            return _empty_array()
        hop = max(1, window // 4)
        if signal.size < window:
            padded = np.pad(signal, (0, window - signal.size))
            rms = float(np.sqrt(np.mean(padded**2) + EPSILON))
            return np.asarray([max(LUFS_FLOOR, _db(rms) - 0.691)])
        values = []
        for start in range(0, signal.size - window + 1, hop):
            rms = float(np.sqrt(np.mean(signal[start : start + window] ** 2) + EPSILON))
            values.append(max(LUFS_FLOOR, _db(rms) - 0.691))
        return np.asarray(values, dtype=float)

    def analyze_frequency_spectrum(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Compute FFT spectrum, frequency-band energy, centroid, rolloff, and bandwidth."""
        try:
            signal = _as_mono(y)
            if signal.size == 0:
                raise ValueError("Audio signal is empty")
            limit = min(signal.size, 2**20)
            segment = signal[:limit]
            frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sr)
            fft = np.fft.rfft(segment * np.hanning(segment.size))
            power = np.abs(fft) ** 2
            usable = (frequencies >= 20.0) & (frequencies <= min(20_000.0, sr / 2.0))
            total_energy = float(np.sum(power[usable]) + EPSILON)
            bands: Dict[str, Dict[str, Any]] = {}
            for name, (low, high, label) in FREQUENCY_BANDS.items():
                mask = (frequencies >= low) & (frequencies < min(high, sr / 2.0))
                energy = float(np.sum(power[mask]))
                bands[name] = {"range": f"{int(low):,}-{int(high):,} Hz", "energy_db": 10.0 * math.log10(max(energy / total_energy, EPSILON)), "percentage": energy / total_energy * 100.0, "label": label}
            display_indices = np.linspace(0, max(0, frequencies.size - 1), min(MAX_PLOT_POINTS, frequencies.size), dtype=int)
            display_frequencies = frequencies[display_indices]
            display_power = power[display_indices]
            if display_power.size >= SPECTRUM_SMOOTHING_BINS:
                kernel = np.ones(SPECTRUM_SMOOTHING_BINS, dtype=float) / SPECTRUM_SMOOTHING_BINS
                display_power = np.convolve(display_power, kernel, mode="same")
            relative_magnitude = 10.0 * np.log10(np.maximum(display_power / (float(np.max(power)) + EPSILON), EPSILON))
            positive_power = power[usable]
            positive_frequencies = frequencies[usable]
            centroid = float(np.sum(positive_frequencies * positive_power) / (np.sum(positive_power) + EPSILON))
            cumulative = np.cumsum(positive_power)
            rolloff_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.85)) if cumulative.size else 0
            rolloff = float(positive_frequencies[min(rolloff_index, max(positive_frequencies.size - 1, 0))]) if positive_frequencies.size else 0.0
            bandwidth = float(np.sqrt(np.sum(((positive_frequencies - centroid) ** 2) * positive_power) / (np.sum(positive_power) + EPSILON))) if positive_frequencies.size else 0.0
            return {"frequency_bands": bands, "spectrum_frequencies": display_frequencies, "spectrum_magnitudes": relative_magnitude, "spectral_centroid": centroid, "spectral_rolloff": rolloff, "spectral_bandwidth": bandwidth}
        except Exception as exc:
            LOGGER.warning("Frequency analysis unavailable: %s", exc)
            return _error_result(f"Frequency spectrum analysis unavailable: {exc}")

    def analyze_stereo_field(self, y_stereo: np.ndarray, sr: int) -> Dict[str, Any]:
        """Measure width, phase correlation, mid/side ratio, balance, and width over time."""
        try:
            stereo = np.asarray(y_stereo, dtype=float)
            if stereo.ndim == 1:
                stereo = np.vstack((stereo, stereo))
            elif stereo.shape[0] > stereo.shape[1]:
                stereo = stereo.T
            if stereo.shape[0] < 2:
                stereo = np.vstack((stereo[0], stereo[0]))
            left, right = stereo[0], stereo[1]
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            mid_energy = float(np.mean(mid**2))
            side_energy = float(np.mean(side**2))
            energy_sum = mid_energy + side_energy
            stereo_width = _clip01(side_energy / energy_sum) if energy_sum > EPSILON else 0.0
            phase_denominator = math.sqrt(float(np.mean(left**2) * np.mean(right**2)))
            phase = float(np.mean(left * right) / phase_denominator) if phase_denominator > EPSILON else 0.0
            phase = float(np.clip(phase, -1.0, 1.0))
            mid_side_ratio = float(mid_energy / max(side_energy, EPSILON))
            left_energy = float(np.mean(left**2))
            right_energy = float(np.mean(right**2))
            balance = float(np.clip((right_energy - left_energy) / (right_energy + left_energy + EPSILON), -1.0, 1.0))
            window = max(1, int(round(sr * 0.5)))
            hop = max(1, window // 2)
            widths: List[float] = []
            for start in range(0, max(1, len(left) - window + 1), hop):
                l_chunk, r_chunk = left[start : start + window], right[start : start + window]
                if l_chunk.size == 0 or r_chunk.size == 0:
                    continue
                mid_chunk = (l_chunk + r_chunk) / 2.0
                side_chunk = (l_chunk - r_chunk) / 2.0
                me = float(np.mean(mid_chunk**2))
                se = float(np.mean(side_chunk**2))
                widths.append(_clip01(se / (me + se + EPSILON)))
            return {"stereo_width": stereo_width, "phase_correlation": phase, "mid_side_ratio": mid_side_ratio, "balance_lr": balance, "width_over_time": np.asarray(widths), "width_times": np.arange(len(widths)) * (window / max(sr, 1) / 2.0)}
        except Exception as exc:
            LOGGER.warning("Stereo analysis unavailable: %s", exc)
            return _error_result(f"Stereo-field analysis unavailable: {exc}")

    def iter_analyze(self, file_path: str) -> Iterator[Tuple[int, str, Dict[str, Any]]]:
        """Yield progressively populated analysis results for UI progress updates."""
        data = self.load_audio(file_path)
        results: Dict[str, Any] = {"metadata": dict(data.metadata), "_audio": data}
        yield 1, ANALYSIS_STEPS[0], results
        results["tempo"] = self.analyze_tempo_rhythm(data.analysis_y, data.analysis_sr)
        yield 2, ANALYSIS_STEPS[1], results
        results["key"] = self.analyze_key_scale(data.analysis_y, data.analysis_sr)
        yield 3, ANALYSIS_STEPS[2], results
        results["chords"] = self.analyze_chords(data.analysis_y, data.analysis_sr)
        results["melody"] = self.analyze_melody(data.analysis_y, data.analysis_sr)
        yield 4, ANALYSIS_STEPS[3], results
        results["spectrum"] = self.analyze_frequency_spectrum(data.analysis_y, data.analysis_sr)
        yield 5, ANALYSIS_STEPS[4], results
        results["loudness"] = self.analyze_loudness_dynamics(data.analysis_y, data.analysis_sr)
        yield 6, ANALYSIS_STEPS[5], results
        results["stereo"] = self.analyze_stereo_field(data.y_stereo, data.sample_rate)
        results["structure"] = self.analyze_structure(data.analysis_y, data.analysis_sr)
        results["bassline"] = self.analyze_bassline(data.analysis_y, data.analysis_sr)
        yield 7, ANALYSIS_STEPS[6], results
        results["report"] = ReportGenerator().generate_report(results)
        yield 8, ANALYSIS_STEPS[7], results

    def analyze(self, file_path: str, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run the complete analysis synchronously.

        Args:
            file_path: Input audio path.
            progress_callback: Optional callback receiving ``(step, message)``.

        Returns:
            Complete result dictionary with graceful per-module errors.
        """
        result: Dict[str, Any] = {}
        for step, message, result in self.iter_analyze(file_path):
            if progress_callback is not None:
                progress_callback(step, message)
        return result

    analyze_file = analyze


# Public functional entry points mirror the class API for notebook and test use.
def analyze_tempo_rhythm(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze tempo and rhythm without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_tempo_rhythm(y, sr)


def analyze_key_scale(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze key and scale without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_key_scale(y, sr)


def analyze_chords(y: np.ndarray, sr: int, hop_length: int = DEFAULT_HOP_LENGTH) -> Dict[str, Any]:
    """Analyze chord progression without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_chords(y, sr, hop_length)


def analyze_structure(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze song structure without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_structure(y, sr)


def analyze_melody(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze melodic pitch without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_melody(y, sr)


def analyze_bassline(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze bassline content without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_bassline(y, sr)


def analyze_loudness_dynamics(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze loudness and dynamics without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_loudness_dynamics(y, sr)


def analyze_frequency_spectrum(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze frequency balance without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_frequency_spectrum(y, sr)


def analyze_stereo_field(y_stereo: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze stereo width and phase without constructing an analyzer manually."""
    return AudioAnalyzer().analyze_stereo_field(y_stereo, sr)


# ---------------------------------------------------------------------------
# Interactive visualizations
# ---------------------------------------------------------------------------


class Visualizer:
    """Create responsive, dark-theme Plotly visualizations for analysis data."""

    def plot_waveform_structure(self, y: np.ndarray, sr: int, structure: Mapping[str, Any], beat_times: Optional[np.ndarray] = None) -> go.Figure:
        """Plot a downsampled waveform with section bands and beat markers."""
        signal = _as_mono(y)
        indices = np.linspace(0, max(0, signal.size - 1), min(MAX_PLOT_POINTS, max(signal.size, 1)), dtype=int)
        times = indices / max(sr, 1)
        figure = go.Figure(go.Scattergl(x=times, y=signal[indices], mode="lines", line=dict(color="#7c5cfc", width=1), name="Waveform", hovertemplate="%{x:.2f}s<br>%{y:.3f}<extra></extra>"))
        section_colors = {"Intro": "#00b4d8", "Verse": "#00e5a0", "Chorus": "#7c5cfc", "Bridge": "#ffa500", "Outro": "#4a5568", "Transition": "#ff6b6b"}
        shapes: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        for item in structure.get("segments", []) if "error" not in structure else []:
            label = str(item.get("label", "Section"))
            color = next((value for key, value in section_colors.items() if label.startswith(key)), "#4a5568")
            shapes.append(dict(type="rect", x0=float(item.get("start", 0.0)), x1=float(item.get("end", 0.0)), y0=-1.0, y1=1.0, fillcolor=color, opacity=0.08, line_width=0, layer="below"))
            annotations.append(dict(x=(float(item.get("start", 0.0)) + float(item.get("end", 0.0))) / 2.0, y=1.05, text=html.escape(label), showarrow=False, font=dict(size=10, color=color)))
        for beat in np.asarray(beat_times if beat_times is not None else [], dtype=float)[:300]:
            shapes.append(dict(type="line", x0=float(beat), x1=float(beat), y0=-1.0, y1=1.0, line=dict(color="rgba(255,255,255,0.16)", width=1, dash="dot")))
        layout = get_plotly_layout("Waveform & Structure", "Time (seconds)", "Amplitude")
        layout.update(shapes=shapes, annotations=annotations, yaxis=dict(layout["yaxis"], range=[-1.05, 1.15], fixedrange=False), height=360)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_chord_progression(self, chords: Mapping[str, Any]) -> go.Figure:
        """Plot a labeled horizontal chord progression timeline."""
        figure = go.Figure()
        palette = ["#7c5cfc", "#00e5a0", "#00b4d8", "#ffa500", "#ff6b6b", "#ff69b4"]
        shapes: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        for index, item in enumerate(chords.get("chords", []) if "error" not in chords else []):
            start = float(item.get("time", 0.0))
            end = start + float(item.get("duration", 0.1))
            color = palette[index % len(palette)]
            shapes.append(dict(type="rect", x0=start, x1=end, y0=0, y1=1, fillcolor=color, opacity=0.78, line=dict(color=color, width=1)))
            annotations.append(dict(x=(start + end) / 2.0, y=0.5, text=html.escape(str(item.get("chord", "N.C."))), showarrow=False, font=dict(color="#06060a", size=11)))
        layout = get_plotly_layout("Chord Progression", "Time (seconds)", "")
        layout.update(shapes=shapes, annotations=annotations, yaxis=dict(layout["yaxis"], range=[0, 1], showticklabels=False), height=360)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_spectrogram(self, y: np.ndarray, sr: int) -> go.Figure:
        """Create a log-frequency interactive dB spectrogram."""
        signal = _as_mono(y)
        stft = librosa.stft(signal, n_fft=STFT_SIZE, hop_length=DEFAULT_HOP_LENGTH)
        db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=STFT_SIZE)
        mask = (freqs >= 20.0) & (freqs <= min(20_000.0, sr / 2.0))
        db = db[mask]
        freqs = freqs[mask]
        time = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=DEFAULT_HOP_LENGTH)
        time_indices = np.linspace(0, max(0, db.shape[1] - 1), min(MAX_SPECTROGRAM_FRAMES, max(1, db.shape[1])), dtype=int)
        freq_indices = np.linspace(0, max(0, db.shape[0] - 1), min(MAX_SPECTROGRAM_BINS, max(1, db.shape[0])), dtype=int)
        figure = go.Figure(go.Heatmap(x=time[time_indices], y=freqs[freq_indices], z=db[np.ix_(freq_indices, time_indices)], colorscale=CUSTOM_COLORSCALE, zmin=-80, zmax=0, colorbar=dict(title="dB"), hovertemplate="Time %{x:.2f}s<br>Frequency %{y:.0f} Hz<br>Magnitude %{z:.1f} dB<extra></extra>"))
        layout = get_plotly_layout("Interactive Spectrogram", "Time (seconds)", "Frequency (Hz)")
        layout.update(yaxis=dict(layout["yaxis"], type="log", range=[math.log10(20), math.log10(max(20_000, float(freqs[-1]) if freqs.size else 20_000))]), height=460)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_chromagram(self, y: np.ndarray, sr: int, chords: Optional[Mapping[str, Any]] = None) -> go.Figure:
        """Plot chroma energy over time with pitch-class labels."""
        chroma = librosa.feature.chroma_stft(y=_as_mono(y), sr=sr, hop_length=DEFAULT_HOP_LENGTH, n_fft=STFT_SIZE)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=DEFAULT_HOP_LENGTH)
        indices = np.linspace(0, max(0, chroma.shape[1] - 1), min(MAX_SPECTROGRAM_FRAMES, max(1, chroma.shape[1])), dtype=int)
        figure = go.Figure(go.Heatmap(x=times[indices], y=list(NOTE_NAMES), z=chroma[:, indices], colorscale=CUSTOM_COLORSCALE, colorbar=dict(title="Energy"), hovertemplate="Time %{x:.2f}s<br>Note %{y}<br>Energy %{z:.3f}<extra></extra>"))
        annotations: List[Dict[str, Any]] = []
        if chords and "error" not in chords:
            for item in list(chords.get("chords", []))[:20]:
                annotations.append(dict(x=float(item.get("time", 0.0)), y=1.10, xref="x", yref="paper", text=html.escape(str(item.get("chord", ""))), showarrow=False, font=dict(size=9, color="#00e5a0")))
        layout = get_plotly_layout("Chromagram", "Time (seconds)", "Pitch class")
        layout.update(annotations=annotations, height=400)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_frequency_balance(self, spectrum: Mapping[str, Any]) -> go.Figure:
        """Plot band energy percentages and relative dB values."""
        bands = spectrum.get("frequency_bands", {}) if "error" not in spectrum else {}
        labels = [str(item.get("label", key)) for key, item in bands.items()]
        percentages = [_safe_float(item.get("percentage")) for item in bands.values()]
        values = [_safe_float(item.get("energy_db")) for item in bands.values()]
        colors = ["#ff6b6b", "#ffa500", "#00b4d8", "#7c5cfc", "#00e5a0", "#00e5a0", "#00b4d8"]
        figure = go.Figure(go.Bar(x=percentages, y=labels, orientation="h", marker_color=colors[: len(labels)], text=[f"{db:.1f} dB · {pct:.1f}%" for db, pct in zip(values, percentages)], textposition="auto", hovertemplate="%{y}<br>%{text}<extra></extra>"))
        ideal_percentage = 100.0 / max(len(labels), 1)
        layout = get_plotly_layout("Frequency Balance", "Share of total energy (%)", "Band")
        layout.update(height=400, yaxis=dict(layout["yaxis"], autorange="reversed"), shapes=[dict(type="line", x0=ideal_percentage, x1=ideal_percentage, y0=-0.5, y1=max(len(labels) - 0.5, 0.5), line=dict(color="rgba(255,255,255,0.45)", width=1, dash="dot"))])
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_beat_grid(self, tempo: Mapping[str, Any]) -> go.Figure:
        """Plot onset strength and detected beat markers."""
        onset = np.asarray(tempo.get("onset_strength", []), dtype=float) if "error" not in tempo else np.asarray([])
        times = np.asarray(tempo.get("onset_times", []), dtype=float) if "error" not in tempo else np.asarray([])
        beats = np.asarray(tempo.get("beat_times", []), dtype=float) if "error" not in tempo else np.asarray([])
        figure = go.Figure(go.Scatter(x=times, y=onset, mode="lines", line=dict(color="#00e5a0", width=1.5), fill="tozeroy", fillcolor="rgba(0,229,160,0.12)", name="Onset strength", hovertemplate="%{x:.2f}s<br>%{y:.3f}<extra></extra>"))
        shapes = [dict(type="line", x0=float(beat), x1=float(beat), y0=0, y1=float(np.max(onset, initial=1.0)), line=dict(color="rgba(124,92,252,0.5)", width=1)) for beat in beats[:300]]
        layout = get_plotly_layout("Onset Strength & Beat Grid", "Time (seconds)", "Onset strength")
        layout.update(shapes=shapes, height=360)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_melody(self, melody: Mapping[str, Any]) -> go.Figure:
        """Plot confidence-filtered melody pitch in note space."""
        contour = np.asarray(melody.get("pitch_contour", []), dtype=float) if "error" not in melody else np.asarray([])
        times = np.asarray(melody.get("pitch_times", []), dtype=float) if "error" not in melody else np.asarray([])
        confidence = np.asarray(melody.get("pitch_confidence", []), dtype=float) if "error" not in melody else np.asarray([])
        midi = _midi_from_hz(contour)
        valid = np.isfinite(midi)
        figure = go.Figure(go.Scatter(x=times[valid], y=midi[valid], mode="lines+markers", line=dict(color="#7c5cfc", width=2), marker=dict(size=5, color=confidence[valid] if confidence.size == midi.size else "#00e5a0", colorscale=[[0, "#4a1a8a"], [1, "#00e5a0"]], cmin=0, cmax=1), name="Melody", hovertemplate="%{x:.2f}s<br>%{y:.0f} MIDI<extra></extra>"))
        if valid.any():
            lower = int(math.floor(float(np.nanmin(midi[valid])) / 12.0) * 12)
            upper = int(math.ceil(float(np.nanmax(midi[valid])) / 12.0) * 12)
        else:
            lower, upper = 36, 84
        ticks = list(range(lower, upper + 1, 2))
        layout = get_plotly_layout("Dominant Melody Pitch", "Time (seconds)", "Note")
        layout.update(yaxis=dict(layout["yaxis"], tickmode="array", tickvals=ticks, ticktext=[_note_from_midi(tick) for tick in ticks]), height=360)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_bassline(self, bassline: Mapping[str, Any]) -> go.Figure:
        """Plot bass energy and detected bass pitch contour."""
        energy = np.asarray(bassline.get("bass_energy_over_time", []), dtype=float) if "error" not in bassline else np.asarray([])
        energy_times = np.asarray(bassline.get("bass_energy_times", []), dtype=float) if "error" not in bassline else np.asarray([])
        contour = np.asarray(bassline.get("bass_pitch_contour", []), dtype=float) if "error" not in bassline else np.asarray([])
        times = np.asarray(bassline.get("bass_times", []), dtype=float) if "error" not in bassline else np.asarray([])
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=energy_times, y=energy, mode="lines", line=dict(color="#00b4d8", width=1.5), fill="tozeroy", fillcolor="rgba(0,180,216,0.12)", name="Bass energy"))
        valid = np.isfinite(contour)
        if valid.any():
            figure.add_trace(go.Scatter(x=times[valid], y=contour[valid], mode="markers", marker=dict(color="#ffa500", size=5), name="Bass pitch", hovertemplate="%{x:.2f}s<br>%{y:.1f} Hz<extra></extra>"))
        layout = get_plotly_layout("Bassline Analysis", "Time (seconds)", "Energy / Frequency")
        layout.update(height=360)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_loudness_timeline(self, loudness: Mapping[str, Any]) -> go.Figure:
        """Plot short-term loudness with streaming target reference lines."""
        values = np.asarray(loudness.get("short_term_lufs", []), dtype=float) if "error" not in loudness else np.asarray([])
        times = np.asarray(loudness.get("loudness_times", []), dtype=float) if "error" not in loudness else np.asarray([])
        figure = go.Figure(go.Scatter(x=times, y=values, mode="lines", line=dict(color="#7c5cfc", width=2), fill="tozeroy", fillcolor="rgba(124,92,252,0.15)", name="Short-term LUFS", hovertemplate="%{x:.1f}s<br>%{y:.1f} LUFS<extra></extra>"))
        colors = {"spotify": "#00e5a0", "apple": "#ff6b6b", "youtube": "#ffa500"}
        shapes: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        for platform, target in PLATFORM_TARGETS.items():
            if platform not in colors:
                continue
            shapes.append(dict(type="line", x0=0, x1=max(float(times[-1]) if times.size else 1.0, 1.0), y0=target, y1=target, line=dict(color=colors[platform], width=1, dash="dot")))
            annotations.append(dict(x=0, y=target, text=platform.title(), xanchor="left", showarrow=False, font=dict(size=9, color=colors[platform])))
        layout = get_plotly_layout("Loudness Timeline", "Time (seconds)", "LUFS")
        layout.update(shapes=shapes, annotations=annotations, height=300)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_stereo_width(self, stereo: Mapping[str, Any]) -> go.Figure:
        """Plot stereo width over time and a healthy reference band."""
        values = np.asarray(stereo.get("width_over_time", []), dtype=float) if "error" not in stereo else np.asarray([])
        times = np.asarray(stereo.get("width_times", []), dtype=float) if "error" not in stereo else np.asarray([])
        if values.size == 0:
            values = np.asarray([_safe_float(stereo.get("stereo_width"))]) if "error" not in stereo else np.asarray([0.0])
            times = np.asarray([0.0])
        figure = go.Figure(go.Scatter(x=times, y=values, mode="lines", line=dict(color="#00e5a0", width=2), name="Stereo width", hovertemplate="%{x:.1f}s<br>%{y:.2f}<extra></extra>"))
        layout = get_plotly_layout("Stereo Width Over Time", "Time (seconds)", "Width")
        layout.update(yaxis=dict(layout["yaxis"], range=[0, 1]), shapes=[dict(type="rect", x0=0, x1=max(float(times[-1]), 1.0), y0=0.3, y1=0.7, fillcolor="rgba(0,229,160,0.08)", line_width=0)], height=300)
        figure.update_layout(**layout)
        return _finalize_figure(figure)

    def plot_spectrum(self, spectrum: Mapping[str, Any]) -> go.Figure:
        """Plot the smoothed full-frequency spectrum curve."""
        frequencies = np.asarray(spectrum.get("spectrum_frequencies", []), dtype=float) if "error" not in spectrum else np.asarray([])
        magnitudes = np.asarray(spectrum.get("spectrum_magnitudes", []), dtype=float) if "error" not in spectrum else np.asarray([])
        figure = go.Figure(go.Scatter(x=frequencies, y=magnitudes, mode="lines", line=dict(color="#7c5cfc", width=2), fill="tozeroy", fillcolor="rgba(124,92,252,0.12)", name="Spectrum", hovertemplate="%{x:.0f} Hz<br>%{y:.1f} dB<extra></extra>"))
        layout = get_plotly_layout("Full Spectrum Curve", "Frequency (Hz)", "Relative level (dB)")
        layout.update(xaxis=dict(layout["xaxis"], type="log", range=[math.log10(20), math.log10(20_000)]), height=380)
        figure.update_layout(**layout)
        return _finalize_figure(figure)


# ---------------------------------------------------------------------------
# Rule-based report generation
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate an actionable production report entirely from measured values."""

    def generate_report(self, all_analysis_results: Mapping[str, Any]) -> str:
        """Generate a Markdown production report from analysis dictionaries.

        Args:
            all_analysis_results: Results returned by :class:`AudioAnalyzer`.

        Returns:
            Markdown-formatted report.
        """
        metadata = all_analysis_results.get("metadata", {})
        tempo = all_analysis_results.get("tempo", {})
        key = all_analysis_results.get("key", {})
        chords = all_analysis_results.get("chords", {})
        structure = all_analysis_results.get("structure", {})
        melody = all_analysis_results.get("melody", {})
        loudness = all_analysis_results.get("loudness", {})
        spectrum = all_analysis_results.get("spectrum", {})
        stereo = all_analysis_results.get("stereo", {})
        bassline = all_analysis_results.get("bassline", {})

        duration = _safe_float(metadata.get("duration"))
        bpm = _safe_float(tempo.get("bpm"), 120.0)
        signature = str(tempo.get("time_signature", "4/4"))
        key_full = str(key.get("key_full", "Unknown key"))
        section_count = int(_safe_float(structure.get("num_segments"))) if "error" not in structure else 0
        lra = _safe_float(loudness.get("loudness_range_lra"))
        integrated = _safe_float(loudness.get("integrated_lufs"), LUFS_FLOOR)
        crest = _safe_float(loudness.get("crest_factor_db"))
        width = _safe_float(stereo.get("stereo_width"))
        phase = _safe_float(stereo.get("phase_correlation"))
        harmonic = _safe_float(chords.get("harmonic_complexity"))
        rhythm = _safe_float(tempo.get("syncopation_index"))
        groove = _safe_float(tempo.get("groove_consistency"))
        genre = self._genre_for_tempo(bpm, signature)
        harmony_rating = int(np.clip(round(1.0 + harmonic * 9.0), 1, 10))
        rhythm_rating = int(np.clip(round(1.0 + (rhythm * 0.55 + (1.0 - groove) * 0.45) * 9.0), 1, 10))
        flow = " → ".join(str(item.get("label", "Section")) for item in structure.get("segments", [])) if "error" not in structure else "Structure analysis unavailable"
        unique_chords = ", ".join(str(item) for item in chords.get("unique_chords", [])) if "error" not in chords else "Chord analysis unavailable"
        melodic_range = f"{melody.get('vocal_range_low', '—')} to {melody.get('vocal_range_high', '—')}" if "error" not in melody else "Melody analysis unavailable"
        freq_verdict = self._frequency_verdict(spectrum)
        dynamic_verdict = self._dynamic_verdict(lra)
        stereo_verdict = self._stereo_verdict(width)
        phase_verdict = self._phase_verdict(phase)
        recommendations = self._mix_recommendations(all_analysis_results)
        mastering = self._mastering_recommendations(all_analysis_results)
        references = self._reference_suggestions(bpm, signature)

        return f"""## 🎵 AudioIntelligence Pro — Production Report

### Track Overview

- **File:** `{metadata.get('filename', 'Unknown track')}`
- **Duration:** `{self._format_duration(duration)}` · **Sample rate:** `{metadata.get('sample_rate', '—')} Hz` · **Channels:** `{metadata.get('channel_count', '—')}`
- **Tempo:** `{bpm:.1f} BPM` · **Time signature:** `{signature}` · **Key:** `{key_full}`
- **Likely profile:** **{genre}**

### Arrangement & Structure

- **Sections detected:** `{section_count}`
- **Arrangement flow:** {flow}
- The arrangement reads as **{self._structure_complexity(section_count, duration)}**, based on section count, transitions, and duration.

### Harmonic Analysis

- Chord vocabulary: **{unique_chords}**
- Harmonic complexity: **{harmony_rating}/10** — {self._complexity_comment(harmonic, 'harmonic')}
- Key confidence: **{_clip01(key.get('confidence')):.0%}**. Extended/diminished chord usage is included in the complexity estimate.

### Rhythmic Profile

- The `{bpm:.1f} BPM` pulse suggests a **{genre}** rhythmic profile in `{signature}`.
- Syncopation index: **{rhythm:.2f}** ({self._level(rhythm)} off-beat activity).
- Groove consistency: **{groove:.2f}** ({'steady timing' if groove >= 0.75 else 'noticeable timing variation'}).
- Rhythmic complexity rating: **{rhythm_rating}/10**.

### Melodic Character

- Detected melodic range: **{melodic_range}**.
- Interval movement: **{self._interval_comment(melody)}**
- Bass root activity: **{len(bassline.get('bass_root_notes', [])) if 'error' not in bassline else '—'}** distinct sequential notes detected.

### Mix Quality Assessment

- Integrated loudness: **{integrated:.1f} LUFS**; true peak: **{_safe_float(loudness.get('true_peak_dbtp')):.1f} dBTP**.
- Loudness range: **{lra:.1f} LU** — **{dynamic_verdict}**.
- Frequency balance verdict: **{freq_verdict}**.
- Stereo width: **{width:.2f}** — **{stereo_verdict}**.
- Phase correlation: **{phase:.2f}** — **{phase_verdict}**.
- Crest factor: **{crest:.1f} dB**.

### 🔧 Mixing Recommendations

{chr(10).join(f'- {item}' for item in recommendations)}

### 🎛 Mastering Recommendations

{chr(10).join(f'- {item}' for item in mastering)}

### Reference Track Suggestions

{chr(10).join(f'- {item}' for item in references)}

> This report is generated locally from measurable DSP features. Always verify final decisions on calibrated monitors, headphones, and a mono playback check.
"""

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as minutes and seconds."""
        minutes, remainder = divmod(max(0, int(seconds)), 60)
        return f"{minutes}:{remainder:02d}"

    @staticmethod
    def _genre_for_tempo(bpm: float, signature: str) -> str:
        """Map tempo ranges to useful production language."""
        if bpm < 90:
            return "Slow / Ballad"
        if bpm < 120:
            return "Mid-tempo / Pop"
        if bpm < 140:
            return "Dance / House"
        if bpm < 170:
            return "Drum & Bass / Dubstep"
        return "Breakbeat / High-energy"

    @staticmethod
    def _structure_complexity(count: int, duration: float) -> str:
        """Describe arrangement density."""
        if count <= 2:
            return "minimal and focused"
        if count <= max(5, int(duration / 30.0)):
            return "clear and conventionally paced"
        return "highly sectional and arrangement-dense"

    @staticmethod
    def _complexity_comment(value: float, kind: str) -> str:
        """Describe a normalized complexity score."""
        if value < 0.25:
            return f"the {kind} vocabulary is restrained and repetitive"
        if value < 0.60:
            return f"the {kind} movement is moderate with useful variation"
        return f"the {kind} language is varied and deserves careful balance"

    @staticmethod
    def _level(value: float) -> str:
        """Describe a unit interval value."""
        return "low" if value < 0.33 else "moderate" if value < 0.66 else "high"

    @staticmethod
    def _interval_comment(melody: Mapping[str, Any]) -> str:
        """Summarize melodic interval behavior."""
        intervals = [abs(int(item)) for item in melody.get("melodic_intervals", []) if isinstance(item, (int, np.integer))]
        if not intervals:
            return "insufficient confident pitch frames for interval statistics"
        leaps = sum(1 for item in intervals if item >= 7)
        return "mostly stepwise" if leaps == 0 else f"{leaps} larger leap(s) with a mix of stepwise motion"

    @staticmethod
    def _dynamic_verdict(lra: float) -> str:
        """Interpret loudness range using production-oriented thresholds."""
        if lra < 4:
            return "heavily compressed"
        if lra < 8:
            return "modern pop/electronic dynamics"
        if lra < 12:
            return "well-balanced dynamics"
        return "very dynamic / cinematic"

    @staticmethod
    def _stereo_verdict(width: float) -> str:
        """Interpret normalized stereo width."""
        if width < 0.2:
            return "very narrow / mono"
        if width < 0.4:
            return "narrow"
        if width < 0.6:
            return "balanced"
        if width < 0.8:
            return "wide"
        return "very wide"

    @staticmethod
    def _phase_verdict(phase: float) -> str:
        """Interpret phase correlation."""
        if phase > 0.8:
            return "mono-compatible"
        if phase > 0.5:
            return "good compatibility"
        if phase >= 0:
            return "caution on mono playback"
        return "phase issues likely"

    @staticmethod
    def _frequency_verdict(spectrum: Mapping[str, Any]) -> str:
        """Compare measured spectral proportions against broad reference ranges."""
        bands = spectrum.get("frequency_bands", {}) if "error" not in spectrum else {}
        if not bands:
            return "unavailable"
        sub = _safe_float(bands.get("sub_bass", {}).get("percentage"))
        brilliance = _safe_float(bands.get("brilliance", {}).get("percentage"))
        mids = _safe_float(bands.get("mids", {}).get("percentage"))
        if sub > 30:
            return "low-end heavy; sub-bass is dominating the energy budget"
        if brilliance < 5:
            return "dark; upper-frequency air is relatively restrained"
        if mids > 40:
            return "mid-forward; inspect masking and vocal/instrument separation"
        return "broadly balanced across the measured bands"

    def _mix_recommendations(self, results: Mapping[str, Any]) -> List[str]:
        """Generate 3–7 targeted mixing recommendations."""
        loudness = results.get("loudness", {})
        spectrum = results.get("spectrum", {})
        stereo = results.get("stereo", {})
        tempo = results.get("tempo", {})
        recommendations: List[str] = []
        bands = spectrum.get("frequency_bands", {}) if "error" not in spectrum else {}
        if _safe_float(bands.get("sub_bass", {}).get("percentage")) > 30:
            recommendations.append("High-pass non-bass elements below roughly 80 Hz to reduce low-end mud and protect headroom.")
        if _safe_float(loudness.get("integrated_lufs"), LUFS_FLOOR) > -10:
            recommendations.append("The track is very loud; ease the limiter or ceiling to preserve transients and streaming flexibility.")
        if _safe_float(stereo.get("stereo_width")) < 0.2:
            recommendations.append("The mix feels narrow; try widening supporting synths and pads while keeping kick, bass, and lead centered.")
        if _safe_float(loudness.get("crest_factor_db")) < 4:
            recommendations.append("The low crest factor suggests heavy limiting; back off compression or limiting on the mix bus.")
        if _safe_float(bands.get("brilliance", {}).get("percentage")) < 5:
            recommendations.append("High-frequency content is restrained; audition a subtle high shelf or add air to individual elements.")
        if _safe_float(stereo.get("phase_correlation")) < 0:
            recommendations.append("Check polarity and stereo widening effects; negative correlation can collapse poorly to mono.")
        if _safe_float(tempo.get("syncopation_index")) > 0.7:
            recommendations.append("The groove is highly syncopated; use transient-aware sidechain and leave room around off-beat accents.")
        if not recommendations:
            recommendations.append("The measured balance is broadly healthy; make small, level-matched EQ moves rather than broad corrective boosts.")
        return recommendations[:7]

    def _mastering_recommendations(self, results: Mapping[str, Any]) -> List[str]:
        """Generate platform-aware mastering suggestions."""
        loudness = results.get("loudness", {})
        integrated = _safe_float(loudness.get("integrated_lufs"), LUFS_FLOOR)
        peak = _safe_float(loudness.get("true_peak_dbtp"), -1.0)
        suggestions = [f"For Spotify / YouTube, target about -14 LUFS integrated; apply approximately {(-14.0 - integrated):+.1f} dB of gain change.", f"For Apple Music, target about -16 LUFS integrated; apply approximately {(-16.0 - integrated):+.1f} dB of gain change."]
        if peak > -1.0:
            suggestions.append("Leave at least 1 dBTP of true-peak headroom and audition an inter-sample peak-safe limiter.")
        else:
            suggestions.append("True peak headroom is conservative; retain it during final limiting and codec auditioning.")
        suggestions.append("Compare the final master at matched loudness against a well-produced reference in the same tempo and genre family.")
        return suggestions

    @staticmethod
    def _reference_suggestions(bpm: float, signature: str) -> List[str]:
        """Offer descriptive reference-track directions without external metadata."""
        profile = "a restrained, vocal-forward ballad" if bpm < 90 else "a polished mid-tempo pop production" if bpm < 120 else "a modern dance/house master" if bpm < 140 else "a high-energy electronic reference"
        return [f"Use {profile} with a comparable {signature} pulse to match macro-dynamics.", "Choose one reference with similar vocal/instrument density and level-match before EQ decisions.", "Add a deliberately dynamic reference to avoid over-compressing the final master."]


# Module-level report helper for integrations and tests.
def generate_report(all_analysis_results: Mapping[str, Any]) -> str:
    """Generate a rule-based report using :class:`ReportGenerator`."""
    return ReportGenerator().generate_report(all_analysis_results)


# ---------------------------------------------------------------------------
# UI formatting and interaction logic
# ---------------------------------------------------------------------------


def _kpi(icon: str, label: str, value: str, sub: str, accent: str = "") -> str:
    """Render a KPI card body using the design-system classes."""
    accent_class = f" accent-{accent}" if accent else ""
    return f'<span class="kpi-icon">{icon}</span><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value{accent_class}">{html.escape(value)}</div><div class="kpi-sub">{html.escape(sub)}</div>'


def _structure_timeline(structure: Mapping[str, Any]) -> str:
    """Render a proportional HTML section timeline."""
    if "error" in structure:
        return f'<div class="glass-card" style="color:#ff6b6b;">Analysis unavailable: {html.escape(str(structure["error"]))}</div>'
    segments = list(structure.get("segments", []))
    if not segments:
        return '<div class="glass-card">No structural segments detected.</div>'
    colors = {"Intro": "#00b4d8", "Verse": "#00e5a0", "Chorus": "#7c5cfc", "Bridge": "#ffa500", "Outro": "#4a5568"}
    total = max(sum(max(0.01, float(item.get("end", 0)) - float(item.get("start", 0))) for item in segments), 0.01)
    blocks = []
    for item in segments:
        label = str(item.get("label", "Section"))
        key = next((key for key in colors if label.startswith(key)), "Outro")
        width = max(2.0, (float(item.get("end", 0)) - float(item.get("start", 0))) / total * 100.0)
        blocks.append(f'<div style="width:{width:.2f}%;background:{colors[key]};padding:10px 4px;text-align:center;color:#06060a;font-size:11px;font-weight:700;overflow:hidden;white-space:nowrap;" title="{html.escape(label)}">{html.escape(label)}</div>')
    return '<div class="glass-card"><div class="section-title"><span class="icon">🧭</span>Structure Timeline</div><div style="display:flex;border-radius:8px;overflow:hidden;min-height:36px;">' + "".join(blocks) + '</div></div>'


def _loudness_meter(loudness: Mapping[str, Any]) -> str:
    """Render the LUFS gauge and platform target markers."""
    if "error" in loudness:
        return f'<div class="meter-gauge" style="color:#ff6b6b;">Analysis unavailable: {html.escape(str(loudness["error"]))}</div>'
    value = _safe_float(loudness.get("integrated_lufs"), LUFS_FLOOR)
    # Map -30..0 LUFS into a readable meter range.
    percent = float(np.clip((value + 30.0) / 30.0 * 100.0, 0.0, 100.0))
    meter_class = "optimal" if -18.0 <= value <= -11.0 else "warning" if value <= -8.0 else "danger"
    return f'''<div class="meter-gauge"><div style="display:flex;justify-content:space-between;align-items:baseline;"><strong style="font-size:28px;color:#f0f0f5;">{value:.1f} LUFS</strong><span style="color:var(--text-secondary);font-size:12px;">Integrated</span></div><div class="meter-track" style="margin-top:18px;"><div class="meter-fill {meter_class}" style="width:{percent:.1f}%;"></div></div><div class="meter-markers"><span>-30</span><span>-24</span><span>-18</span><span>-14</span><span>-9</span><span>0 LUFS</span></div></div>'''


def _loudness_table(loudness: Mapping[str, Any]) -> str:
    """Render dynamics metrics and platform deltas as a styled table."""
    if "error" in loudness:
        return f'<div class="glass-card" style="color:#ff6b6b;">Analysis unavailable: {html.escape(str(loudness["error"]))}</div>'
    rows = [("Integrated LUFS", f"{_safe_float(loudness.get('integrated_lufs')):.1f}"), ("Short-term LUFS", f"{_safe_float(np.mean(loudness.get('short_term_lufs', [0]))):.1f}"), ("Momentary LUFS", f"{_safe_float(np.mean(loudness.get('momentary_lufs', [0]))):.1f}"), ("True Peak", f"{_safe_float(loudness.get('true_peak_dbtp')):.1f} dBTP"), ("RMS", f"{_safe_float(loudness.get('rms_db')):.1f} dB"), ("Crest Factor", f"{_safe_float(loudness.get('crest_factor_db')):.1f} dB"), ("Loudness Range", f"{_safe_float(loudness.get('loudness_range_lra')):.1f} LU"), ("Dynamic Range", f"{_safe_float(loudness.get('dynamic_range_db')):.1f} dB")]
    body = "".join(f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(value)}</code></td></tr>" for name, value in rows)
    return f'<table class="data-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{body}</tbody></table>'


def _stereo_data(stereo: Mapping[str, Any]) -> str:
    """Render stereo metrics as a styled table."""
    if "error" in stereo:
        return f'<div style="color:#ff6b6b;">Analysis unavailable: {html.escape(str(stereo["error"]))}</div>'
    rows = [("Stereo Width", f"{_safe_float(stereo.get('stereo_width')):.2f}"), ("Phase Correlation", f"{_safe_float(stereo.get('phase_correlation')):+.2f}"), ("Mid / Side Energy", f"{_safe_float(stereo.get('mid_side_ratio')):.2f}"), ("Stereo Balance", f"{_safe_float(stereo.get('balance_lr')):+.2f} (L − R)")]
    return '<table class="data-table"><tbody>' + "".join(f'<tr><td>{name}</td><td><code>{value}</code></td></tr>' for name, value in rows) + "</tbody></table>"


def _error_banner(message: str) -> str:
    """Render the prescribed red warning banner."""
    return f'<div class="glass-card" style="border-color: rgba(255,107,107,0.3);"><div style="color:#ff6b6b;font-weight:600;">⚠ Analysis Error</div><div style="color:var(--text-secondary);margin-top:8px;">{html.escape(message)}</div></div>'


def sync_audio_preview(file_path: Optional[str]) -> Optional[str]:
    """Pass a selected File filepath to the read-only cross-platform player."""
    if not file_path:
        return None
    if isinstance(file_path, (list, tuple)):
        return str(file_path[0]) if file_path else None
    return str(file_path)


def _module_errors(results: Mapping[str, Any]) -> List[str]:
    """Collect non-fatal analysis module errors."""
    errors = []
    for name in ("tempo", "key", "chords", "structure", "melody", "bassline", "loudness", "spectrum", "stereo"):
        value = results.get(name, {})
        if isinstance(value, Mapping) and "error" in value:
            errors.append(f"{name}: {value['error']}")
    return errors


def _render_results(results: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Build all final Gradio output values from an analysis result."""
    data = results.get("_audio")
    if not isinstance(data, AudioData):
        raise ValueError("Audio data is missing from the analysis result.")
    tempo = results.get("tempo", {})
    key = results.get("key", {})
    loudness = results.get("loudness", {})
    confidence = _clip01(tempo.get("bpm_confidence"))
    key_confidence = _clip01(key.get("confidence"))
    bpm_value = f"{_safe_float(tempo.get('bpm'), 0.0):.1f}" if "error" not in tempo else "—"
    key_value = str(key.get("key_full", "—")) if "error" not in key else "—"
    time_value = str(tempo.get("time_signature", "—")) if "error" not in tempo else "—"
    lufs_value = f"{_safe_float(loudness.get('integrated_lufs'), LUFS_FLOOR):.1f}" if "error" not in loudness else "—"
    bpm_sub = f"±{confidence:.0%} confidence" if "error" not in tempo else "Analysis unavailable"
    key_sub = f"{key_confidence:.0%} confidence" if "error" not in key else "Analysis unavailable"
    time_sub = f"{_clip01(tempo.get('time_signature_confidence')):.0%} confidence" if "error" not in tempo else "Analysis unavailable"
    lufs_sub = f"LRA {_safe_float(loudness.get('loudness_range_lra')):.1f} LU" if "error" not in loudness else "Analysis unavailable"
    waveform = Visualizer().plot_waveform_structure(data.analysis_y, data.analysis_sr, results.get("structure", {}), np.asarray(tempo.get("beat_times", [])))
    visualizer = Visualizer()
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        '<div class="progress-container"><div class="progress-bar" style="width:100%"></div></div>',
        '<div class="progress-step"><span>✓</span><span>Analysis complete.</span></div>',
        _kpi("🎵", "Tempo", bpm_value, bpm_sub, "purple"),
        _kpi("🎹", "Musical Key", key_value, key_sub, "green"),
        _kpi("⏱", "Time Signature", time_value, time_sub, "blue"),
        _kpi("📊", "Loudness", lufs_value, lufs_sub),
        waveform,
        visualizer.plot_chord_progression(results.get("chords", {})),
        _structure_timeline(results.get("structure", {})),
        visualizer.plot_spectrogram(data.analysis_y, data.analysis_sr),
        visualizer.plot_chromagram(data.analysis_y, data.analysis_sr, results.get("chords", {})),
        visualizer.plot_frequency_balance(results.get("spectrum", {})),
        visualizer.plot_beat_grid(tempo),
        visualizer.plot_melody(results.get("melody", {})),
        visualizer.plot_bassline(results.get("bassline", {})),
        _loudness_meter(loudness),
        _loudness_table(loudness),
        visualizer.plot_loudness_timeline(loudness),
        _stereo_data(results.get("stereo", {})),
        visualizer.plot_stereo_width(results.get("stereo", {})),
        visualizer.plot_spectrum(results.get("spectrum", {})),
        str(results.get("report", "Report unavailable.")),
        gr.update(visible=False),
    )


def _progress_values(step: int, message: str) -> Tuple[Any, ...]:
    """Create an intermediate UI update while analysis is running."""
    percentage = min(100, int(round(step / len(ANALYSIS_STEPS) * 100)))
    bar = f'<div class="progress-container"><div class="progress-bar" style="width:{percentage}%"></div></div>'
    text = f'<div class="progress-step"><div class="spinner"></div><span>Step {step}/{len(ANALYSIS_STEPS)}: {html.escape(message)}</span></div>'
    return (
        gr.update(visible=True), gr.update(visible=False), bar, text,
        _kpi("🎵", "Tempo", "—", "Analyzing...", "purple"),
        _kpi("🎹", "Musical Key", "—", "Analyzing...", "green"),
        _kpi("⏱", "Time Signature", "—", "Analyzing...", "blue"),
        _kpi("📊", "Loudness", "—", "Analyzing..."),
        None, None, "", None, None, None, None, None, None,
        "", "", None, "", None, None, "", gr.update(visible=False),
    )


def _failure_values(message: str) -> Tuple[Any, ...]:
    """Create an error-state UI update while keeping the app usable."""
    values = list(_progress_values(len(ANALYSIS_STEPS), "Analysis stopped."))
    values[0] = gr.update(visible=False)
    values[1] = gr.update(visible=True)
    values[3] = f'<div class="progress-step" style="color:#ff6b6b;"><span>⚠</span><span>{html.escape(message)}</span></div>'
    values[24] = gr.update(value=_error_banner(message), visible=True)
    return tuple(values)


def analyze_track(file_path: Optional[str], progress: Any = gr.Progress(track_tqdm=False)) -> Iterator[Tuple[Any, ...]]:
    """Gradio generator callback with eight visible progress stages."""
    if progress is not None:
        try:
            progress(0.0, desc="Preparing analysis...")
        except Exception:
            pass
    yield _progress_values(0, "Initializing...")
    analyzer = AudioAnalyzer()
    try:
        final_results: Dict[str, Any] = {}
        for step, message, current in analyzer.iter_analyze(str(file_path or "")):
            final_results = current
            if progress is not None:
                try:
                    progress(step / len(ANALYSIS_STEPS), desc=f"Step {step}/{len(ANALYSIS_STEPS)}: {message}")
                except Exception:
                    pass
            if step < len(ANALYSIS_STEPS):
                yield _progress_values(step, message)
        output_values = list(_render_results(final_results))
        errors = _module_errors(final_results)
        if errors:
            output_values[24] = gr.update(value=_error_banner("Some optional modules were unavailable: " + " | ".join(errors)), visible=True)
        yield tuple(output_values)
    except Exception as exc:
        LOGGER.error("Critical analysis failure: %s", exc, exc_info=True)
        yield _failure_values(str(exc))


def download_report(report_text: Optional[str]) -> Optional[str]:
    """Create a lightweight PDF report using the installed Matplotlib backend."""
    if not report_text:
        return None
    path = Path(tempfile.gettempdir()) / "audiointelligence_pro_report.pdf"
    try:
        from matplotlib.backends.backend_pdf import PdfPages

        plain_lines = re.sub(r"[*#`]", "", str(report_text)).splitlines()
        with PdfPages(path) as pdf:
            page_lines: List[str] = []
            for line in plain_lines:
                wrapped = [line[index : index + 105] for index in range(0, max(len(line), 1), 105)] or [""]
                page_lines.extend(wrapped)
            page_size = 48
            for start in range(0, max(len(page_lines), 1), page_size):
                figure = plt.figure(figsize=(8.27, 11.69), facecolor="#06060a")
                axis = figure.add_axes([0.08, 0.06, 0.84, 0.88])
                axis.set_facecolor("#06060a")
                axis.axis("off")
                axis.text(0.0, 1.0, "\n".join(page_lines[start : start + page_size]), va="top", ha="left", color="#f0f0f5", family="monospace", fontsize=9, linespacing=1.35)
                pdf.savefig(figure, facecolor=figure.get_facecolor())
                plt.close(figure)
        return str(path)
    except Exception as exc:
        LOGGER.error("PDF report generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Gradio Blocks application
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    """Build the complete AudioIntelligence Pro Gradio application."""
    try:
        theme = gr.themes.Base(primary_hue="violet", secondary_hue="emerald", neutral_hue="slate", font=gr.themes.GoogleFont("Inter"), font_mono=gr.themes.GoogleFont("JetBrains Mono"))
    except Exception:
        theme = gr.themes.Base(primary_hue="violet", secondary_hue="emerald", neutral_hue="slate")

    with gr.Blocks(css=CUSTOM_CSS, theme=theme, title=APP_NAME, analytics_enabled=False) as app:
        gr.HTML('<div class="aurora-bg"></div>')

        with gr.Row(elem_classes="app-header"):
            with gr.Column(scale=3):
                gr.HTML("<h1>AudioIntelligence Pro</h1><p class=\"tagline\">Deep musicological &amp; audio engineering analysis powered by advanced DSP</p>")
            with gr.Column(scale=1):
                gr.HTML('<div style="text-align:right;padding-top:8px;"><span class="version-badge">v2.0 Live</span></div>')

        with gr.Column(elem_classes="glass-card", elem_id="upload-section"):
            gr.HTML('<div class="upload-icon">🎵</div><div class="upload-text">Tap or click to choose your track</div><div class="upload-hint">The system file picker works on Windows, macOS, Linux, Android, and iOS</div>')
            # A dedicated File component is more reliable than the combined
            # Audio uploader on touch browsers, Safari, and mobile WebViews.
            # It still returns a normal local filepath for the DSP pipeline.
            audio_input = gr.File(label="Choose audio file", show_label=True, file_count="single", file_types=sorted(SUPPORTED_EXTENSIONS), type="filepath", elem_classes="upload-zone", elem_id="audio-file-upload", height=160)
            audio_preview = gr.Audio(label="Selected track", type="filepath", sources=[], interactive=False, elem_classes="audio-preview")
            audio_input.change(fn=sync_audio_preview, inputs=[audio_input], outputs=[audio_preview], show_progress="hidden")
            gr.HTML('<div class="format-badges"><span class="format-badge">MP3</span><span class="format-badge">WAV</span><span class="format-badge">FLAC</span><span class="format-badge">OGG</span><span class="format-badge">M4A</span></div><div class="upload-hint">Max 50MB · Native sample rate preserved for stereo analysis</div>')
            analyze_btn = gr.Button("⚡  ANALYZE TRACK", variant="primary", size="lg", elem_classes="analyze-btn")
            progress_area = gr.Column(visible=False)
            with progress_area:
                progress_bar = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                progress_text = gr.HTML('<div class="progress-step"><div class="spinner"></div><span>Initializing...</span></div>')

        results_dashboard = gr.Column(visible=False)
        with results_dashboard:
            with gr.Row(elem_classes="kpi-grid"):
                with gr.Column(elem_classes="kpi-card animate-in"):
                    kpi_bpm = gr.HTML(_kpi("🎵", "Tempo", "—", "BPM", "purple"))
                with gr.Column(elem_classes="kpi-card animate-in"):
                    kpi_key = gr.HTML(_kpi("🎹", "Musical Key", "—", "Confidence", "green"))
                with gr.Column(elem_classes="kpi-card animate-in"):
                    kpi_time = gr.HTML(_kpi("⏱", "Time Signature", "—", "Detected", "blue"))
                with gr.Column(elem_classes="kpi-card animate-in"):
                    kpi_lufs = gr.HTML(_kpi("📊", "Loudness", "—", "LUFS Integrated"))

            with gr.Tabs(elem_classes="results-tabs"):
                with gr.Tab("📋  Overview", elem_id="tab-overview"):
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Waveform &amp; Structure</div>')
                            waveform_plot = gr.Plot(elem_classes="chart-plot")
                        with gr.Column(scale=1, elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Chord Progression</div>')
                            chord_plot = gr.Plot(elem_classes="chart-plot")
                    structure_timeline = gr.HTML(elem_classes="glass-card")

                with gr.Tab("🌈  Spectral Analysis", elem_id="tab-spectral"):
                    with gr.Row():
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Spectrogram</div>')
                            spectrogram_plot = gr.Plot()
                    with gr.Row():
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Chromagram</div>')
                            chromagram_plot = gr.Plot()
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Frequency Balance</div>')
                            freq_balance_plot = gr.Plot()

                with gr.Tab("🥁  Rhythm &amp; Melody", elem_id="tab-rhythm"):
                    with gr.Row():
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Onset Strength &amp; Beat Grid</div>')
                            beat_plot = gr.Plot()
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Melody Pitch Contour</div>')
                            melody_plot = gr.Plot()
                    with gr.Row():
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Bassline Analysis</div>')
                            bass_plot = gr.Plot()

                with gr.Tab("🎛  Mix &amp; Master", elem_id="tab-mixing"):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes="glass-card"):
                            gr.HTML('<div class="section-title"><span class="icon">📏</span> Loudness Meter</div>')
                            loudness_meter = gr.HTML()
                            loudness_table = gr.HTML()
                            loudness_plot = gr.Plot()
                        with gr.Column(scale=1, elem_classes="glass-card"):
                            gr.HTML('<div class="section-title"><span class="icon">🔊</span> Stereo Field</div>')
                            stereo_data = gr.HTML()
                            stereo_plot = gr.Plot()
                    with gr.Row():
                        with gr.Column(elem_classes="chart-container"):
                            gr.HTML('<div class="chart-title">Full Spectrum Curve</div>')
                            spectrum_plot = gr.Plot()

                with gr.Tab("🤖  AI Report", elem_id="tab-report"):
                    with gr.Column(elem_classes="report-content"):
                        report_output = gr.Markdown("Run analysis to generate your production report.")
                    with gr.Row():
                        download_btn = gr.DownloadButton("📥  Download Report as PDF", variant="secondary", size="sm")

            error_banner = gr.HTML(visible=False)

        gr.HTML('<div style="text-align:center;padding:32px 0 24px;color:var(--text-tertiary);font-size:12px;border-top:1px solid var(--border-subtle);margin-top:48px;">Built with AudioIntelligence Pro v2.0 &nbsp;·&nbsp; Powered by Librosa + Plotly + Gradio &nbsp;·&nbsp; All analysis runs locally</div>')

        analysis_outputs = [progress_area, results_dashboard, progress_bar, progress_text, kpi_bpm, kpi_key, kpi_time, kpi_lufs, waveform_plot, chord_plot, structure_timeline, spectrogram_plot, chromagram_plot, freq_balance_plot, beat_plot, melody_plot, bass_plot, loudness_meter, loudness_table, loudness_plot, stereo_data, stereo_plot, spectrum_plot, report_output, error_banner]
        analyze_btn.click(fn=analyze_track, inputs=[audio_input], outputs=analysis_outputs, show_progress="full")
        download_btn.click(fn=download_report, inputs=[report_output], outputs=[download_btn])

    return app


if __name__ == "__main__":
    build_ui().launch(share=False, server_name="0.0.0.0")
