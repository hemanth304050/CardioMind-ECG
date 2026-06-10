import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os
import time

# Import custom modules
from simulator import ECGSimulator
from preprocessing import ECGPreprocessor
from model import CardioMindECG
from explainability import compute_gradcam_1d, generate_clinical_report
from tinyml import TinyMLProfiler

# Set Page Config
st.set_page_config(
    page_title="CardioMind-ECG: Wearable Cardiac Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection for Rich Aesthetics)
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Neon Glow Titles */
    .title-text {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-family: 'Outfit', sans-serif;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphic Containers */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f2fe !important;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 65, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    /* Neon Borders on Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 2px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #8b949e;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom: 2px solid #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run fast synthetic model training
def train_dummy_model(device):
    from main import train_model_pipeline
    with st.spinner("Initializing Model... Generating physiological synthetic datasets and training CardioMind-ECG network (approx. 8s)..."):
        train_model_pipeline(epochs=2, num_sequences=35, device=device)
    st.success("CardioMind-ECG successfully initialized! Model weights saved to 'cardio_mind_model.pth'.")
    st.rerun()

# ----------------- App Initialization -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "cardio_mind_model.pth"

# App Header
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown('<div class="title-text">CardioMind-ECG 🩺</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Deep Temporal Attention Networks for Real-Time Stress, Arrhythmia, and Emotion Recognition from Wearable ECG</div>', unsafe_allow_html=True)
with col2:
    # Status Indicators
    if os.path.exists(model_path):
        st.success("Model Status: READY")
    else:
        st.warning("Model Status: UNINITIALIZED")
        if st.button("🚀 Initialize Model"):
            train_dummy_model(device)

# Load Model
@st.cache_resource
def load_trained_model(path, device):
    if not os.path.exists(path):
        return None
    model = CardioMindECG()
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

model = load_trained_model(model_path, device)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Signal Settings")
scenario = st.sidebar.selectbox(
    "ECG Scenario",
    ["Normal Sinus Rhythm", "Atrial Fibrillation", "Premature Ventricular Contractions", "Ventricular Tachycardia", "Stress (High)", "Stress (Low)", "Emotion (Happy)", "Emotion (Sad)"],
    index=0
)

# Convert scenario to key
scenario_keys = {
    "Normal Sinus Rhythm": "normal",
    "Atrial Fibrillation": "arrhythmia_af",
    "Premature Ventricular Contractions": "arrhythmia_pvc",
    "Ventricular Tachycardia": "arrhythmia_vt",
    "Stress (High)": "stress_high",
    "Stress (Low)": "stress_low",
    "Emotion (Happy)": "emotion_happy",
    "Emotion (Sad)": "emotion_sad"
}
scenario_key = scenario_keys[scenario]

noise_level = st.sidebar.slider("Motion & EMG Noise", 0.0, 0.4, 0.08, step=0.01)
duration = st.sidebar.slider("Recording Duration (s)", 10, 60, 45, step=5)

# Simulate ECG Signal
fs = 256
simulator = ECGSimulator(fs=fs)
preprocessor = ECGPreprocessor(fs=fs)

# Run Simulation and Preprocess
@st.cache_data(show_spinner="Simulating ECG Waveform...")
def run_simulation(s_key, dur, noise):
    sim_data = simulator.generate_scenario(s_key, duration_seconds=dur, noise_level=noise)
    # Preprocess
    filtered = preprocessor.denoise(sim_data["raw"])
    peaks = preprocessor.detect_r_peaks(filtered)
    beats, valid_peaks = preprocessor.segment_beats(filtered, peaks)
    hrv = preprocessor.extract_hrv_features(peaks)
    
    return sim_data, filtered, peaks, beats, valid_peaks, hrv

sim_data, filtered_signal, r_peaks, beats, valid_peaks, hrv_metrics = run_simulation(scenario_key, duration, noise_level)

# ----------------- Dashboard Layout & Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Real-Time Signal Monitor", 
    "🎯 Diagnostics & Multi-Task Predictor", 
    "👁️ Explainability Explorer (XAI)", 
    "⚡ TinyML Edge Profiler"
])

# ------------- TAB 1: Real-Time Signal Monitor -------------
with tab1:
    st.markdown("### Continuous Signal Stream & Preprocessing")
    
    # Live signal plot (display first 10 seconds for detail, or slider)
    display_sec = st.slider("Signal Display Window (seconds)", 2, min(duration, 15), 8)
    n_display_samples = int(display_sec * fs)
    
    # Raw vs Filtered plotting
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        x=sim_data["time"][:n_display_samples],
        y=sim_data["raw"][:n_display_samples],
        mode='lines',
        name='Raw Waveform (with artifacts)',
        line=dict(color='#ff6b6b', width=1.2)
    ))
    fig_signal.add_trace(go.Scatter(
        x=sim_data["time"][:n_display_samples],
        y=filtered_signal[:n_display_samples],
        mode='lines',
        name='Filtered Waveform (Butterworth)',
        line=dict(color='#00f2fe', width=1.8)
    ))
    
    # Add R-peaks markers
    display_peaks = [p for p in r_peaks if p < n_display_samples]
    if display_peaks:
        fig_signal.add_trace(go.Scatter(
            x=sim_data["time"][display_peaks],
            y=filtered_signal[display_peaks],
            mode='markers',
            name='Detected R-Peaks (Pan-Tompkins)',
            marker=dict(color='#ff007f', size=10, symbol='triangle-up', line=dict(width=1, color='white'))
        ))
        
    fig_signal.update_layout(
        plot_bgcolor='rgba(13, 17, 23, 0.9)',
        paper_bgcolor='rgba(13, 17, 23, 0.9)',
        xaxis=dict(title="Time (seconds)", gridcolor='#21262d'),
        yaxis=dict(title="Amplitude (mV)", gridcolor='#21262d'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=10, b=20),
        height=380
    )
    st.plotly_chart(fig_signal, use_container_width=True)
    
    # HRV Metrics Row
    st.markdown("### Heart Rate Variability (HRV) Diagnostics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.metric("Mean Heart Rate", f"{hrv_metrics['mean_hr']:.1f} BPM", delta=f"{hrv_metrics['mean_hr'] - 72.0:+.1f} vs standard")
    with m_col2:
        st.metric("RMSSD (Vagal Index)", f"{hrv_metrics['rmssd']:.2f} ms", help="Root Mean Square of Successive Differences. Reflects vagal regulation.")
    with m_col3:
        st.metric("SDNN (Overall HRV)", f"{hrv_metrics['sdnn']:.2f} ms", help="Standard deviation of NN intervals. High values indicate healthy variability.")
    with m_col4:
        st.metric("LF/HF Ratio", f"{hrv_metrics['lf_hf_ratio']:.2f}", help="Sympathovagal balance. Elevated ratio suggests stress.")
    with m_col5:
        st.metric("Sample Entropy", f"{hrv_metrics['sample_entropy']:.2f}", help="Complexity of interval sequence. Stress decreases entropy.")
        
    # Tachogram and Poincaré Plot
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### RR Interval Tachogram")
        rr_intervals = np.diff(r_peaks) / fs * 1000.0
        fig_tach = go.Figure()
        fig_tach.add_trace(go.Scatter(
            x=np.arange(len(rr_intervals)),
            y=rr_intervals,
            mode='lines+markers',
            line=dict(color='#4facfe', width=2),
            marker=dict(size=4),
            name='R-R Interval'
        ))
        fig_tach.update_layout(
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(title='Beat Count', gridcolor='#21262d'),
            yaxis=dict(title='R-R Interval (ms)', gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_tach, use_container_width=True)
        
    with c2:
        st.markdown("#### Autonomic Poincaré Plot")
        if len(rr_intervals) > 2:
            x_poinc = rr_intervals[:-1]
            y_poinc = rr_intervals[1:]
            
            fig_poinc = go.Figure()
            fig_poinc.add_trace(go.Scatter(
                x=x_poinc,
                y=y_poinc,
                mode='markers',
                marker=dict(color='#00f2fe', size=7, opacity=0.8),
                name='RR[n] vs RR[n+1]'
            ))
            
            # Draw standard SD1, SD2 axis orientation center
            center_x = np.mean(x_poinc)
            center_y = np.mean(y_poinc)
            
            fig_poinc.add_trace(go.Scatter(
                x=[center_x - hrv_metrics['sd2']/np.sqrt(2), center_x + hrv_metrics['sd2']/np.sqrt(2)],
                y=[center_y - hrv_metrics['sd2']/np.sqrt(2), center_y + hrv_metrics['sd2']/np.sqrt(2)],
                mode='lines',
                line=dict(color='#ff007f', width=2, dash='dash'),
                name='SD2 (Long-term)'
            ))
            
            fig_poinc.add_trace(go.Scatter(
                x=[center_x - hrv_metrics['sd1']/np.sqrt(2), center_x + hrv_metrics['sd1']/np.sqrt(2)],
                y=[center_y + hrv_metrics['sd1']/np.sqrt(2), center_y - hrv_metrics['sd1']/np.sqrt(2)],
                mode='lines',
                line=dict(color='#fffa65', width=2, dash='dash'),
                name='SD1 (Short-term)'
            ))
            
            fig_poinc.update_layout(
                plot_bgcolor='rgba(13, 17, 23, 0.9)',
                paper_bgcolor='rgba(13, 17, 23, 0.9)',
                xaxis=dict(title="RR(n) (ms)", gridcolor='#21262d'),
                yaxis=dict(title="RR(n+1) (ms)", gridcolor='#21262d'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280
            )
            st.plotly_chart(fig_poinc, use_container_width=True)

# ------------- TAB 2: Diagnostics & Multi-Task Predictor -------------
with tab2:
    if model is None:
        st.info("💡 Please initialize the CardioMind-ECG model using the button in the upper right to run diagnostics.")
    else:
        st.markdown("### Multi-Task Neural Network Diagnostics")
        
        # Prepare inputs
        seq_len = 60
        if len(beats) == 0:
            padded_beats = np.zeros((seq_len, 150))
        elif beats.ndim == 1:
            padded_beats = np.zeros((seq_len, 150))
        elif len(beats) < seq_len:
            # Pad with repeated beats if signal is too short
            padded_beats = np.pad(beats, ((0, seq_len - len(beats)), (0, 0)), 'edge')
        else:
            padded_beats = beats[:seq_len]
            
        beat_seq_tensor = torch.FloatTensor(padded_beats).unsqueeze(0).to(device) # [1, 60, 150]
        
        hrv_list = [
            hrv_metrics["mean_hr"], hrv_metrics["sdnn"], hrv_metrics["rmssd"], hrv_metrics["pnn50"],
            hrv_metrics["lf"], hrv_metrics["hf"], hrv_metrics["lf_hf_ratio"],
            hrv_metrics["sd1"], hrv_metrics["sd2"], hrv_metrics["sd1_sd2_ratio"], hrv_metrics["sample_entropy"]
        ]
        hrv_tensor = torch.FloatTensor([hrv_list]).to(device) # [1, 11]
        
        # Inference
        with torch.no_grad():
            preds, attn_weights = model(beat_seq_tensor, hrv_tensor)
            
        # Softmax predictions
        arr_probs = torch.softmax(preds["arrhythmia"], dim=1)[0].cpu().numpy()
        stress_probs = torch.softmax(preds["stress"], dim=1)[0].cpu().numpy()
        valence_probs = torch.softmax(preds["valence"], dim=1)[0].cpu().numpy()
        arousal_probs = torch.softmax(preds["arousal"], dim=1)[0].cpu().numpy()
        
        col_dia1, col_dia2 = st.columns([0.6, 0.4])
        
        with col_dia1:
            st.markdown("#### Arrhythmia Classification Probability")
            arr_classes = ["Normal (NSR)", "Atrial Fibrillation (AF)", "Premature Ventricular (PVC)", "Ventricular Tachycardia (VT)", "Bradycardia"]
            
            # Select colors based on class risk
            colors = ['#00e676', '#ff9100', '#ff3d00', '#d50000', '#2979ff']
            
            fig_arr = go.Figure(go.Bar(
                x=arr_probs * 100,
                y=arr_classes,
                orientation='h',
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in arr_probs],
                textposition='auto'
            ))
            fig_arr.update_layout(
                plot_bgcolor='rgba(13, 17, 23, 0.9)',
                paper_bgcolor='rgba(13, 17, 23, 0.9)',
                xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_arr, use_container_width=True)
            
            st.markdown("#### Stress & Cognitive Workload Assessment")
            stress_labels = ["Low (Relaxed)", "Medium (Baseline)", "High Stress"]
            stress_colors = ["#00e676", "#ffb300", "#ff3d00"]
            stress_idx = np.argmax(stress_probs)
            
            # Create subcolumns for stress progress bars
            for s_idx, (label, prob) in enumerate(zip(stress_labels, stress_probs)):
                prog_col, val_col = st.columns([0.85, 0.15])
                with prog_col:
                    st.write(label)
                    st.progress(float(prob))
                with val_col:
                    st.write(f"\n{prob*100:.1f}%")
                    
        with col_dia2:
            st.markdown("#### Emotional State Mapping (Valence & Arousal)")
            # Map probabilities to a 2D coordinate system
            # Valence: Positive/Neutral is x > 0, Negative is x < 0.
            # Arousal: High is y > 0, Low is y < 0.
            # We map 2-class softmax into scale [-1, 1]
            val_x = float(valence_probs[0] - valence_probs[1]) # Pos - Neg
            ar_y = float(arousal_probs[0] - arousal_probs[1])   # High - Low
            
            fig_emo = go.Figure()
            # Draw Quadrants
            fig_emo.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1, fillcolor="rgba(255, 61, 0, 0.05)", line=dict(width=0)) # Angry/Anxious
            fig_emo.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor="rgba(0, 230, 118, 0.05)", line=dict(width=0)) # Happy/Excited
            fig_emo.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0, fillcolor="rgba(41, 121, 255, 0.05)", line=dict(width=0)) # Depressed/Bored
            fig_emo.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0, fillcolor="rgba(156, 39, 176, 0.05)", line=dict(width=0)) # Calm/Relaxed
            
            # Add axes
            fig_emo.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="#303641", width=1.5))
            fig_emo.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="#303641", width=1.5))
            
            # Plot current user state point
            fig_emo.add_trace(go.Scatter(
                x=[val_x],
                y=[ar_y],
                mode='markers+text',
                marker=dict(color='#00f2fe', size=16, line=dict(width=2, color='white')),
                text=["Current State"],
                textposition="top center",
                name="Subject State"
            ))
            
            fig_emo.update_layout(
                plot_bgcolor='rgba(13, 17, 23, 0.9)',
                paper_bgcolor='rgba(13, 17, 23, 0.9)',
                xaxis=dict(title="Valence (Negative ← | → Positive)", range=[-1.1, 1.1], gridcolor='#21262d', zeroline=False),
                yaxis=dict(title="Arousal (Low/Calm  ← | → High)", range=[-1.1, 1.1], gridcolor='#21262d', zeroline=False),
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_emo, use_container_width=True)

# ------------- TAB 3: Explainability Explorer (XAI) -------------
with tab3:
    if model is None:
        st.info("💡 Please initialize the CardioMind-ECG model first to explore Explainable AI maps.")
    else:
        st.markdown("### Neural Explanations & Cardiologist Interpretability")
        
        # 1. Attention heatmaps
        attn_np = attn_weights[0].cpu().numpy()
        
        # Plot attention weight bar chart
        fig_attn = go.Figure(go.Bar(
            x=np.arange(len(attn_np)),
            y=attn_np * 100,
            marker_color=np.where(attn_np > 0.03, '#ff007f', '#4facfe'),
            text=[f"{w*100:.1f}%" if w > 0.03 else "" for w in attn_np],
            textposition='outside',
            name="Attention Weight"
        ))
        fig_attn.update_layout(
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(title="ECG Beat Sequence Index", gridcolor='#21262d'),
            yaxis=dict(title="Attention Weight (%)", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        
        # 2. Grad-CAM 1D on selected beat
        st.markdown("#### Morphological Attribution Explorer (Grad-CAM 1D)")
        
        # Select target beat (default: the one with the highest attention)
        max_attn_beat = int(np.argmax(attn_np))
        beat_idx = st.slider("Select Beat Index for Morphological Saliency", 0, len(attn_np)-1, max_attn_beat)
        
        target_beat = padded_beats[beat_idx]
        
        # Choose diagnostic head and class for Grad-CAM
        gc_task = st.selectbox("Grad-CAM Task Objective", ["arrhythmia", "stress", "valence", "arousal"])
        
        # Get target class count
        task_classes = {
            "arrhythmia": ["Normal", "AFib", "PVC", "VT", "Bradycardia"],
            "stress": ["Low", "Medium", "High"],
            "valence": ["Positive", "Negative"],
            "arousal": ["High", "Low"]
        }
        gc_class_idx = st.selectbox("Grad-CAM Target Class", 
                                    range(len(task_classes[gc_task])), 
                                    format_func=lambda x: task_classes[gc_task][x])
        
        # Compute Grad-CAM 1D
        cam_1d = compute_gradcam_1d(model, beat_seq_tensor, hrv_tensor, task=gc_task, class_idx=gc_class_idx, beat_idx=beat_idx, device=device)
        
        # Plot Grad-CAM with line coloring or color gradient scatter
        t_beat = np.arange(len(target_beat)) / fs * 1000.0  # ms
        
        fig_cam = go.Figure()
        # Underlying line
        fig_cam.add_trace(go.Scatter(
            x=t_beat, y=target_beat,
            mode='lines',
            line=dict(color='#8b949e', width=2),
            name='ECG Beat Segment'
        ))
        # Scatter colored by Grad-CAM score
        fig_cam.add_trace(go.Scatter(
            x=t_beat, y=target_beat,
            mode='markers',
            marker=dict(
                size=6,
                color=cam_1d,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Importance", thickness=15, len=0.8)
            ),
            name='Grad-CAM Saliency Map'
        ))
        
        fig_cam.update_layout(
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(title="Time from R-peak (ms)", gridcolor='#21262d'),
            yaxis=dict(title="Normalized Amplitude (z-score)", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300
        )
        st.plotly_chart(fig_cam, use_container_width=True)
        
        # 3. Clinical Report Card
        st.markdown("#### Clinical Interpretation Report")
        
        # Construct soft labels from prediction tensors
        softmax_preds = {
            "arrhythmia": torch.softmax(preds["arrhythmia"], dim=1).detach().cpu().numpy(),
            "stress": torch.softmax(preds["stress"], dim=1).detach().cpu().numpy(),
            "valence": torch.softmax(preds["valence"], dim=1).detach().cpu().numpy(),
            "arousal": torch.softmax(preds["arousal"], dim=1).detach().cpu().numpy(),
        }
        
        report_md = generate_clinical_report(softmax_preds, attn_np, sim_data["beat_types"][:seq_len], hrv_metrics)
        st.markdown(report_md)

# ------------- TAB 4: TinyML Edge Profiler -------------
with tab4:
    st.markdown("### Edge Deployment Benchmarking & TinyML Optimization")
    
    # TinyML Sidebar control simulations
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        pruning_ratio = st.slider("Pruning Ratio (Filters with low magnitude weight)", 0.0, 0.90, 0.60, step=0.05)
    with c_opt2:
        is_quantized = st.checkbox("Enable INT8 Post-Training Quantization (PTQ)", value=True)
        
    profiler = TinyMLProfiler()
    
    # Run profiling simulation
    if model is not None:
        stats = profiler.benchmark_deployment(model, pruning_ratio, is_quantized)
    else:
        # Fallback dummy stats if model not loaded
        temp_model = CardioMindECG()
        stats = profiler.benchmark_deployment(temp_model, pruning_ratio, is_quantized)
        
    # Build dataframes for visualization
    hw_names = list(stats.keys())
    sizes = [stats[h]["size_mb"] for h in hw_names]
    latencies = [stats[h]["latency_ms"] for h in hw_names]
    powers = [stats[h]["power_mw"] for h in hw_names]
    accuracies = [stats[h]["acc_retention_pct"] for h in hw_names]
    energies = [stats[h]["energy_per_inf_uj"] for h in hw_names]
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        # Latency chart
        fig_lat = go.Figure(go.Bar(
            x=hw_names,
            y=latencies,
            marker_color='#4facfe',
            text=[f"{l:.1f} ms" for l in latencies],
            textposition='auto'
        ))
        fig_lat.update_layout(
            title="Inference Latency (lower is better)",
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title="Latency (ms)", type="log" if max(latencies)/min(latencies) > 20 else "linear", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_lat, use_container_width=True)
        
        # Memory/Flash footprint chart
        fig_size = go.Figure(go.Bar(
            x=hw_names,
            y=sizes,
            marker_color='#00f2fe',
            text=[f"{s:.2f} MB" for s in sizes],
            textposition='auto'
        ))
        fig_size.update_layout(
            title="Model Binary Size / Flash Footprint (lower is better)",
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title="Size (MB)", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_size, use_container_width=True)
        
    with col_t2:
        # Power Draw chart
        fig_pow = go.Figure(go.Bar(
            x=hw_names,
            y=powers,
            marker_color='#ff3d00',
            text=[f"{p:,.0f} mW" if p >= 1000 else f"{p:.1f} mW" for p in powers],
            textposition='auto'
        ))
        fig_pow.update_layout(
            title="Continuous Power Draw (lower is better)",
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title="Power (mW)", type="log", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_pow, use_container_width=True)
        
        # Energy per Inference chart
        fig_eng = go.Figure(go.Bar(
            x=hw_names,
            y=energies,
            marker_color='#ff007f',
            text=[f"{e:,.0f} uJ" if e >= 10 else f"{e:.2f} uJ" for e in energies],
            textposition='auto'
        ))
        fig_eng.update_layout(
            title="Energy Consumed per Inference (lower is better)",
            plot_bgcolor='rgba(13, 17, 23, 0.9)',
            paper_bgcolor='rgba(13, 17, 23, 0.9)',
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title="Energy (uJ)", type="log", gridcolor='#21262d'),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280
        )
        st.plotly_chart(fig_eng, use_container_width=True)
        
    # Summary Table
    st.markdown("#### Hardware Suitability Summary Matrix")
    data_matrix = []
    for h in hw_names:
        suitability = "✅ High"
        limitations = "None"
        
        if h == "NVIDIA A100 GPU (Cloud)":
            suitability = "⚠️ Unsuitable for Wearable"
            limitations = "Requires Cloud link, >300W power draw, high server costs"
        elif h == "Raspberry Pi 4 (Edge CPU)":
            suitability = "⚡ Moderate (Companion app)"
            limitations = "Requires pocket battery, too large for watch case"
        elif h == "ARM Cortex-M7 (STM32H7)":
            suitability = "🏆 Optimal (Always-on)"
            limitations = "Requires quant/pruning size < 2MB"
        elif h == "ARM Cortex-M4 (STM32F4)":
            suitability = "✅ Highly Suitable (Ultra-low power)"
            limitations = "Higher latency (134ms), requires optimized math kernels"
            
        data_matrix.append({
            "Hardware Platform": h,
            "Model Size": f"{stats[h]['size_mb']:.2f} MB",
            "Inference Latency": f"{stats[h]['latency_ms']:.1f} ms",
            "Power Draw": f"{stats[h]['power_mw']:.1f} mW",
            "Relative Accuracy": f"{stats[h]['acc_retention_pct']:.1f}%",
            "Suitability Rating": suitability,
            "Critical Constraint": limitations
        })
    st.table(pd.DataFrame(data_matrix))
