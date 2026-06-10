# CardioMind-ECG 🩺🤖

### Wearable Cardiac Intelligence: Deep Temporal Attention Networks for Real-Time Stress, Arrhythmia, and Emotion Recognition

**CardioMind-ECG** is an end-to-end, edge-ready artificial intelligence framework designed to process raw, noisy Electrocardiogram (ECG) data from wearable devices. By leveraging state-of-the-art Deep Temporal Attention Networks, it simultaneously extracts autonomic biomarkers, detects cardiac arrhythmias, maps emotional states, and monitors physiological stress levels in real time.

---

## 🚀 Core Features

* **📈 Real-Time Signal Monitor & Preprocessing:** Features an interactive stream that instantly applies digital filtering (e.g., Butterworth bandpass) to isolate raw cardiac waveforms from heavy motion and electromyogram (EMG) artifacts.
* **🎯 Diagnostics & Multi-Task Predictor:** A unified deep learning architecture that outputs multi-class arrhythmia classifications, real-time stress indexes, and valence/arousal emotion mappings.
* **👁️ Explainability Explorer (XAI):** Uses temporal attention weights to visually highlight exactly *which* waves (P-wave, QRS complex, or T-wave) triggered the model's diagnostic decisions.
* **⚡ TinyML Edge Profiler:** Profiles the model's footprint for microcontroller deployment (e.g., ESP32, STM32), tracking RAM/Flash usage, latency, and quantization metrics.

---

## 📊 Extracted Biomarkers & HRV Metrics

The system calculates comprehensive Time-Domain, Frequency-Domain, and Non-Linear Heart Rate Variability (HRV) metrics on the fly:

| Metric | Category | Description |
| :--- | :--- | :--- |
| **Mean HR** | Time-Domain | Average heart rate measured in Beats Per Minute (BPM). |
| **RMSSD** | Time-Domain | Root mean square of successive differences; the primary index for **Vagal/Parasympathetic tone**. |
| **SDNN** | Time-Domain | Standard deviation of NN intervals; reflects overall autonomic nervous system variability. |
| **LF/HF Ratio** | Frequency-Domain | Ratio of Low Frequency to High Frequency power; gauges **Sympathovagal balance**. |
| **Sample Entropy** | Non-Linear | Quantifies the complexity and regularity of the R-R interval tachogram. |

---

## 🛠️ Tech Stack & Architecture

* **Signal Processing:** SciPy, NeuroKit2 (Butterworth filtering, Pan-Tompkins QRS detection)
* **Deep Learning:** PyTorch / TensorFlow (Temporal Convolutional Networks + Multi-Head Self-Attention)
* **Edge Deployment:** TinyML, TensorFlow Lite (TFLite) Micro, ONNX Runtime
* **Dashboard UI:** Streamlit (Dynamic plotting with Plotly/Matplotlib)

---

## ⚙️ Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your local environment.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/your-username/CardioMind-ECG.git](https://github.com/your-username/CardioMind-ECG.git)
cd CardioMind-ECG
pip install -r requirements.txt

```

### 3. Running the Dashboard Locally

Launch the real-time simulation and intelligence interface:

```bash
streamlit run app.py

```

---

## 🖥️ Using the Interface

1. **Select Your ECG Scenario:** Use the **Signal Settings** sidebar to switch between clinical patterns (e.g., *Normal Sinus Rhythm*, *Atrial Fibrillation*, or *Premature Ventricular Contractions*).
2. **Inject Synthetic Noise:** Adjust the **Motion & EMG Noise** slider to evaluate how resilient the Butterworth preprocessing filter and deep learning model are against artifact-heavy real-world environments.
3. **Deploy & Analyze:** Click **[Deploy]** to stream the data. Explore the **RR Interval Tachogram** and the **Autonomic Poincaré Plot** to visually assess cardiac stability and autonomic health.

---

## 🔮 Future Roadmap

* [ ] Integration with live Bluetooth Low Energy (BLE) wearable sensors (e.g., Polar H10, Shimmer).
* [ ] On-device INT8 quantization deployment verification for ultra-low-power ARM Cortex-M microcontrollers.
* [ ] Federated Learning pipelines to continuously update stress models while preserving user privacy.

---

## 📄 License

This project is licensed under the MIT License.
