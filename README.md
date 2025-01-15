# Thesis: Concept Evaluation for Lifetime Prediction of Electrical Contacts under Dynamic Load

This repository contains materials and scripts from my Master's thesis, which focuses on analyzing contact dynamics and predicting the lifecycle of electrical contacts using Python and MATLAB. The project includes steps from data preprocessing to fatigue curve generation, prediction and validation of Result.

Prediction Accuracy: 90%


---

## **Features**
1. **Contact Dynamics Analysis:**
   - Data loading and hysteresis filtering.
   - Fast Fourier Transform (FFT) for signal analysis.
2. **Wöhler Curve Generation:**
   - Using failure characteristics data for fatigue analysis.
   - Includes manual input for specific amplitudes (e.g., 10 µm, 20 µm).
3. **Prediction Tool (Python):**
   - Lifecycle predictions using contact dynamics and failure characteristics data.
   - Outputs cycle counts for specific limiting resistance thresholds.
4. **Validation and Comparison:**
   - Resistance and amplitude data comparisons across experiments.

---

## **How to Use**

### **1. Preprocess Data (MATLAB)**
1. Convert raw `.txt` experimental data into `.mat` format using `conversion.m`.
2. Use `Loading_data_10_exp.m` to combine datasets.
3. Generate failure characteristics data using `Experiments_analysis.m`.

### **2. Generate Wöhler Curve (MATLAB)**
- Run `spread_trendline_extrapolation.py` for specific amplitude data.
- Ensure paths for `.mat` files are correctly set.

### **3. Predict Lifecycle (Python)**
1. Use the `Prediction_tool.py` script:
   - Define the limiting resistance, hysteresis filter threshold, and sampling rate.
   - Provide paths for input `.txt` files (contact dynamics) and `.mat` files (failure characteristics).
2. Run the script for predictions (approximately 10 minutes).

---

## **Requirements**
- MATLAB (tested with version 2014)
- Python (version >= 3.8) with the following libraries:
  - NumPy
  - SciPy
  - Matplotlib
  - Rainflow
  - Pandas
- Sefram Viewer for `.rec` to `.txt` file conversion.

---

## **Results**
The key results include:
- Wöhler curve data and fatigue point predictions for various amplitudes (5 µm to 30 µm).
- Validation metrics based on resistance and amplitude comparison.

---

## **Acknowledgments**
Special thanks to THD (my university), Rosenberger Group and colleagues for their support during this research.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

