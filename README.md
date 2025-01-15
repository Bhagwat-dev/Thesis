# Thesis: Concept Evaluation for Lifetime Prediction of Electrical Contacts under Dynamic Load

This repository contains materials and scripts from my Master's thesis, which focuses on analyzing contact dynamics and predicting the lifecycle of electrical contacts using Python and MATLAB. The project includes steps from data preprocessing to fatigue curve generation, prediction and validation of Result.

Prediction Accuracy: 90%

---

## **Repository Structure**

Lifetime-prediction-thesis/
├── code/                          # Scripts and tools
│   ├── matlab/                    # MATLAB scripts
│   │   ├── conversion.m           # Script to convert .txt to .mat files
│   │   ├── Loading_data_10_exp.m  # Combines multiple .mat files
│   │   ├── Experiments_analysis.m # Generates failure characteristics
│   │   ├── Final/                 # Final MATLAB scripts for analysis
│   ├── python/                    # Python scripts
│   │   ├── Prediction_tool.py     # Lifecycle prediction tool
│   │   ├── spread_trendline_extrapolation.py # Updates Wöhler curve
│   │   ├── Resistance/            # Resistance analysis scripts
│   │   ├── Amplitude/             # Amplitude comparison scripts
│   │   ├── Functions/             # Signal processing and utilities
├── data/                          # Experimental and processed data
│   ├── raw/                       # Raw experimental .txt data
│   ├── processed/                 # Processed .mat files for analysis
│   ├── validation/                # Validation datasets
│   ├── examples/                  # Example data for scripts
├── docs/                          # Documentation and resources
│   ├── thesis.pdf                 # Full thesis document
│   ├── figures/                   # Images, graphs, and plots
│   ├── README.md                  # Main repository README
├── results/                       # Outputs and results of analysis
│   ├── wohler_curves/             # Generated Wöhler curves
│   ├── predictions/               # Prediction outputs (e.g., cycle numbers)
├── LICENSE                        # License for the repository
└── README.md                      # Main description of the project

