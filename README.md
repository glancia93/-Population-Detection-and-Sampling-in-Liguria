# Data-Driven Strategies for Detecting and Sampling misrepresented subgroups

This repository contains the code, simulations, and analytical notebooks developed for the study **“Data-Driven Strategies for Detecting and Sampling misrepresented subgroups”**, based on the **EU-SILC 2019 Ligurian sub-sample**.

The project focuses on identifying **underrepresented and overrepresented population subgroups**, assessing sampling bias, and proposing methodological strategies to improve detection and allocation in official survey data.

---

## Project Objectives

- Detect rare and hard-to-reach population groups within survey data  
- Analyze sampling distortions in the EU-SILC Ligurian sub-sample  
- Propose simulation-based strategies for optimal sample allocation  
- Provide visual and quantitative tools to support public statistical use  

---

## Data Source

- **EU-SILC (European Union Statistics on Income and Living Conditions)**  
- Cross-sectional data, **Liguria – 2019**

**Note:** Due to data privacy restrictions, the original EU-SILC microdata are not included in this repository.

---

## Folder Description

### Simulation

This folder contains the core methodological and computational components of the project.

- **Simulation.py**  
  Main simulation engine for sampling and subgroup detection experiments.

- **sampling.py**  
  Functions implementing sampling strategies and subgroup extraction procedures.

- **optimal_cost_allocation.py**  
  Cost-based optimal allocation methods for rare population detection.

- **Plots.py**  
  Utility functions for visualization and comparison of sampling outcomes.

- **Comparisions_plots_MFS.ipynb**  
  Notebook comparing multiple sampling frameworks and allocation strategies.

- **Data_Visualization.ipynb**  
  Exploratory data analysis and visualization of simulated data.

### Outlier Detection

This folder contains the core metodological and computation to investigate the under-represented subgroups within the framework of outlier detection

- **Outlier_Detection_EUSILC.py**
Function and python classes to train and validate the outlier detection models developed (both univariate and multivariate)

- ** Example_Investigation_Underrepresented_Groups.ipynb**
Notebook providin an illustrative example of outlioer detection analysis to search for under represented groups

#### Supplementary Files

- **PytonScripts.zip**: additional development scripts  
- **PseudoCode_Instructions.zip**: pseudocode and methodological notes  

---

## Requirements

The project is implemented in **Python 3.x** and relies on the following libraries:

- `numpy`  
- `pandas`  
- `scipy`  
- `matplotlib`  
- `seaborn`  
- `jupyter`

## Repository Structure

```text
Population-Detection-and-Sampling-in-Liguria/
│
├── README.md
│
├── Simulation/
│   ├── Simulation.py
│   ├── sampling.py
│   ├── optimal_cost_allocation.py
│   ├── Plots.py
│   ├── Comparisions_plots_MFS.ipynb
│   ├── Data_Visualization.ipynb
│   ├── PytonScripts.zip
│   ├── PseudoCode_Instructions.zip
│   └── __pycache__/
│
└── .git/
```
---


