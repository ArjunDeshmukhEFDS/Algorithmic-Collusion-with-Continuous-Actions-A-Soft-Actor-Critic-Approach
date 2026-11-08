
Code and reproduction guide for the Advanced Machine Learning individual project.

The reproducibility has two stages:

Data generation — Python scripts train models for each experimental case and write results to Data/.
Figures — Two Jupyter notebooks read from Data/ and produce every figure used in the paper.

Folder Structure
Code/
├── Data Generation/
│   ├── Baseline/           # baseline experiment
│   ├── 3 Firms/            # three-firm variant
│   ├── Asymmetric Costs/   # asymmetric-cost variant
│   ├── Discrete Actions/   # discrete-action variant
│   └── Robustness/         # robustness checks
│
└── Graphs in Paper/
    ├── Data_Section.ipynb     # figures for the data section
    └── Results_Section.ipynb  # figures for the results section

Data/                       # populated by the scripts above; read by the notebooks


Requirements

Python 3.10+
Install dependencies:

Core packages used: pip install numpy, pandas, matplotlib, jupyter, torch, random, csv, time , scipy. 

Reproducing the results

Step 1 — Generate the data
python "Code/Data Generation/Baseline/Soft-Actor-Critic Fixed Convergence.py"
python "Code/Data Generation/3 Firms/3 firms.py"
python "Code/Data Generation/Asymmetric Costs/Diff MCs.py"
python "Code/Data Generation/Discrete Actions/Discrete Actions.py"
python "Code/Data Generation/Robustness/Robustness.py"
python "Code/Data Generation/Robustness/RobustnessTauMu.py"

Compute time
Each training session takes approximately 40 minutes on a 6 core CPU 3.8GHz (AMD Ryzen 7600) and NVIDIA RTX 3060. 

Runtime Estimates (40 min per session)
Case                Sessions    Runtime
----------------    --------    ----------------
Baseline            20          800 min  ~ 13.3 h
3 Firms             5           200 min  ~  3.3 h
Asymmetric Costs    5           200 min  ~  3.3 h
Discrete Actions    5           200 min  ~  3.3 h
Robustness          3           120 min  =  2.0 h
----------------    --------    ----------------
Total               38          1520 min ~ 25.3 h


Step 2 — Generate the figures
Once you have the data, launch Jupyter and run the two notebooks.
