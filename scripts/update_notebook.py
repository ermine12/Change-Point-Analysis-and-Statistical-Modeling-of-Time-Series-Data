import json
import os

notebook_path = r'c:\Users\ELITEBOOK\Documents\Projects\AI_engineering\Change Point Analysis and Statistical Modeling of Time Series Data\notebooks\01_change_point_analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Imports and Config
nb['cells'][1]['source'] = [
    "\n",
    "import sys\n",
    "print(sys.executable)\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pymc as pm\n",
    "import arviz as az\n",
    "import os\n",
    "import pytensor\n",
    "\n",
    "# Pre-emptive PyTensor configuration for Windows robustness\n",
    "pytensor.config.gcc__cxxflags = \"-DMS_WIN64\"\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn-v0_8-whitegrid')"
]

# Cell 3: Data Loading (fix date format)
# Note: cell indices might vary, but in the previous view it was 'execution_count': 3
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any("pd.read_csv" in line for line in cell['source']):
        cell['source'] = [
            "data_path = '../data/BrentOilPrices.csv'\n",
            "df = pd.read_csv(data_path)\n",
            "df['Date'] = pd.to_datetime(df['Date'], format='mixed')\n",
            "df = df.sort_values('Date').reset_index(drop=True)\n",
            "\n",
            "# Focus on a more recent period for clearer analysis (e.g., 2015 - 2022)\n",
            "df_recent = df[df['Date'] >= '2015-01-01'].copy()\n",
            "\n",
            "plt.figure(figsize=(15, 6))\n",
            "plt.plot(df_recent['Date'], df_recent['Price'])\n",
            "plt.title('Brent Oil Prices (2015-2022)')\n",
            "plt.xlabel('Date')\n",
            "plt.ylabel('Price (USD)')\n",
            "plt.show()"
        ]
        break

# Add extra EDA cell after Cell 4 (Log Returns)
# Find index of Cell 4
log_returns_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any("df_recent['Log_Returns']" in line for line in cell['source']):
        log_returns_idx = i
        break

if log_returns_idx != -1:
    new_eda_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Rolling statistics and distribution of log returns\n",
            "window = 30\n",
            "df_recent['Rolling_Mean'] = df_recent['Log_Returns'].rolling(window=window).mean()\n",
            "df_recent['Rolling_Std'] = df_recent['Log_Returns'].rolling(window=window).std()\n",
            "\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))\n",
            "\n",
            "# Rolling stats plot\n",
            "ax1.plot(df_recent['Date'], df_recent['Rolling_Mean'], label='Rolling Mean (30d)', color='blue')\n",
            "ax1.plot(df_recent['Date'], df_recent['Rolling_Std'], label='Rolling Std (30d)', color='red', alpha=0.5)\n",
            "ax1.set_title('Rolling Statistics of Log Returns')\n",
            "ax1.legend()\n",
            "\n",
            "# Distribution plot\n",
            "sns.histplot(df_recent['Log_Returns'], kde=True, ax=ax2, color='purple')\n",
            "ax2.set_title('Distribution of Daily Log Returns')\n",
            "ax2.set_xlabel('Log Return')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    nb['cells'].insert(log_returns_idx + 1, new_eda_cell)

# Cell 5: Modeling sampling parameters
# Re-index because of insertion
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "pm.sample" in "".join(cell['source']):
        new_source = []
        for line in cell['source']:
            if "pm.sample" in line:
                new_source.append("    trace = pm.sample(1000, tune=500, chains=2, cores=1, return_inferencedata=True, random_seed=42)\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully.")
