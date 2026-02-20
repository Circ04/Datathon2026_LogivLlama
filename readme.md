# Datathon 2026 - LogivLlama

## About the Dataset

**Duolingo 2020 Notification Bandit Dataset** - This dataset contains millions of practice reminder push notifications sent to Duolingo users over a 35-day period. It's designed for offline evaluation of multi-armed bandit algorithms that optimize notification selection to increase user engagement.

### Dataset Citation
If you use this dataset, please cite:
```
@inproceedings{
  yancey2020sleeping,
  title={A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on Knowledge discovery and data mining},
  author={Kevin Yancey and Burr Settles},
  year={2020},
  doi = {10.1145/3394486.3403351},
  url = {https://doi.org/10.1145/3394486.3403351}
}
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd Datathon2026_LogivLlama
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
The dataset is distributed across 6 ZIP files on Google Drive (not included in git).

**To download:**
```bash
python download_data.py
```

This will:
- Create the `Data/` folder automatically
- Download all 6 parts (training + test data)
- Extract parquet files to `Data/`
- The script already has the correct Google Drive file IDs configured

**Expected structure after download:**
```
Data/
├── training/          # 15 days, 87.7M rows
│   └── *.parquet
└── test/             # 19 days, 114.5M rows
    └── *.parquet
```

## Dataset Schema

Each notification event has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | `float` | Days since experiment start (e.g., 4.25 = 4 days, 6 hours) |
| `ui_language` | `string` | User's UI language (ISO 639-1 code) |
| `eligible_templates` | `List[string]` | Templates available for this notification (A-L) |
| `history` | `List[Tuple[string, float]]` | Past templates sent + performance scores |
| `selected_template` | `string` | Template actually sent |
| `session_end_completed` | `bool` | Whether user completed a lesson within 2 hours (reward) |

### Template Examples
- Templates A-L represent different notification messages (e.g., "Time for your Spanish lesson!", "Don't break your streak!")
- `eligible_templates` = which templates are allowed for this user/time
- `history` = weighted performance scores for each template based on past outcomes

**This dataset is designed for bandit algorithms, but feel free to think outside the box for creative analyses!**

## Project Structure
```
Datathon2026_LogivLlama/
├── Data/                    # Dataset (gitignored, download via script)
│   ├── training/
│   └── test/
├── download_data.py         # Automated download script
├── Datathon Analysis.ipynb  # Main analysis notebook
├── requirements.txt         # Python dependencies
└── readme.md               # This file
```
