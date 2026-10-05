# Dataset Guide & Directory Structure

## Dataset Source
- **Official Google Drive Folder:** [Dataset Link](https://drive.google.com/drive/folders/1I3sQZnnjeutdWXM9fNloVTk26zIxXYky)
- Provided for the **SAMATRIX RESUMEFORGE 2026 Hackathon**.

> **IMPORTANT:** Per project rules and competition guidelines, **RAW AND PROCESSED DATASETS MUST NEVER BE COMMITTED TO GITHUB**. The `.gitignore` is pre-configured to exclude all `.csv`, `.xlsx`, `.pdf`, `.zip`, and data artifacts. Only directory structures (`.gitkeep`) and this documentation are tracked.

---

## Directory Organization

```text
data/
├── README.md               <-- Dataset documentation (this file)
├── raw/                    <-- Local storage for raw files (Git-ignored)
│   ├── csv/                <-- Place Resume.csv & Resume.xlsx here
│   │   ├── Resume.csv
│   │   └── Resume.xlsx
│   └── pdf_resumes/        <-- Extracted PDF category folders (2,500 PDFs)
│       ├── ACCOUNTANT/
│       ├── ADVOCATE/
│       ├── AGRICULTURE/
│       ├── APPAREL/
│       ├── ARTS/
│       ├── AUTOMOBILE/
│       ├── AVIATION/
│       ├── BANKING/
│       ├── BPO/
│       ├── BUSINESS-DEVELOPMENT/
│       ├── CHEF/
│       ├── CONSTRUCTION/
│       ├── CONSULTANT/
│       ├── DESIGNER/
│       ├── DIGITAL-MEDIA/
│       ├── ENGINEERING/
│       ├── FINANCE/
│       ├── FITNESS/
│       ├── HEALTHCARE/
│       ├── HR/
│       ├── INFORMATION-TECHNOLOGY/
│       ├── PUBLIC-RELATIONS/
│       ├── SALES/
│       └── TEACHER/
└── processed/              <-- Storage for cleaned text, splits, and tokenized artifacts (Git-ignored)
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## Dataset Schema

### Tabular Data (`Resume.csv` / `Resume.xlsx`)
- **Total Records:** 2,484 resumes
- **Target Classes:** 24 distinct professional categories
- **Columns:**
  - `ID`: Unique identifier for each candidate / resume
  - `Resume_str`: Extracted plain-text resume content
  - `Resume_html`: Raw HTML-formatted resume text
  - `Category`: Target job classification category (Label)

### PDF Resumes (`data/raw/pdf_resumes/`)
- Contains ~2,500 categorized PDF documents structured across the 24 categories.
- Useful for end-to-end PDF parsing and prediction testing in the demo app.

---

## The 24 Target Categories
1. `ACCOUNTANT`
2. `ADVOCATE`
3. `AGRICULTURE`
4. `APPAREL`
5. `ARTS`
6. `AUTOMOBILE`
7. `AVIATION`
8. `BANKING`
9. `BPO`
10. `BUSINESS-DEVELOPMENT`
11. `CHEF`
12. `CONSTRUCTION`
13. `CONSULTANT`
14. `DESIGNER`
15. `DIGITAL-MEDIA`
16. `ENGINEERING`
17. `FINANCE`
18. `FITNESS`
19. `HEALTHCARE`
20. `HR`
21. `INFORMATION-TECHNOLOGY`
22. `PUBLIC-RELATIONS`
23. `SALES`
24. `TEACHER`

---

## Quick Setup Instructions for Local Development
1. Extract `Resume-20261005T051145Z-1-001.zip`:
   - Copy `Resume.csv` and `Resume.xlsx` into `data/raw/csv/`.
2. Extract `data-20261005T051214Z-1-001.zip`:
   - Copy category folders into `data/raw/pdf_resumes/`.
3. Run `notebooks/01_data_quality_eda.ipynb` or `src/data_loader.py` to verify loading.
