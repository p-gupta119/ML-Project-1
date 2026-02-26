# Machine Learning Project 1

**Team:** Pavi Gupta (tgz5gw), Sae-Jin Moon (rhn9qs), Timothy Lee (ghk5cd), Jonathan Lee (kkg2qc), David Kim (rcy3cs), Lydia Lee (yzg7bf), Kaitlin Luu (byw7uh), Isabelle Lee (vjm3mp)

## About
The goal of the first project is to do some wrangling, EDA, and visualization, and generate sequences of values. We will focus on:

- CDC National Health and Nutritional Examination Survey (NHANES, 1999-2000): https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=1999  
- CDC Linked Mortality File (LMF, 1999-2000): https://www.cdc.gov/nchs/data-linkage/mortality-public.html  

NHANES is a rich panel dataset on health and behavior, collected bi-yearly from around 1999 to now. We will focus on the 1999 wave, because that has the largest follow-up window, providing us with the richest mortality data. The mortality data is provided by the CDC Linked Mortality File.

The purpose of the project is to use NN to predict who dies (hard or soft classification) and how long they live (regression).

## MAP

```
.
├── data
│   ├── DEMO.xpt
│   ├── lmf_parsed.csv
│   ├── public-use-linked-mortality-file-description.pdf
│   └── SAS_ReadInProgramAllSurveys.sas
├── nb.ipynb                               - Jupyter notebook converted from nb.py for day 1 EDA (for easier viewing); Final product for all code/visualizations for day 1 responses
├── part-2-question4.ipynb                 - Day 2 Part 2 Question 4 (kNN regressor)
├── part-2.ipynb                           - Day 2 Part 2 Question 3 (kNN classification)
├── project_1_SAQresponses.ipynb           - Answers for Project 1
├── public/                                - Codebooks for variables (Day 1)
│   ├── codebook1.png                      - Ridageex (Exam Age in Months)
│   ├── codebook2.png                      - DMDEDUC2 (Education Level)
│   ├── codebook3.png                      - DMDMARTL (Martial Status)
│   ├── codebook4.png                      - DMDHHSIZ (Total number of people in the Household)
│   ├── codebook5.png                      - INDHHINC (Annual Household Income)
│   ├── codebook6.png                      - PERMTH_INT (Number of Persons Follow-up)
│   └── codebook7.png                      - MORTSTAT & ElIGSTAT
│   ├── contingency.png                    - Contingency Tables
│   ├── correlation.png                    - Correlation Matrix
│   ├── outlier.png                        - Outlier Boxplot
│   ├── ridge.png                          - Ridge plot
│   └── skew.png                           - Histograms
├── README.md                              - Project overview and instructions
└── temp
    ├── day1_wrangling_eda.py              - Basic EDA for Day 1
    ├── get_data.py                        - Basic EDA for Day 1
    ├── nb.py                              - Marimo notebook for day 1 EDA
    └── project_1_bsds.ipynb               - Instructions to Project 1
```