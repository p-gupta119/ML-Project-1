# Machine Learning Project 1

**Team:** Pavi Gupta (tgz5gw), Sae-jin Moon (rhn9qs), Timothy Lee (ghk5cd), Jonathan Lee (kkg2qc), David Kim (rcy3cs), Lydia Lee (yzg7bf), Kaitlin Luu (byw7uh), Isabelle Lee (vjm3mp)

## About
The goal of the first project is to do some wrangling, EDA, and visualization, and generate sequences of values. We will focus on:

- CDC National Health and Nutritional Examination Survey (NHANES, 1999-2000): https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=1999  
- CDC Linked Mortality File (LMF, 1999-2000): https://www.cdc.gov/nchs/data-linkage/mortality-public.html  

NHANES is a rich panel dataset on health and behavior, collected bi-yearly from around 1999 to now. We will focus on the 1999 wave, because that has the largest follow-up window, providing us with the richest mortality data. The mortality data is provided by the CDC Linked Mortality File.

The purpose of the project is to use NN to predict who dies (hard or soft classification) and how long they live (regression).

## MAP

```text
ML-Project-1/
├── .ipynb_checkpoints/                  
│   ├── nb-checkpoint.ipynb                - Day 1 EDAs
│   ├── part-2-question4.ipynb             - Day 2 Part 2 Question 4 (kNN regressor)
│   └── part-2.ipynb                       - Day 2 Part 2 Question 3 (kNN classification)
│
├── data/                                  
│   ├── DEMO.xpt                           
│   ├── SAS_ReadInProgramAllSurveys...    
│   ├── lmf_parsed.csv                     
│   └── public-use-linked-mortality-file...
│
├── public/                                - Output for variables (Day 1)
│   ├── codebook1.png                      - Ridageex (Exam Age in Months)
│   ├── codebook2.png                      - DMDEDUC2 (Education Level)
│   ├── codebook3.png                      - DMDMARTL (Martial Status)
│   ├── codebook4.png                      - DMDHHSIZ (Total number of people in the Household)
│   ├── codebook5.png                      - INDHHINC (Annual Household Income)
│   ├── codebook6.png                      - PERMTH_INT (Number of Persons Follow-up)
│   └── codebook7.png                      - MORTSTAT & ElIGSTAT
│
├── .gitignore                             
├── README.md                              - Project overview and instructions
├── day1_wrangling_eda.py                  - Basic EDA for Day 1
├── get_data.py                           
├── histograms.png                         - Saved histogram figure output (Basic EDA for Day 1)
├── nb.ipynb                               - Main EDA for day 1
├── nb.py                                  - Script/Code for notebook
├── project_1_SAQresponses.ipynb           - Solution to Answers for Project 1
└── project_1_bsds.ipynb                   - Instructions to Project 1
