Healthcare Operational Analytics.
Project Overview.

This project presents an interactive healthcare analytics dashboard built with Streamlit, designed to explore operational and financial patterns in hospital data.

The dashboard focuses on identifying cost drivers, length of stay behavior, and executive level insights that support data driven decision making in healthcare operations.

Objectives:
Analyze Length of Stay (LOS) patterns across patient segments.
Evaluate whether longer stays correlate with higher billing amounts.
Compare average billing amounts by medical condition.
Assess whether medical conditions significantly drive cost differences.

Key KPIs ( Key Performance Indicators ).
Total Patients.
Average Length of Stay ( days ).
Average Billing Amount ( $ ).
Billing Variability Across Medical Conditions.

Key Insights.
Length of Stay shows variability across admissions, but no extreme outliers dominate costs.
Longer stays may correlate with higher billing, suggesting LOS is a potential operational cost driver.
Average billing amounts are highly uniform across medical conditions.

Statistical validation using Coefficient of Variation ( CV Coefficient of Variation) confirms that:
Cost differences by medical condition are statistically negligible.
Medical condition alone is not a strong billing driver.
Tech Stack.
Python.
Streamlit ( interactive dashboard ).
Pandas ( data manipulation ).
Matplotlib ( visualization ).

Deployment:
The application is deployed as a web based dashboard and can be accessed online:

Live App: ( add Render URL here once deployed ).
Project Structure
healthcare-operational-analytics/
│
├── app.py # Streamlit application
├── data/
│ └── healthcare_dataset_clean.csv
├── .streamlit/
│ └── config.toml
├── requirements.txt
├── runtime.txt
└── README.md

Notes:
This project was designed as a portfolio ready analytics case, emphasizing:
Clean data preprocessing.
Business-oriented insights.
Clear executive communication.
