# Chelsea methodology

## Original study

1. **Unit of analysis:**
2. **Targets:**
3. **Features:**
4. **Train/test split:**
5. **Models:**
6. **Evaluation metric:**

## Other notes
Chelsea – Code Violations and Public Health

Date: 06/11/2026
Notes by: Dhatri Anne

Overview
Leveraging data to expand city capacity to identify and resolve housing-related public health problems.
Implemented in 2014, expanded in 2018 to cover a larger target area.
This project uses machine learning to predict which properties in the City of Chelsea, MA will be exposed to public health hazards.
The model works by training on a subsection of the city that was subject to a more proactive code inspection program and then making predictions for the rest of the city.
Data was collected by different departments in Chelsea and uploaded to the Building Blocks platform.
In 2019, the city had 5,989 residential properties, 1,611 of which had been inspected between 2014–2018.
Analysis

The researchers built three models for each of their target variables:

Properties with any code violation.
Properties with code violations that present an elevated public health risk (a composite of four code violations selected in consultation with city housing inspectors).
Properties with code violations associated with overcrowding (illegal locks found inside building units).

For each target variable:

They evaluated multiple machine learning algorithms:
XGBoost
Random Forest
LASSO
The models were trained using a dataset of properties that were proactively inspected between 2014 and 2018 in a target area of the city.
Hyperparameters for each model were optimized using cross-validation.
Model performance was evaluated using the Precision–Recall Curve (PR Curve) metric.
After training, each algorithm was tested using inspection data collected outside of the target area, which was held out until the evaluation stage.
Results
Housing code violations were found in 54% of inspected properties, and 85% of those were classified as high-risk public health violations.
The machine learning approach resulted in approximately a 1.8-fold increase in the number of inspections that identified code violations compared with current inspection practices.