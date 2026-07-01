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

### Chelsea Study — Methodology Notes

Below are the key characteristics of the original Chelsea multitenant inspection study, organized into the six categories requested.

---

### 1. Unit of Analysis
The Chelsea study works at the **building level**. Each row in the dataset represents a single multifamily property. All features, inspection outcomes, and signals are tied to the building itself, not to individual units or census blocks. This makes the model very operational: it is directly predicting conditions at specific properties the city might inspect.

---

### 2. Targets (Outcome Variable)
Chelsea’s target variable is **whether a building fails inspection**.  
This is a binary outcome created from historical inspection records. A “fail” indicates meaningful distress or code issues. Because Chelsea cannot inspect every building, predicting failures helps them decide which buildings should be prioritized next.

---

### 3. Features (Predictors)
Chelsea uses a rich set of **building‑level and neighborhood‑level features**, including:

- Historical inspection outcomes  
- Code violations  
- 311 complaint patterns  
- Building age and size  
- Ownership/management characteristics  
- Neighborhood indicators (e.g., socioeconomic context)

These features give the model strong predictive power because they directly reflect building conditions and resident experiences.

---

### 4. Train/Test Split
Chelsea uses a **temporal train/test split**.  
Older inspection data is used for training, and newer inspection cycles are used for testing. This mirrors how the model would be used in real life: training on past inspections to predict future ones. The split is designed to avoid leakage and to reflect real operational timelines.

---

### 5. Models
Chelsea tests several models, but the two main ones are:

- **Random Forest**  
- **XGBoost**

Both models handle nonlinear relationships well and work effectively with mixed feature types. They also produce feature importance measures that help the city understand which signals matter most.

---

### 6. Evaluation Metric
Chelsea evaluates models using **precision at K**.  
This metric focuses on how well the model identifies the highest‑risk buildings — the top K properties the city would actually inspect. Because Chelsea has limited inspection capacity, precision at K is the most meaningful metric: it tells them how many of the “top picks” are truly distressed.

---

### Summary
Chelsea’s model is designed for **prioritization**, not description. It predicts *which buildings should be inspected next* based on rich operational data. The entire methodology reflects the city’s limited inspection capacity and the need to target the highest‑risk properties.
