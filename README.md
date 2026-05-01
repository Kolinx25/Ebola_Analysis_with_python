# Ebola Outbreak Analysis — Sierra Leone, 2014

This project analyzes 200 confirmed Ebola cases from the early months
of the 2014 West Africa outbreak in Sierra Leone. It combines
epidemiological analysis with machine learning to answer two questions:
how fast was the outbreak growing, and what factors predicted whether
a case was detected quickly or late?

I originally worked on a version of this in R. This is the Python
rebuild — extended with R0 estimation and a Random Forest classifier.

---

## What's in here
---

## Dataset

Each row is a confirmed Ebola case with the following fields:

| Column | Description |
|---|---|
| id | Case identifier |
| age | Patient age in years |
| sex | M / F |
| status | All confirmed in this subset |
| date_of_onset | Date symptoms began |
| date_of_sample | Date sample was collected |
| district | District where case was recorded |

Source: Sierra Leone Ministry of Health / WHO 2014 outbreak records.

---

## What the analysis covers

**1. Demographics**
The median case age was 35 years. Women accounted for 57% of cases
(114 female, 86 male), which aligns with documented patterns of female
caregivers having higher exposure risk during the West Africa outbreak.

**2. Geographic distribution**
Kailahun district accounted for 155 of the 200 cases — 77.5% of the
entire dataset. Kenema was a distant second at 34. This reflects the
outbreak's origin point near the Guinea border and the initial
concentration of cases before spread to other districts.

**3. Temporal trend**
Cases were recorded between May 18 and June 30, 2014. The single
highest-incidence day was June 10, with 20 new onset cases. A 7-day
rolling average is plotted alongside daily incidence to smooth
day-to-day noise.

**4. R0 estimation**
Using the exponential growth rate method on cumulative cases during
the first 30 days, and applying Ebola's published serial interval of
15 days (Camacho et al., 2014):
R0 = 2.79 means each infected person was generating roughly 3 secondary
cases before any significant intervention. This sits slightly above
the WHO Ebola Response Team's published range of 1.71–2.02 for Sierra
Leone overall, which is expected — this dataset captures only Kailahun
and Kenema in the very first weeks before contact tracing and isolation
measures took hold.

**5. Surveillance response**
The median time from symptom onset to sample collection was 5 days.
75% of cases clustered exactly at 5 days, suggesting a fairly
consistent but slow field response. A small number of cases had delays
exceeding 15 days — those are the ones that likely drove onward
transmission the most.

**6. Random Forest — predicting detection speed**
I trained a Random Forest classifier (200 trees, balanced class weights)
to predict whether a case was detected rapidly (sampled within 4 days
of onset) or at the standard/delayed rate (5+ days). Features used:
age, sex, and district.

Feature importances:
- District : 0.58 — strongest predictor by far
- Age      : 0.39 — older patients tended toward slower detection  
- Sex      : 0.03 — negligible

The district result makes intuitive sense. Kailahun and Kenema had
very different surveillance infrastructure and case loads at this point
in the outbreak. Where you were mattered more than who you were when
it came to getting tested quickly.

A note on the class imbalance: rapid detection cases (<=4 days) made
up only 32 of 200 records, because 5 days was the near-universal
response time. The model handles this with balanced class weights, but
the rapid detection class still has limited test samples. The results
should be interpreted cautiously for that class.

---

## How to run it

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

Make sure `ebola_sierra_leone.csv` is in the same directory, then:

```bash
python ebola_analysis.py
```

Six plots will be saved to the working directory:
- `demographics.png`
- `district_distribution.png`
- `temporal_trend.png`
- `r0_estimation.png`
- `sample_delay.png`
- `random_forest_results.png`

---

## References

- WHO Ebola Response Team (2014). Ebola Virus Disease in West Africa.
  *New England Journal of Medicine*, 371(16), 1481–1495.
- Camacho et al. (2014). Potential for large outbreaks of Ebola virus
  disease. *Epidemics*, 9, 70–78.

---

## Author

Collins Amoo
MSc Data Management & Analysis, University of Cape Coast, Ghana
