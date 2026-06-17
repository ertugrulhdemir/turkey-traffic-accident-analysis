# 🚗 Turkey Traffic Accident Analysis (2013–2025)

An interactive dashboard analyzing road traffic accidents across Turkey's 81 provinces using official TÜİK data spanning 13 years (2013–2025), with machine learning forecasts through 2030 and a global road safety comparison powered by WHO data.

## 🌐 Live Demo
**[turkey-traffic-accident-analysis.streamlit.app](https://turkey-traffic-accident-analysis.streamlit.app)**

---

## 📊 Key Findings (2025)

| Metric | Value |
|--------|-------|
| Most accidents | İstanbul — 369,281 |
| Most fatalities | Ankara — 159 |
| Most dangerous province | Kilis — 15.9 deaths per 1,000 accidents |
| Driver fault rate | 90.6% of all accidents |
| Deadliest month | August |
| Deadliest day | Friday |
| Turkey vs world average | 51% lower death rate (WHO, 2021) |

---

## 🔍 Dashboard Features

- **13-Year National Trend** — Accidents, fatalities and injuries from 2013–2025 with COVID-19 impact highlighted; province-level year-by-year breakdown
- **Province Analysis** — Per-province metrics and danger rankings across all 81 provinces, with three comparison views
- **Time Analysis** — Monthly and daily accident distribution patterns for 2025
- **Fault Analysis** — Driver, pedestrian, road and vehicle fault breakdowns from 2010–2025
- **ML Prediction** — National and province-level forecasts through 2030, normalized risk trends per 100k population and vehicles, feature importance
- **Interactive Map** — Choropleth map of Turkey with switchable metrics (accidents, deaths, fatality rate)
- **Global Context** — Turkey ranked against 197 countries using WHO road safety data, with an interactive country comparison tool

---

## 🤖 Machine Learning

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| Accident Forecasting | National 2026–2030 accident estimates | GradientBoosting + Ridge blend |
| Death Forecasting | Chained prediction via fatal accident rate trend | GradientBoosting + Ridge blend |
| Province Forecasting | Per-province 2026–2030 accident estimates | Random Forest Regressor |
| Risk Normalization | Deaths per 100k population and per 100k vehicles | Linear Regression |

**Features used:** year · registered vehicles · driver fault ratio · vehicles involved in accidents

**Chained death prediction:** Instead of predicting deaths directly, the model first estimates total accidents, then applies a separately predicted fatality rate — making the forecast more realistic and avoiding the assumption that every accident results in a fatality.

---

## ⚙️ Autonomous Data Pipeline

- **TÜİK bulletin monitoring** — On every load, the app queries the TÜİK data portal API to check whether a newer annual bulletin has been published. If new data is available, a notification appears with the files to download.
- **WHO data** — The Global Context tab always fetches live data from the WHO Global Health Observatory API, so rankings update automatically when WHO publishes new figures.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Dashboard:** Streamlit
- **Data processing:** pandas, numpy
- **Machine learning:** scikit-learn (RandomForest, GradientBoosting, Ridge, LinearRegression)
- **Visualisation:** Plotly
- **Geospatial:** GeoJSON (81-province Turkey map)
- **External APIs:** TÜİK Veri Portalı, WHO Global Health Observatory

---

## 🚀 Run Locally

```bash
git clone https://github.com/ertugrulhdemir/turkey-traffic-accident-analysis
cd turkey-traffic-accident-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Data Sources

| Source | Content | Years |
|--------|---------|-------|
| [TÜİK — Road Traffic Accidents](https://data.tuik.gov.tr) | Province-level accidents, deaths, injuries | 2013–2025 |
| [TÜİK — Fault Statistics](https://data.tuik.gov.tr) | Driver, pedestrian, road, vehicle fault breakdown | 2020–2025 |
| [TÜİK — Registered Vehicles](https://data.tuik.gov.tr) | Annual registered vehicle count | 2010–2025 |
| [TÜİK — Risk Metrics](https://data.tuik.gov.tr) | Deaths and injuries per 100k population and vehicles | 2015–2025 |
| [WHO Global Health Observatory](https://ghoapi.azureedge.net) | Road traffic death rate per 100k population | Latest available |

---

## 💡 Key Insights

- Despite COVID-19 reducing traffic volume in 2020, fatalities dropped less than expected — suggesting speed increased on emptier roads
- Eastern provinces (Kilis, Hakkari, Şanlıurfa) show disproportionately high death rates relative to accident counts, pointing to road infrastructure gaps and limited emergency response capacity
- Driver fault accounts for over 90% of all accidents consistently across 15 years
- August and Friday are consistently the deadliest month and day
- Turkey's road death rate per capita has been declining year-on-year, even as total vehicle numbers rise — suggesting road safety improvements are outpacing traffic growth
- Turkey sits in the safer half globally (rank 152/197), with a death rate 51% below the world average (WHO, 2021)

---

## 👤 Developer

**Ertuğrul Halisdemir** · [GitHub](https://github.com/ertugrulhdemir)