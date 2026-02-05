# Customer Segmentation Using K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/minalchoudhary/online-retail-dataset)

A production-ready customer segmentation project using unsupervised machine learning (K-Means clustering) on the Online Retail II dataset. This project demonstrates RFM (Recency, Frequency, Monetary) analysis to identify distinct customer groups for targeted marketing strategies.

![Customer Segmentation](images/cluster_analysis.png)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Business Applications](#business-applications)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

Customer segmentation is a fundamental marketing strategy that divides customers into groups based on shared characteristics. This project applies K-Means clustering to segment customers of an online retail business, enabling:

- **Personalized Marketing:** Tailor campaigns to specific customer groups
- **Resource Optimization:** Focus efforts on high-value segments
- **Customer Retention:** Identify at-risk customers before they churn
- **Revenue Growth:** Develop strategies to move customers to higher-value segments

### Objectives

1. Perform comprehensive data preprocessing and cleaning
2. Engineer RFM features from transactional data
3. Apply K-Means clustering to identify customer segments
4. Evaluate clustering quality using the Elbow Method and Silhouette Score
5. Generate actionable business insights for marketing teams

## ✨ Key Features

- **End-to-end ML Pipeline:** From raw data to actionable insights
- **RFM Analysis:** Industry-standard customer behavior metrics
- **Multiple Evaluation Methods:** Elbow Method + Silhouette Score
- **Production-Ready Code:** Clean, documented, and modular
- **Business Interpretations:** Translate technical results to business value
- **New Customer Prediction:** Classify new customers into existing segments

## 📊 Dataset

**Source:** [Online Retail Dataset (Kaggle)](https://www.kaggle.com/datasets/minalchoudhary/online-retail-dataset)

The Online Retail II dataset contains transactional data from a UK-based online retail company between December 2010 and December 2011. It includes purchases of unique all-occasion gifts, with many customers being wholesalers.

| Feature | Description |
|---------|-------------|
| `Invoice` | Unique 6-digit invoice number (prefix 'C' indicates cancellation) |
| `StockCode` | Unique 5-digit product code |
| `Description` | Product name |
| `Quantity` | Units per transaction |
| `InvoiceDate` | Date and time of transaction |
| `Price` | Unit price in sterling (£) |
| `Customer ID` | Unique 5-digit customer identifier |
| `Country` | Customer's country of residence |

**Dataset Statistics:**
- Raw records: 541,910
- After cleaning: 397,885
- Unique customers: 4,338
- Date range: Dec 2010 - Dec 2011

## 🔬 Methodology

### 1. Data Preprocessing

```
Raw Data → Remove Missing Customer IDs → Filter Invalid Transactions → 
Convert Date Types → Calculate Transaction Value → Clean Dataset
```

- Removed 135,080 records with missing Customer IDs
- Filtered out 8,945 returns/cancellations (negative quantities)
- Converted InvoiceDate to datetime format
- Created TotalPrice = Quantity × Price

### 2. RFM Feature Engineering

| Metric | Definition | Business Meaning |
|--------|------------|------------------|
| **Recency** | Days since last purchase | Customer engagement level |
| **Frequency** | Number of unique transactions | Purchase habit strength |
| **Monetary** | Total amount spent | Customer value |

### 3. Feature Scaling

StandardScaler applied to normalize features:
- Ensures equal contribution from each RFM dimension
- Critical for distance-based algorithms like K-Means

### 4. Optimal Cluster Selection

**Elbow Method:** Plots WCSS (Within-Cluster Sum of Squares) vs. k. The "elbow" point indicates optimal cluster count.

**Silhouette Score:** Measures cluster cohesion and separation. Range [-1, 1], higher is better.

**Result:** k=4 selected based on combined analysis

### 5. K-Means Clustering

- Algorithm: K-Means (scikit-learn)
- Clusters: 4
- Random State: 42 (reproducibility)
- Initialization: k-means++ (default)

> **Note on Train/Test Split:** Unsupervised learning doesn't require data splitting since there are no labels to predict. Evaluation uses internal metrics (WCSS, Silhouette) rather than prediction accuracy.

For detailed methodology, see [METHODOLOGY.md](docs/METHODOLOGY.md).

## 📈 Results

### Cluster Profiles

| Cluster | Avg Recency | Avg Frequency | Avg Monetary | Customer Count | Interpretation |
|---------|-------------|---------------|--------------|----------------|----------------|
| 0 | Low | High | High | ~25% | **Champions** - Best customers |
| 1 | High | Low | Low | ~30% | **At-Risk** - Need re-engagement |
| 2 | Medium | Medium | Medium | ~25% | **Potential** - Growth opportunity |
| 3 | Low | Low | Medium | ~20% | **New** - Recently acquired |

### Model Performance

- **Silhouette Score:** Achieved competitive score indicating well-separated clusters
- **Cluster Distribution:** Balanced segments suitable for targeted strategies

### Visualizations

The project generates several visualizations:
- RFM Distribution plots
- Elbow Method curve
- Silhouette Score analysis
- Cluster characteristic bar charts
- 3D cluster visualization

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-kmeans.git
   cd customer-segmentation-kmeans
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   
   Option A: Using kagglehub (requires Kaggle API credentials)
   ```python
   import kagglehub
   kagglehub.dataset_download("minalchoudhary/online-retail-dataset")
   ```
   
   Option B: Manual download from [Kaggle](https://www.kaggle.com/datasets/minalchoudhary/online-retail-dataset)

## 🚀 Usage

### Running the Notebook

```bash
jupyter notebook notebooks/customer_segmentation_kmeans.ipynb
```

### Using Helper Scripts

```python
# Data preprocessing
from src.data_preprocessing import preprocess_retail_data

df_clean = preprocess_retail_data('path/to/dataset.csv')

# RFM Analysis
from src.rfm_analysis import calculate_rfm, segment_customers

rfm = calculate_rfm(df_clean)
rfm_segmented = segment_customers(rfm, n_clusters=4)
```

### Predicting New Customer Segments

```python
from src.rfm_analysis import predict_customer_segment

# Customer with: 30 days recency, 5 purchases, $3000 total spent
segment = predict_customer_segment(
    recency=30, 
    frequency=5, 
    monetary=3000,
    scaler=fitted_scaler,
    kmeans_model=trained_model
)
print(f"Predicted Segment: {segment}")
```

## 📁 Project Structure

```
customer-segmentation-kmeans/
│
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── notebooks/
│   └── customer_segmentation_kmeans.ipynb  # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning functions
│   └── rfm_analysis.py                  # RFM calculation & clustering
│
├── docs/
│   ├── PROJECT_REPORT.md               # Detailed project report
│   └── METHODOLOGY.md                  # Technical methodology
│
├── images/                             # Generated visualizations
│   ├── rfm_distributions.png
│   ├── cluster_evaluation.png
│   ├── cluster_analysis.png
│   └── cluster_3d_visualization.png
│
└── data/                               # Data directory (not tracked)
    └── .gitkeep
```

## 💼 Business Applications

### Segment-Specific Strategies

| Segment | Strategy | Tactics |
|---------|----------|---------|
| **Champions** | Retain & Reward | VIP programs, exclusive previews, referral programs |
| **At-Risk** | Re-engage | Win-back campaigns, personalized discounts, surveys |
| **Potential** | Develop | Upselling, cross-selling, loyalty programs |
| **New** | Onboard | Welcome sequences, product education, first-purchase incentives |

### Implementation Recommendations

1. **CRM Integration:** Export segments to marketing automation platforms
2. **Dynamic Segmentation:** Re-run analysis quarterly to capture changes
3. **A/B Testing:** Test strategies per segment to optimize ROI
4. **Lifecycle Marketing:** Design campaigns aligned with segment characteristics

## ⚠️ Limitations & Future Work

### Current Limitations

- **Temporal Scope:** Single year of data may not capture seasonal variations
- **Feature Set:** RFM alone may miss product preferences, demographics
- **Static Segments:** Point-in-time analysis; customer behavior evolves
- **Geographic Bias:** Predominantly UK customers

### Future Improvements

- [ ] Add product category preferences as clustering features
- [ ] Implement DBSCAN for density-based comparison
- [ ] Build real-time segmentation pipeline
- [ ] Add customer lifetime value (CLV) predictions
- [ ] Create interactive dashboard (Streamlit/Dash)
- [ ] Incorporate time-series analysis for trend detection

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [RFM Analysis - Wikipedia](https://en.wikipedia.org/wiki/RFM_(market_research))
- [K-Means Clustering - scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Online Retail Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)

---

<p align="center">
  Made with ❤️ for Data Science Portfolio
</p>
