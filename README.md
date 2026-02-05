# Customer Segmentation Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-important.svg)

A production-ready customer segmentation project using unsupervised K-Means clustering on the Online Retail II dataset. This project demonstrates RFM (Recency, Frequency, Monetary) analysis to identify distinct customer groups for targeted marketing and retention strategies.

## Overview

This is a comprehensive data science project that applies machine learning techniques to segment customers based on their purchasing behavior. Using K-Means clustering on transaction data, we identify customer personas automatically.

### Key Objectives:
- Understand customer purchasing patterns
- - Identify distinct customer groups (segments)
  - - Enable targeted marketing campaigns
    - - Improve customer retention strategies
      - - Support business decision-making
       
        - ## Features
       
        - - **RFM Analysis**: Recency, Frequency, and Monetary analysis
          - - **K-Means Clustering**: Unsupervised segmentation with optimized clusters
            - - **Elbow Method**: Systematic cluster optimization
              - - **Silhouette Score**: Clustering quality evaluation
                - - **Professional Documentation**: Complete methodology and insights
                 
                  - ## Dataset
                 
                  - **Source**: [Kaggle - Online Retail II Dataset](https://www.kaggle.com/datasets/minalchoudhary/online-retail-dataset)
                 
                  - - **Size**: 541,909 transactions
                    - - **Period**: 2010-2011
                      - - **Countries**: Multiple
                        - - **Key Variables**: InvoiceNo, StockCode, Quantity, UnitPrice, InvoiceDate, CustomerID, Country
                         
                          - ## Quick Start
                         
                          - ### Installation
                          - ```bash
                            git clone https://github.com/tahamohmadf19-dev/customer-segmentation-kmeans.git
                            cd customer-segmentation-kmeans

                            python -m venv env
                            source env/bin/activate  # Windows: env\Scripts\activate

                            pip install -r requirements.txt
                            ```

                            ### Run Analysis
                            ```bash
                            jupyter notebook notebooks/customer_segmentation_kmeans.ipynb
                            ```

                            ## Project Structure

                            ```
                            ├── notebooks/
                            │   └── customer_segmentation_kmeans.ipynb
                            ├── src/
                            │   ├── data_preprocessing.py
                            │   └── rfm_analysis.py
                            ├── docs/
                            │   ├── PROJECT_REPORT.md
                            │   ├── METHODOLOGY.md
                            │   └── RESULTS_INTERPRETATION.md
                            ├── requirements.txt
                            ├── LICENSE
                            └── README.md
                            ```

                            ## Results

                            **Optimal Clusters**: 3-4 segments
                            **Silhouette Score**: ~0.40-0.50
                            **Business Impact**: Clear customer personas for targeted strategies

                            ## Key Findings

                            ### Customer Segments:
                            1. **High-Value Customers**: Premium buyers with frequent purchases
                            2. 2. **Mid-Range Customers**: Regular customers with stable patterns
                               3. 3. **Low-Engagement Customers**: Occasional buyers at churn risk
                                 
                                  4. ## Evaluation Metrics
                                 
                                  5. **Why No Train/Test Split?**
                                  6. This is unsupervised learning - we have no labels. Instead, we use:
                                  7. - Elbow Method (WCSS visualization)
                                     - - Silhouette Score (cluster separation quality)
                                       - - Within-Cluster Sum of Squares (tightness)
                                         - - Visual inspection of cluster distributions
                                          
                                           - ## Usage
                                          
                                           - See [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for detailed findings
                                           - See [METHODOLOGY.md](docs/METHODOLOGY.md) for technical details
                                          
                                           - ## Future Improvements
                                          
                                           - - Include product category features
                                             - - Try alternative algorithms (DBSCAN, Hierarchical Clustering)
                                               - - Build predictive churn models
                                                 - - Implement temporal analysis
                                                   - - Develop automated re-segmentation pipelines
                                                    
                                                     - ## License
                                                    
                                                     - MIT License - see [LICENSE](LICENSE) for details
                                                    
                                                     - ## Contact
                                                    
                                                     - **Author**: Mohmad Taha
                                                     - **GitHub**: [@tahamohmadf19-dev](https://github.com/tahamohmadf19-dev)
                                                     - **Email**: tahamohmadf19@gmail.com
                                                    
                                                     - ---
                                                     Last Updated: February 2026
