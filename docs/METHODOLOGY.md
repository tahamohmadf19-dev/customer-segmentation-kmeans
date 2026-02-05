# Technical Methodology: K-Means Customer Segmentation

## 1. Algorithm Overview

### K-Means Clustering
K-Means is an unsupervised machine learning algorithm that partitions data into k clusters by minimizing within-cluster variance.

**Key Characteristics:**
- Unsupervised learning (no labeled data required)
- - Partitioning algorithm (hard assignment to clusters)
  - - Scalable and efficient
    - - Works well with numerical features
     
      - ---

      ## 2. Mathematical Foundation

      ### Objective Function
      Minimize the Within-Cluster Sum of Squares (WCSS):

      ```
      WCSS = Σ Σ ||xi - μj||²
      ```

      Where:
      - xi = data point i
      - - μj = centroid of cluster j
        - - k = number of clusters
         
          - ### Algorithm Steps
         
          - **Step 1: Initialization**
          - - Select k initial centroids randomly or using k-means++
            - - k-means++ improves convergence speed and result quality
             
              - **Step 2: Assignment**
              - For each iteration:
              - - Assign each point to nearest centroid (min Euclidean distance)
                - - Formula: C(i) = argmin ||xi - μj||²
                 
                  - **Step 3: Update**
                  - - Recalculate centroid as mean of assigned points
                    - - μj = (1/|Cj|) Σ xi for all xi in cluster j
                     
                      - **Step 4: Convergence**
                      - - Repeat steps 2-3 until centroids stabilize
                        - - Convergence criteria: centroid movement < threshold or max iterations reached
                         
                          - ---

                          ## 3. Implementation Details

                          ### Feature Normalization
                          Essential step to prevent features with large scales from dominating clustering.

                          **StandardScaler (Z-score normalization):**
                          ```python
                          x_normalized = (x - mean) / std_dev
                          ```

                          **Why normalize?**
                          - K-Means uses Euclidean distance
                          - - Features with different scales affect distance calculation
                            - - Recency (0-365 days) vs Monetary ($0-100,000) without normalization would be dominated by Monetary
                             
                              - ### Distance Metric
                              - **Euclidean Distance:**
                              - ```
                                d(xi, μj) = sqrt(Σ (xi - μj)²)
                                ```

                                This is the default and most common distance metric for K-Means.

                                ---

                                ## 4. RFM Feature Engineering

                                ### Recency (R)
                                **Definition**: Number of days since customer's most recent purchase

                                **Calculation:**
                                ```python
                                analysis_date = df['InvoiceDate'].max()
                                recency = (analysis_date - customer_last_purchase_date).days
                                ```

                                **Interpretation:**
                                - Lower value = More recent engagement
                                - - Higher value = Risk of churn
                                  - - Range: 0-365+ days
                                   
                                    - ### Frequency (F)
                                    - **Definition**: Number of distinct transactions per customer
                                   
                                    - **Calculation:**
                                    - ```python
                                      frequency = customer_df.groupby('CustomerID')['InvoiceNo'].nunique()
                                      ```

                                      **Interpretation:**
                                      - Higher value = Loyal customer
                                      - - Count of unique invoices/transactions
                                        - - Range: 1-100+ transactions
                                         
                                          - ### Monetary (M)
                                          - **Definition**: Total spending amount per customer
                                         
                                          - **Calculation:**
                                          - ```python
                                            monetary = customer_df.groupby('CustomerID')['Amount'].sum()
                                            where Amount = Quantity * UnitPrice
                                            ```

                                            **Interpretation:**
                                            - Higher value = More valuable customer
                                            - - Total revenue generated
                                              - - Range: $0-$100,000+
                                               
                                                - ---

                                                ## 5. Optimal Cluster Determination

                                                ### Elbow Method

                                                **Process:**
                                                1. Fit K-Means for k = 1, 2, 3, 4, 5, ...
                                                2. 2. Calculate WCSS for each k
                                                   3. 3. Plot WCSS vs k
                                                      4. 4. Identify "elbow" point where improvement diminishes
                                                        
                                                         5. **Elbow Characteristic:**
                                                         6. - Sharp decrease: Model improves significantly
                                                            - - Elbow point: Optimal k selected
                                                              - - Gradual decrease: Diminishing returns
                                                               
                                                                - **Example Results:**
                                                                - ```
                                                                  k=2: WCSS = 2000
                                                                  k=3: WCSS = 1200  ← Significant improvement
                                                                  k=4: WCSS = 1000  ← Smaller improvement (elbow)
                                                                  k=5: WCSS = 950   ← Minimal improvement
                                                                  ```

                                                                  Selected k=3 or k=4 based on business context.

                                                                  ### Silhouette Score

                                                                  **Definition**: Measure of cluster cohesion and separation per sample

                                                                  **Calculation:**
                                                                  ```
                                                                  s(i) = (b(i) - a(i)) / max(a(i), b(i))

                                                                  where:
                                                                  a(i) = mean distance to other points in same cluster
                                                                  b(i) = mean distance to nearest cluster
                                                                  ```

                                                                  **Interpretation:**
                                                                  - Value range: -1 to +1
                                                                  - - Close to +1: Well-clustered, good separation
                                                                    - - Close to 0: On cluster boundary
                                                                      - - Negative: Possibly assigned to wrong cluster
                                                                       
                                                                        - **Score Guidelines:**
                                                                        - - > 0.5: Strong structure
                                                                            > - 0.4-0.5: Reasonable structure
                                                                            > - - 0.3-0.4: Weak structure
                                                                            >   - - < 0.3: Very weak clustering
                                                                            >    
                                                                            >     - ---
                                                                            >
                                                                            > ## 6. Data Preprocessing Steps
                                                                            >
                                                                            > ### 1. Data Loading & Exploration
                                                                            > ```python
                                                                            > df = pd.read_csv('online_retail_II.csv')
                                                                            > df.info()      # Data types and missing values
                                                                            > df.describe()  # Statistical summary
                                                                            > df.duplicated().sum()  # Check for duplicates
                                                                            > ```
                                                                            >
                                                                            > ### 2. Handling Missing Values
                                                                            > ```python
                                                                            > # Remove records with missing CustomerID (required for segmentation)
                                                                            > df = df[df['CustomerID'].notna()]
                                                                            >
                                                                            > # Check for missing in other columns
                                                                            > df.isnull().sum()
                                                                            > ```
                                                                            >
                                                                            > ### 3. Data Cleaning
                                                                            > ```python
                                                                            > # Remove cancelled orders (negative quantities)
                                                                            > df = df[df['Quantity'] > 0]
                                                                            >
                                                                            > # Remove zero-price items
                                                                            > df = df[df['UnitPrice'] > 0]
                                                                            >
                                                                            > # Check for duplicates
                                                                            > df = df.drop_duplicates()
                                                                            > ```
                                                                            >
                                                                            > ### 4. Feature Engineering
                                                                            > ```python
                                                                            > # Calculate amount per transaction
                                                                            > df['Amount'] = df['Quantity'] * df['UnitPrice']
                                                                            >
                                                                            > # Convert date to datetime
                                                                            > df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                                                                            > ```
                                                                            >
                                                                            > ### 5. RFM Aggregation
                                                                            > ```python
                                                                            > analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
                                                                            >
                                                                            > rfm_df = df.groupby('CustomerID').agg({
                                                                            >     'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
                                                                            >     'InvoiceNo': 'nunique',  # Frequency
                                                                            >     'Amount': 'sum'  # Monetary
                                                                            > }).rename(columns={
                                                                            >     'InvoiceDate': 'Recency',
                                                                            >     'InvoiceNo': 'Frequency',
                                                                            >     'Amount': 'Monetary'
                                                                            > })
                                                                            > ```
                                                                            >
                                                                            > ### 6. Outlier Detection
                                                                            > ```python
                                                                            > # Using Interquartile Range (IQR)
                                                                            > Q1 = rfm_df.quantile(0.25)
                                                                            > Q3 = rfm_df.quantile(0.75)
                                                                            > IQR = Q3 - Q1
                                                                            >
                                                                            > outliers = ((rfm_df < (Q1 - 1.5*IQR)) | (rfm_df > (Q3 + 1.5*IQR))).sum()
                                                                            >
                                                                            > # Decision: Cap or remove based on business context
                                                                            > rfm_df = rfm_df[(rfm_df >= Q1 - 1.5*IQR) & (rfm_df <= Q3 + 1.5*IQR)]
                                                                            > ```
                                                                            >
                                                                            > ### 7. Feature Normalization
                                                                            > ```python
                                                                            > from sklearn.preprocessing import StandardScaler
                                                                            >
                                                                            > scaler = StandardScaler()
                                                                            > rfm_scaled = scaler.fit_transform(rfm_df)
                                                                            >
                                                                            > # Create DataFrame with normalized values
                                                                            > rfm_normalized = pd.DataFrame(
                                                                            >     rfm_scaled,
                                                                            >     columns=rfm_df.columns,
                                                                            >     index=rfm_df.index
                                                                            > )
                                                                            > ```
                                                                            >
                                                                            > ---
                                                                            >
                                                                            > ## 7. Model Training & Evaluation
                                                                            >
                                                                            > ### K-Means Fitting
                                                                            > ```python
                                                                            > from sklearn.cluster import KMeans
                                                                            > from sklearn.metrics import silhouette_score
                                                                            >
                                                                            > # Determine optimal k
                                                                            > inertias = []
                                                                            > silhouette_scores = []
                                                                            >
                                                                            > for k in range(2, 6):
                                                                            >     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                                                            >     kmeans.fit(rfm_scaled)
                                                                            >     inertias.append(kmeans.inertia_)
                                                                            >     silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
                                                                            >
                                                                            > # Plot elbow curve
                                                                            > plt.plot(range(2, 6), inertias, 'bo-')
                                                                            > plt.xlabel('Number of Clusters (k)')
                                                                            > plt.ylabel('Inertia (WCSS)')
                                                                            > plt.title('Elbow Method')
                                                                            > plt.show()
                                                                            > ```
                                                                            >
                                                                            > ### Final Model Training
                                                                            > ```python
                                                                            > # Fit final model with selected k
                                                                            > final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                                                                            > rfm_df['Cluster'] = final_kmeans.fit_predict(rfm_scaled)
                                                                            > ```
                                                                            >
                                                                            > ### Evaluation Metrics
                                                                            > ```python
                                                                            > from sklearn.metrics import silhouette_score, davies_bouldin_score
                                                                            >
                                                                            > # Silhouette Score
                                                                            > sil_score = silhouette_score(rfm_scaled, final_kmeans.labels_)
                                                                            > print(f'Silhouette Score: {sil_score:.3f}')
                                                                            >
                                                                            > # Davies-Bouldin Index (lower is better)
                                                                            > db_score = davies_bouldin_score(rfm_scaled, final_kmeans.labels_)
                                                                            > print(f'Davies-Bouldin Index: {db_score:.3f}')
                                                                            >
                                                                            > # Cluster sizes
                                                                            > print(rfm_df['Cluster'].value_counts().sort_index())
                                                                            > ```
                                                                            >
                                                                            > ---
                                                                            >
                                                                            > ## 8. Cluster Interpretation
                                                                            >
                                                                            > ### Centroid Analysis
                                                                            > ```python
                                                                            > # Get centroids in original scale
                                                                            > centroids_scaled = final_kmeans.cluster_centers_
                                                                            > centroids_original = scaler.inverse_transform(centroids_scaled)
                                                                            >
                                                                            > centroid_df = pd.DataFrame(
                                                                            >     centroids_original,
                                                                            >     columns=['Recency', 'Frequency', 'Monetary']
                                                                            > )
                                                                            > print(centroid_df)
                                                                            > ```
                                                                            >
                                                                            > ### Cluster Profiling
                                                                            > ```python
                                                                            > # Analyze cluster characteristics
                                                                            > for cluster_id in range(3):
                                                                            >     cluster_data = rfm_df[rfm_df['Cluster'] == cluster_id]
                                                                            >     print(f'\nCluster {cluster_id} Statistics:')
                                                                            >     print(cluster_data[['Recency', 'Frequency', 'Monetary']].describe())
                                                                            > ```
                                                                            >
                                                                            > ---
                                                                            >
                                                                            > ## 9. Hyperparameter Tuning
                                                                            >
                                                                            > ### Important Parameters
                                                                            >
                                                                            > **n_clusters (k)**
                                                                            > - Determined by Elbow Method and domain knowledge
                                                                            > - - Typical range: 3-5 clusters
                                                                            >   - - Balance interpretability vs. granularity
                                                                            >    
                                                                            >     - **init**
                                                                            >     - - 'k-means++': Smart initialization (recommended)
                                                                            >       - - 'random': Random initialization
                                                                            >         - - k-means++ typically converges faster and better quality
                                                                            >          
                                                                            >           - **n_init**
                                                                            >           - - Number of times to run algorithm with different initializations
                                                                            >             - - Default: 10 (older scikit-learn: 10, new: 10)
                                                                            >               - - Higher values ensure better global optimum
                                                                            >                
                                                                            >                 - **max_iter**
                                                                            >                 - - Maximum iterations for convergence
                                                                            >                   - - Default: 300
                                                                            >                     - - Increase if algorithm doesn't converge
                                                                            >                      
                                                                            >                       - **random_state**
                                                                            >                       - - Seed for reproducibility
                                                                            >                         - - Set to fixed value for consistency
                                                                            >                           - - Recommended: 42
                                                                            >                            
                                                                            >                             - ### Example Configuration
                                                                            >                             - ```python
                                                                            >                               kmeans = KMeans(
                                                                            >                                   n_clusters=3,
                                                                            >                                   init='k-means++',
                                                                            >                                   n_init=10,
                                                                            >                                   max_iter=300,
                                                                            >                                   random_state=42,
                                                                            >                                   algorithm='lloyd'  # or 'elkan'
                                                                            >                               )
                                                                            >                               ```
                                                                            >
                                                                            > ---
                                                                            >
                                                                            > ## 10. Why No Train/Test Split?
                                                                            >
                                                                            > ### Unsupervised Learning Context
                                                                            > This is an **unsupervised learning** problem because:
                                                                            > 1. **No labels**: We don't have ground truth cluster assignments
                                                                            > 2. 2. **Exploration**: Goal is to discover patterns, not predict
                                                                            >    3. 3. **Descriptive**: Segmentation is inherently descriptive, not predictive
                                                                            >      
                                                                            >       4. ### Evaluation Approach Instead
                                                                            >       5. Instead of train/test split, we use:
                                                                            >      
                                                                            >       6. 1. **Elbow Method**: Optimizes model selection
                                                                            >          2. 2. **Silhouette Score**: Validates cluster quality without labels
                                                                            >             3. 3. **Visual Inspection**: Scatter plots confirm interpretability
                                                                            >                4. 4. **Business Validation**: Do clusters align with business intuition?
                                                                            >                  
                                                                            >                   5. ### Why Not Train/Test?
                                                                            >                   6. - No labels to evaluate predictions
                                                                            >                      - - Both would see same data anyway
                                                                            >                        - - Would artificially reduce training data
                                                                            >                          - - Unsupervised metrics are more appropriate
                                                                            >                           
                                                                            >                            - ---
                                                                            >
                                                                            > ## 11. Common Pitfalls & Solutions
                                                                            >
                                                                            > ### Pitfall 1: Choosing Wrong k
                                                                            > **Problem**: Too few clusters lose detail; too many are uninterpretable
                                                                            > **Solution**: Use Elbow Method + business context
                                                                            >
                                                                            > ### Pitfall 2: Forgetting to Normalize
                                                                            > **Problem**: Features with large ranges dominate clustering
                                                                            > **Solution**: Always use StandardScaler before K-Means
                                                                            >
                                                                            > ### Pitfall 3: Local Minima
                                                                            > **Problem**: Algorithm converges to suboptimal solution
                                                                            > **Solution**: Use n_init > 1 or k-means++
                                                                            >
                                                                            > ### Pitfall 4: Ignoring Outliers
                                                                            > **Problem**: Extreme values skew cluster means
                                                                            > **Solution**: Detect and handle outliers before clustering
                                                                            >
                                                                            > ### Pitfall 5: Non-spherical Clusters
                                                                            > **Problem**: K-Means assumes spherical clusters
                                                                            > **Solution**: Consider DBSCAN or Hierarchical Clustering for other shapes
                                                                            >
                                                                            > ---
                                                                            >
                                                                            > ## 12. References
                                                                            >
                                                                            > - scikit-learn K-Means Documentation
                                                                            > - - "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
                                                                            >   - - Kaggle K-Means Best Practices
                                                                            >     - - Unsupervised Learning Literature
                                                                            >      
                                                                            >       - ---
                                                                            >
                                                                            > **Last Updated**: February 2026
                                                                            > **Status**: Complete
