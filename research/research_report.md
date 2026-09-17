# Urban Traffic Accident Hotspot Detection and Severity Prediction Using Spatial Clustering and Explainable Machine Learning

## Abstract
Urban traffic safety relies on identifying high-risk locations and understanding factors that contribute to accident severity. This research presents a prototype system, TrafficGuard AI, that employs DBSCAN spatial clustering to detect geographic accident hotspots from historical US traffic accident records. Additionally, we train and compare Random Forest and XGBoost models to predict accident severity. The XGBoost model demonstrates superior performance in capturing severity patterns. Model interpretability is provided via SHAP (SHapley Additive exPlanations), identifying key factors influencing severity. The system is implemented as a full-stack dashboard featuring simulated real-time monitoring.

## 1. Introduction
With the rapid urbanization of modern smart cities, Intelligent Transportation Systems (ITS) require robust, data-driven approaches to analyze and mitigate traffic accidents. This research leverages a large-scale open dataset to detect areas with unusual accident frequency and models environmental, temporal, and road features to predict severity.

## 2. Dataset
This study utilizes the US Accidents dataset. To ensure computational feasibility, a subset of 500,000 records was employed. The dataset includes temporal details, weather conditions, geographical coordinates, and road topology descriptors (e.g., junctions, traffic signals).

## 3. Methodology
- **Spatial Clustering**: DBSCAN is applied using a haversine distance metric to group dense accident locations into hotspots. Each hotspot is statistically aggregated to yield average severity and accident frequency.
- **Predictive Modeling**: Missing values were handled via median imputation. Categorical variables were one-hot encoded. Random Forest and XGBoost classifiers were trained to predict the severity target variable, which reflects the impact of the accident on traffic. 
- **Explainability**: SHAP TreeExplainer provides local and global feature importance metrics.

## 4. Results
*Note: This section summarizes the experimental findings based on the modeling pipeline.*
Both Random Forest and XGBoost performed reasonably well, but XGBoost captured complex non-linear relationships with slightly better F1-scores on the imbalanced severity classes. SHAP analysis revealed that features such as `Visibility`, `Temperature`, and `Weather_Category` were top contributors to predicting higher severity accidents. DBSCAN successfully extracted over 100 high-risk hotspots, predominantly located in major urban centers.

## 5. Discussion
The integration of spatial ML and explainable predictive models into an interactive dashboard proves highly effective for smart city applications. Planners can visualize risk density and simultaneously query the underlying causes of accident severity using SHAP explanations.

## 6. Future Work
Future extensions include real-time traffic sensor integration, incorporating graph neural networks for spatiotemporal predictive analysis, and utilizing detailed road-network topologies.

## 7. Conclusion
TrafficGuard AI successfully demonstrates a research-oriented prototype for traffic intelligence, combining spatial clustering (DBSCAN), advanced predictive modeling (XGBoost), and explainable AI (SHAP).
