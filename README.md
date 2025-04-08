# CAP5771 - Spring 2025 - Class Project

## Data Science Stack Exchange Dataset Analysis
**Project Overview:**  
This project involves analyzing the Data Science Stack Exchange dataset to gain insights into user engagement, post trends, and interaction patterns. The findings will help in developing a conversational agent for improved recommendations and responses.

**Repository Structure:**  
- reports/ – Contains the submitted project report with detailed findings and analysis.  
- scripts/ – Includes the Python notebook used for data preprocessing, EDA, and visualization.

**Dataset:**  
The dataset used for this project is available on Kaggle:
[Data Science Stack Exchange](https://www.kaggle.com/datasets/aneeshtickoo/data-science-stack-exchange/data?select=metadata.txt)

Due to storage limitations, the dataset is not included in this repository. You can download it from Kaggle and place it in the appropriate directory before running the analysis.

---

## Milestone 1: Data Collection, Preprocessing, and Exploratory Data Analysis (EDA)

**Summary:**

In Milestone 1, I analyzed the Data Science Stack Exchange dataset to uncover insights about user behavior, post trends, and engagement, which will later improve a conversational agent. I collected the dataset from Kaggle (8 XML files + 1 TXT metadata file), converted it to CSV using Python’s `xml.etree.ElementTree`, and stored it locally. Preprocessing involved cleaning (e.g., dropping irrelevant columns like `ContentLicense`), handling missing values (e.g., filling NaN `UserId` with 0), and engineering features (e.g., extracting `Year`, `Month`, `Day` from `CreationDate`). 

EDA revealed key insights:
- **User Reputation**: Highly skewed distribution—most users have low reputation, but a few have very high scores (up to 25,908).
- **Post Trends**: Posts peaked in 2019, with a surge from 2017–2018, then declined in 2020–2021. Summer months (e.g., May, July) showed higher activity.
- **Tags**: Machine learning, deep learning, and Python were the most common topics.
- **Engagement**: Posts with higher votes got more answers/comments. View counts, answer counts, and favorites were also skewed, with outliers indicating highly popular posts.

Visualizations included histograms, bar charts, and scatter plots (e.g., ViewCount vs. Score) to reveal these trends, and outliers were identified using IQR and Z-score methods. Tools used included Python, Pandas, NumPy, Matplotlib, Seaborn, and SciPy. These findings set the stage for feature engineering and modeling in Milestone 2.

---

## Milestone 2: Feature Engineering, Feature Selection, and Data Modelling

### Tech Stack

**Programming & Libraries**
- **Python**: Core language for scripting, data manipulation, and model development.
- **Pandas & NumPy**: For advanced data manipulation, feature creation, and numerical computations.
- **Scikit-learn**: For feature selection (e.g., `SelectKBest`, `RFECV`), model training (e.g., Logistic Regression, Random Forest, SVC), and evaluation metrics (e.g., F1 score, AUC).
- **XGBoost**: For high-performance gradient boosting model implementation.
- **LightGBM**: For efficient and scalable gradient boosting, with SHAP support for interpretability.
- **SHAP (SHapley Additive exPlanations)**: To analyze feature importance and model explainability.
- **Matplotlib & Seaborn**: For visualizing feature distributions and model performance.

**Techniques & Tools**
- **Feature Engineering**: Created new features using Pandas.
- **Feature Selection**: Applied statistical methods and model-based techniques.
- **Data Modeling**: Implemented and tuned classification models to predict post outcomes, optimized via `GridSearchCV`.
- **SMOTE (Synthetic Minority Oversampling Technique)**: Addressed class imbalance in the dataset.

**Storage & Environment**
- **Data Format**: Processed CSV files from Milestone 1, stored locally.
- **Environment**: Jupyter Notebook (e.g., Google Colab) for development and experimentation.

### Model Performance and Selection

In Milestone 2, I trained five models—Logistic Regression, Random Forest, SVC, XGBoost, and LightGBM—to predict whether a post gets answered. 

**Model Selection Rationale**
LightGBM was selected as the best model due to its high test F1 (0.884), strong AUC (0.871), and practical advantages—fast training, scalability, and SHAP explainability—making it ideal for the Retrieval-Augmented Generation (RAG) pipeline in Milestone 3. Despite XGBoost’s slightly higher F1, LightGBM’s efficiency and interpretability made it the preferred choice.

**Top Three Models**
1. **XGBoost**: Highest test F1 and robust generalization.
2. **LightGBM**: Chosen for F1, AUC, and RAG compatibility.
3. **Random Forest**: Strong performer but less efficient.

### Next Steps
Milestone 2 successfully built and evaluated predictive models, with LightGBM selected for integration into the conversational agent. Milestone 3 will focus on developing the RAG-based conversational agent, leveraging the insights and models from the previous milestones to provide accurate and explainable responses to data science queries.

---

### How to Use

1. Navigate to the `reports/` folder to review the project documentation.
2. Check the `scripts/` folder for the Jupyter notebooks containing code and visualizations.
3. Download the dataset from the above link and place it in the working directory for analysis.

