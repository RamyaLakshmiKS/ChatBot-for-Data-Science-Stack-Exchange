# CAP5771 - Spring 2025 - Class Project

## Data Science Stack Exchange Dataset Analysis
**Project Overview:**  
This project involves analyzing the Data Science Stack Exchange dataset to gain insights into user engagement, post trends, and interaction patterns. The findings will help in developing a conversational agent for improved recommendations and responses.

**Repository Structure:**  
- **Data/**: Contains the dataset and processed CSV files.
  - `metadata.txt`: Metadata about the dataset.
  - `csv_output/`: Cleaned CSV files for analysis, including:
    - `Badges_cleaned.csv`
    - `Comments_cleaned.csv`
    - `Posts_cleaned.csv`
    - `Users_cleaned.csv`, etc.

- **Reports/**: Includes detailed milestone reports.
  - `Milestone1.pdf`: Covers data collection, preprocessing, and EDA.
  - `Milestone2.pdf`: Focuses on feature engineering, selection, and modeling.
  - `Milestone3.pdf`: Discusses the integration of models into the conversational agent.

- **Scripts/**: Contains code and resources for analysis and application development.
  - `Milestone-1.ipynb`: Jupyter notebook for Milestone 1.
  - `Milestone-2.ipynb`: Jupyter notebook for Milestone 2.
  - `Project PPT.pdf`: Presentation summarizing the project.
  - `app/`: Directory for the chatbot application.
    - `chatbot.py`: Main Streamlit app for the chatbot.
    - `ingest_data.py`: Script for data ingestion and FAISS index creation.
    - `requirements.txt`: Python dependencies for the app.
    - `data/`: Contains the cleaned dataset and FAISS index files.
    - `model/`: Includes the trained LightGBM model (`lgbm_model.pkl`).

For more details about the chatbot application, refer to the [app/README.md](Scripts/app/README.md) file.

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

## Milestone 3: Tool Development

**Summary:**

In Milestone 3, the focus was on developing a Retrieval-Augmented Generation (RAG)-based conversational agent to provide accurate and explainable responses to data science-related queries. The chatbot application was built using Streamlit and integrates the LightGBM model trained in Milestone 2 for predictive capabilities. The application leverages FAISS (Facebook AI Similarity Search) for efficient similarity search and retrieval of relevant posts from the dataset.

**Key Components in `app/` Directory:**

- `chatbot.py`: The main Streamlit application that serves as the user interface for the chatbot. It allows users to input queries and receive responses based on the RAG pipeline.
- `ingest_data.py`: A script to preprocess the dataset and create a FAISS index for efficient similarity search.
- `requirements.txt`: Lists all the Python dependencies required to run the chatbot application.
- `data/`:
  - `cleaned_stack_exchange_data.csv`: The cleaned dataset used for retrieval.
  - `faiss_index/`: Contains the FAISS index files (`index.faiss` and `index.pkl`) for similarity search.
- `model/`:
  - `lgbm_model.pkl`: The trained LightGBM model used for predictions.

**Features of the Chatbot Application:**

- **Query Understanding**: The chatbot processes user queries and retrieves relevant posts from the dataset using FAISS.
- **Predictive Insights**: Integrates the LightGBM model to predict the likelihood of a post receiving answers or engagement.
- **Explainability**: Provides SHAP-based explanations for the model's predictions, enhancing transparency and trust.
- **Interactive Interface**: Built with Streamlit, offering a user-friendly and responsive interface.

**Next Steps:**

1. **Deployment**:
   - Deploy the chatbot application on a cloud platform (e.g., AWS, Azure, or Heroku) for public access.
   - Ensure scalability and reliability of the application.

2. **Enhancements**:
   - Improve the RAG pipeline by incorporating additional datasets or fine-tuning the FAISS index.
   - Explore advanced NLP techniques (e.g., transformers) to enhance query understanding and response generation.

3. **User Testing**:
   - Conduct user testing to gather feedback on the chatbot's performance and usability.
   - Iterate on the design and functionality based on user feedback.

---

## Presentation and Demo Video
- **Presentation Link**: To be added
- **Demo Video Link**: To be added

---

## How to Use

1. **Dataset Preparation**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aneeshtickoo/data-science-stack-exchange/data?select=metadata.txt).
   - Place the dataset in the `Data/` directory.

2. **Run Analysis**:
   - Use the Jupyter notebooks in `Scripts/` for data preprocessing, EDA, and modeling.

3. **Chatbot Application**:
   - Navigate to `Scripts/app/`.
   - Install dependencies using `pip install -r requirements.txt`.
   - Run the chatbot with `streamlit run chatbot.py`.

---

