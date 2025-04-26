# CAP5771 - Spring 2025 - Class Project

## Data Science Stack Exchange Dataset Analysis

This project involves analyzing the Data Science Stack Exchange dataset to gain insights into user engagement, post trends, and interaction patterns. The findings are used to develop a conversational agent for improved recommendations and responses.

---

## Milestone 1: Data Collection, Preprocessing, and Exploratory Data Analysis (EDA)

### Summary
In Milestone 1, the Data Science Stack Exchange dataset was analyzed to uncover insights about user behavior, post trends, and engagement. The dataset was collected from Kaggle (8 XML files + 1 TXT metadata file), converted to CSV using Python’s `xml.etree.ElementTree`, and stored locally. Preprocessing involved:
- Cleaning irrelevant columns (e.g., `ContentLicense`).
- Handling missing values (e.g., filling NaN `UserId` with 0).
- Engineering features (e.g., extracting `Year`, `Month`, `Day` from `CreationDate`).

### Key Insights
- **User Reputation**: Highly skewed distribution—most users have low reputation, but a few have very high scores (up to 25,908).
- **Post Trends**: Posts peaked in 2019, with a surge from 2017–2018, then declined in 2020–2021. Summer months (e.g., May, July) showed higher activity.
- **Tags**: Machine learning, deep learning, and Python were the most common topics.
- **Engagement**: Posts with higher votes got more answers/comments. View counts, answer counts, and favorites were also skewed, with outliers indicating highly popular posts.

### Tools Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, SciPy.
- **Visualization**: Histograms, bar charts, scatter plots.
- **Outlier Detection**: IQR and Z-score methods.

---

## Milestone 2: Feature Engineering, Feature Selection, and Data Modeling

### Objective
The goal of this milestone was to build and evaluate machine learning models to classify posts from the cleaned Data Science Stack Exchange dataset, determining whether a post was answered or not. This work lays the groundwork for Milestone 3, where the trained model is integrated into a Retrieval-Augmented Generation (RAG)-based conversational agent.

### Key Processes
1. **Feature Engineering**:
   - Created features such as `body_length`, `title_length`, `tag_count`, `primary_tag_encoded`, and `Answered` (target variable).
   - Merged additional features from other datasets (e.g., user metrics, comment-related features, edit counts).

2. **Feature Selection**:
   - Selected 15 features for modeling, including post-specific, user-specific, and aggregated features.

3. **Data Preparation**:
   - Applied train-test split (80%-20%) with stratification.
   - Used SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.

4. **Data Modeling**:
   - Trained and tuned five machine learning models using GridSearchCV:
     - Logistic Regression
     - Random Forest
     - Support Vector Classifier (SVC)
     - XGBoost
     - LightGBM
   - Evaluated models using metrics such as F1 Score, AUC, and SHAP explainability.

### Model Selection
- **LightGBM** was selected as the best model due to its high test F1 (0.884), strong AUC (0.871), and practical advantages like fast training, scalability, and SHAP explainability.

### Tools Used
- **Python Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP.
- **Visualization**: Matplotlib, Seaborn.
- **Evaluation Metrics**: F1 Score, AUC, Confusion Matrix.

---

## Milestone 3: Tool Development (Implementation of conversational agent)

### Objective
The final milestone focused on developing a Retrieval-Augmented Generation (RAG)-based conversational agent to provide accurate and explainable responses to data science-related queries. The chatbot application was built using Streamlit and integrates the LightGBM model trained in Milestone 2 for predictive capabilities.

### Key Components
1. **Chatbot Application**:
   - Built using Streamlit for an interactive user interface.
   - Processes user queries and retrieves relevant posts from the dataset using FAISS.
   - Generates responses using Google Gemini LLM.

The chatbot leverages the Google Gemini LLM for generating conversational responses. The LLM is integrated into the Retrieval-Augmented Generation (RAG) pipeline, where it:
- Processes user queries and reformulates them for better context understanding.
- Generates concise and contextually relevant answers based on retrieved documents.
- Enhances the chatbot's ability to provide accurate and human-like responses.

2. **Integration of LightGBM Model**:
   - Predicts the likelihood of a post receiving answers or engagement.
   - Ranks answers based on relevance and quality.

3. **Explainability**:
   - Provides SHAP-based explanations for the model's predictions, enhancing transparency and trust.

4. **Data Ingestion**:
   - Preprocessed the dataset and created a FAISS index for efficient similarity search.

### Tools Used
- **Streamlit**: For building the chatbot interface.
- **FAISS**: For similarity search and document retrieval.
- **Google Gemini LLM**: For generating conversational responses.
- **LightGBM**: For predictive insights (Trained Model)

### Next Steps
1. **Deployment**:
   - Deploy the chatbot application for Public access.
   - Ensure scalability and reliability of the application.

2. **Enhancements**:
   - Improve the RAG pipeline by incorporating additional datasets or fine-tuning the FAISS index.
   - Explore advanced NLP techniques (e.g., transformers) to enhance query understanding and response generation.
   - Expand the UI with more helpful features in later versions

3. **User Testing**:
   - Conduct user testing to gather feedback on the chatbot's performance and usability.
   - Iterate on the design and functionality based on user feedback.

---

## Presentation and Tool Demo

- **Presentation Video**: [Link to Presentation Video](https://uflorida-my.sharepoint.com/:v:/g/personal/ra_kuppasundarar_ufl_edu/ESLMM7JRBAlAr_SRTYHCKj0BE4IgW48ZHEWDAMT8xmUoyQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=6DIUqH)
- **Tool Demo**: [Link to Tool Demo](https://uflorida-my.sharepoint.com/:v:/r/personal/ra_kuppasundarar_ufl_edu/Documents/Ids/Videos/Demo.mp4?csf=1&web=1&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=hIrqQc)

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

## Declaration of LLM Use

In this project, I used ChatGPT/Gemini for the following purposes, ensuring transparency and adherence to the regulations:

1. Brainstorming and Idea Refinement  
Prompt Used:  
*"Suggest 5 potential use cases"*  

How Output Was Used:  
- The LLM generated a list of ideas, which I evaluated for relevance and feasibility. I selected one use case and refined it further based on my Project.

2. Understanding Technical Documentation  
Prompt Used:  
*"Summarize the key functions of library/tool name from its official documentation in simple terms."*  
*"Explain how (e.g.,embeddings) works based on the docs at their [website](https://python.langchain.com/api_reference/huggingface/index.html)."*  

How Output Was Used:  
- The LLM helped me quickly grasp complex documentation by providing concise summaries. I verified all information against the original docs before implementation.

3. Debugging Code  
Prompt Used:  
*"Why does this Python code throw an error"*

How Output Was Used:  
- The LLM identified a missing null-check, which I verified and corrected. I tested the fix and documented the changes in my code comments.

4. Sentence Rephrasing & Proofreading  
Prompt Used:  
*"Rephrase this sentence for better clarity while keeping the original meaning"*  
*"Check this paragraph for grammatical errors and suggest improvements."*

How Output Was Used:  
- The LLM provided alternative phrasings for sentences that were unclear or repetitive. I reviewed each suggestion and only used those that improved readability without altering the intended meaning.
- Alongside Grammarly (for grammar/spelling checks), the LLM helped refine awkward phrasing, but all final edits were manually approved by me.

5. Template for Structure  
Prompt Used:  
*"Provide an outline for a project report including standard sections mentioned in rubrics."*  
*"How to make my project report better"*  
*"How to prepare better README. Show me examples of different README files"*

How Output Was Used:  
- I adapted the LLM’s generic template to fit my project’s unique needs, adding/removing sections as required.

---

## Repository Owner

- **Name**: Ramya Lakshmi Kuppa Sundararajan
- **Email**: ra.kuppasundarar@ufl.edu
- **LinkedIn**: [Profile](https://www.linkedin.com/in/ramyalakshmiks/)