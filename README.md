# Sentiment_Analysis
### **Project Overview & Purpose**
The project, implemented in **`Task_4.ipynb`**, focuses on building an end-to-end **Twitter Sentiment Analysis pipeline**. It processes raw tweet data, cleans and vectorizes text features, trains machine learning classifiers, and evaluates model performance to categorize tweets into four sentiment classes: **Positive**, **Negative**, **Neutral**, and **Irrelevant**.

---

### **Dataset Specifications**
* **Primary Source:** `twitter_training.csv`.
* **Columns:** **Tweet ID**, **Entity** (brand or game name), **Sentiment** label, and **Tweet content**.
* **Processed Dataset:** Saved as `twitter_training_renamed.csv` after mapping raw headers to human-readable column names.

---

### **Tech Stack & Libraries**
* **Data Manipulation:** `pandas`, `numpy`
* **Natural Language Processing (NLP):** `nltk`, `TextBlob`, `wordcloud`
* **Machine Learning (`scikit-learn`):** `LogisticRegression`, `MultinomialNB`, `LinearSVC`, `TfidfVectorizer`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Model Serialization:** `joblib`
* **Text Processing & Regex:** `re`, `string`

---

### **Pipeline Execution & Notebook Architecture**

1. **Imports & Setup:**
   * Downloads essential NLTK dependencies (`stopwords`, `punkt`, `wordnet`) and initializes libraries.

2. **Data Loading & Inspection:**
   * Ingests `twitter_training.csv`, renames columns, saves `twitter_training_renamed.csv`, and prints sentiment class distribution counts.

3. **Text Preprocessing Pipeline:**
   * **Basic Cleaning:** Converts text to strings, strips URLs, removes numeric digits, and cleans special characters.
   * **NLTK-Based Pipeline (v1):** Explored stopword filtering alongside `WordNetLemmatizer` and `PorterStemmer`.
   * **Optimized Pipeline (Final Version):** Replaced heavy stemming/lemmatization with a lighter regex cleaner and custom stopword list for improved execution speed.

4. **Feature Extraction (TF-IDF Vectorization):**
   * Configures `TfidfVectorizer` with:
     * English stopword removal (`stop_words='english'`)
     * Unigram and bigram extraction (`ngram_range=(1,2)`)
     * Sublinear TF scaling (`sublinear_tf=True`)
     * Frequency thresholds (`max_df=0.95`, `min_df=2`)

5. **Model Development & Training:**
   * **Logistic Regression:** Trained on a 70/30 stratified train/test split (`test_size=0.3`, `random_state=42`) as an initial exploratory baseline.
   * **Naive Bayes (`MultinomialNB`):** Evaluated on an 80/20 stratified split (`test_size=0.2`).
   * **Linear SVM (`LinearSVC`):** Trained on an 80/20 stratified split using balanced class weights (`class_weight='balanced'`).

6. **Evaluation & Visualization:**
   * Implements a reusable `evaluate_model()` function calculating overall accuracy, per-class precision, recall, and F1-scores.
   * Plots confusion matrix heatmaps for each model.
   * Generates a sentiment class distribution bar chart using Seaborn's `viridis` palette.
   * Creates individual WordClouds for each sentiment category (**Positive**, **Negative**, **Neutral**, **Irrelevant**).

7. **Artifact Persistence:**
   * Exports the final trained classifiers and the fitted TF-IDF vectorizer to disk as `sentiment_models.joblib`.

---

💡 *Would you like me to create a detailed report or slide deck presenting the comparative results and performance metrics of these models?*
