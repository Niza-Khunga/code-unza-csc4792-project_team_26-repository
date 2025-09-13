# CSC-4792-Group-26
## 1. Business Understanding
### 1.1 Problem Statement
Zambian legislative documents are currently categorized and sorted manually. 
This Process is time_consuming, prone to human error, and can make it difficult for citizens, 
legal professionals and researchers to effeciently search for and retrieve specific types of legislation.

Our project aims to automate this classification process. 
By developing a machine learning model, we can accurately and efficiently 
classify zambian legislative documents(such as bills, acts, and statutory instruments) 
based on their content, thereby stramlining the retrieval of legal information. 
This will improve accessibility and reduce the time and effort required for manual sorting.

### 1.2 Business Objectives
The primary objective of this project is to improve access 
to and organisation of Zambian legislation. 
By implementing an automated classification system:

*Enhance legal research efficiency:* enabling users to quickly locate relevant legislation by category or type.
*Reduce manual workload:* decreasing the time and resources spent on sorting legal documents by at least 70%.
*Increase public access to information :* providing non-specialists, including journalists and citizens, 
with user-friendly tools to find relevant laws.

### 1.3 Data Mining Goals
To meet the business objectives, we need to:

- Build a multiclass text classification model to categorise Zambian legislative documents into predifined legal categories 
such as Constitutional law, Criminal law  and Commercial law.
- Employ text preprocessing techniques e.g tokenization, stopword removal, stemming.
- Use TF-IDF for feature extraction and evaluate advanced models such as BERT for semantic understanding.
- Evaluate model performance using metrics such as F1-score, precision and recall to ensure balanced classification performance.

### 1.4 Success Criteria
The success of this project will be evaluated based on both techinical performance and its 
usefulness to stakeholders in the legal sector.

From a technical perspective, the classification model should achieve an accuracy 
of at least 85% on the test data. additional performance metrics such as precision, 
recall, and F1-score will also be assessed to ensure the model is balanced and biased 
towards any specific catergory of legislation. From a business perspective, 
the model should significantly improve the efficiency of sorting and categorizing legal documents, 
making it easier for legal professional, researcher and government agencies to access releveant legislation. 
if the model can correctly categorize atleast 8 out of 10 new legal documents during user.
testing it will be considered successful. furthermore, success includes proper documentation, 
version control via github, an smooth intergration of the model into a usable interface or workflow.


## 2. Data Understanding

### 2.1 Overview of Dataset

The dataset contains a collecction of Zambian legislation documents, including Acts and Bills. Each row represnts a single piece of legislation, with columns for:
Title: The official title of the Act or Bill. Text: Full or partial text of the legislation category: the assigned Category for classification (e.g., Finance, Labour, Trade, Security, etc.).

Initial exploration helps identify structure, completeness and characteristics of the data

### 2.2 Data Exploration

Number of records: 60 Columns:Title, Text, Category Missing values: Some documents have missing or very short text Distribution of categories: Unequal representation; some categories are underrepresented Text length: High variability, indicating some Acts are very short, others very long


### 2.2 Summary of initial findings

Dataset size: 60rows x 7 columns Category distribution: Most documents are of Finance or Labour Acts. categories like Security and NGOs have got fewer examples Text completeness: A few Acts have missing text ot extremely short descriptions Length variation: Text lengths vary widely (Certain Acts have a lot of pages while some have a few paragraphs)

*Implications*:
Some preprocessing will be necessary (removing or padding short texts) Categories may need balancing for classification models visualizations help identify patterns in category distribution and text lengths

## 3 Data Preparation
### 3.1 Data Cleaning
Removed duplicate entries to avoid bias.
Dropped rows with missing or empty text fields.
Filtered out very short texts(<20 characters) since they don't provide meaniningful signals.
Standardized formatting (lowerecased text, stripped whitespace).

### 3.2 Feature Engineering
Added text_length feature to capture complexity of legislation.
Added word_count feature to measure verbosity.
Extracted year from legislation titles where available.

## DATA TRASNFORMATION 
Label Encoding: Converted Category into numerical codes for ML compatibility.
Text Vectorization: Used TF-IDF with a 5,000-term vocabulary and English stopword removal to represent legislative text numerically. 
Scaling: Standardized numerical features (text_length, word_count) to prevent them from overpowering text features.

### 3.4 Documentation & Rationale
Dropping missing/short entries improves data quality and prevents noise in the model.
Added features (text_length, word_count, year) because document structure often reflects category type (e.g., Finance Acts are typically shorter).
TF-IDF chosen over Bag-of-Words because it highlights distinctive terms and reduces the impact of common legal words like “Act” and “provide.”
Label encoding ensures categorical labels can be used in supervised learning.

## 4 Modeling 
In this section, we build and train machine learning models to classify Zambian legislative documents.  
We experiment with two baseline algorithms commonly used for text classification:  
- Logistic Regression  
- Naive Bayes
- Support Vector Machine (SVM)  

These models are chosen because they are effective with TF-IDF features and provide a solid baseline for text classification tasks.

### 4.1 Models Tested
We trained and evaluated three classification models using TF-IDF features:
Logistic Regression
Accuracy: 58%
Strong performance on majority categories (Finance, Environmental Law).
Failed to predict minority categories.

Naïve Bayes
Accuracy: 58%
Performed similarly to Logistic Regression.
Best suited for text data but still struggled due to small dataset size.

Support Vector Machine (SVM)
Accuracy: 42%
Lower performance overall.
Captured some majority categories but missed minority classes entirely.

### 4.2 Observations
- Both *Logistic Regression* and *Naive Bayes* achieved ~58% accuracy.  
- *Underrepresented categories* (NGOs, Security) were not predicted well.  
- More preprocessing (balancing categories, expanding dataset) is needed to improve performance.

## 5. Evaluation of Model  
### 5.1 Results  
- Best model: Logistic Regression  
- Accuracy: 58%  
- Macro Precision: 0.23428571428571426  
- Macro Recall: 0.4  
- Macro F1: 0.29545454545454547
  
  N:B However Logistic Regression and Naive Bayes had the same accuracy. We selected one.

### 5.2 Confusion Matrix  
| True \ Pred | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|-------------|---------|---------|---------|---------|---------|
| Class 1     | 4       | 0       | 0       | 0       | 0       |
| Class 2     | 0       | 3       | 0       | 0       | 0       |
| Class 3     | 1       | 1       | 0       | 0       | 0       |
| Class 4     | 2       | 0       | 0       | 0       | 0       |
| Class 5     | 0       | 1       | 0       | 0       | 0       |

### 5.3 Interpretation  
The model performs well on majority categories but poorly on minority ones.  
It does not yet meet the 85% accuracy success criterion.

## 6. Deployment  

- *Dataset Quality & Structure*  
  - Total of 60 legislative documents (Acts and Bills).  
  - Some documents had missing or very short text.  
  - Strong imbalance: Finance and Labour categories had more examples, while Security and NGO categories were underrepresented.  
  - Wide variation in text length across Acts

- **Classification Results**  
  - Finance and Labour Acts were classified most accurately.  
  - Categories such as Security and NGOs were rarely predicted correctly due to too few training examples.  
  - Logistic Regression and Naive Bayes both achieved ~58% accuracy, while SVM performed worse (42%).

- **Comparison to Success Criteria**  
  - Original target: ≥85% accuracy.  
  - Achieved: 58% accuracy.  
  - Although the target was not met, the results show that automated classification of Zambian legislation is **feasible** and would improve with a larger, more balanced dataset and advanced models (e.g., BERT).  

---

### 6.2 Deployment Plan  
- *User Interaction*  
  - The model can be integrated into a simple interface (web or desktop).  
  - Users paste or upload the text of a Bill/Act.  
  - The model predicts the most likely category (e.g. Finance, Security, Environmental, NGOs, Professional Regulation).
  

Musengwa Type here

- **Future Deployment Possibilities**  
  - Build a **REST API** that takes legislative text and returns the predicted category for integration into other systems.  
  - Integrate into the **UNZA Institutional Repository** or government legal archives to support better organization and discovery of legal documents.  
  - Extend into a **dashboard** where trends can be visualized (e.g., how many Finance Acts vs. Environmental Acts were passed in a given year).  

---

### 6.3 Example Deployment Function  
We implemented a simple function `fxn_predict_new_instance()` to simulate deployment.  
This function takes in a new legal text, transforms it using TF-IDF, and predicts the category using our final model.  
It also outputs a **confidence score** for the prediction.  

```python
def fxn_predict_new_instance(text):
    """
    Simulates deployment of the model.
    Takes raw legal text and returns both the predicted category 
    and the model's confidence score.
    """
    # Convert input text into TF-IDF features
    X_new = vectorizer.transform([text])

    # Predict numerical label
    prediction = best_model.predict(X_new)[0]

    # Get prediction probability (confidence score)
    if hasattr(best_model, "predict_proba"):
        confidence = best_model.predict_proba(X_new).max()
    else:
        confidence = None  

    # Convert numerical label back to category name
    category = le.inverse_transform([prediction])[0]

    return category, confidence






