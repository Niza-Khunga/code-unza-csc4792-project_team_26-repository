# CSC-4792-Group-26
Business Understanding
# 1.1 Problem Statement
Zambian legislative documents are currently categorized and sorted manually. 
This Process is time_consuming, prone to human error, and can make it difficult for citizens, 
legal professionals and researchers to effeciently search for and retrieve specific types of legislation.

Our project aims to automate this classification process. 
By developing a machine learning model, we can accurately and efficiently 
classify zambian legislative documents(such as bills, acts, and statutory instruments) 
based on their content, thereby stramlining the retrieval of legal information. 
This will improve accessibility and reduce the time and effort required for manual sorting.

# 1.2 Business Objectives
The primary objective of this project is to improve access 
to and organisation of Zambian legislation. 
By implementing an automated classification system:

*Enhance legal research efficiency:* enabling users to quickly locate relevant legislation by category or type.
*Reduce manual workload:* decreasing the time and resources spent on sorting legal documents by at least 70%.
*Increase public access to information :* providing non-specialists, including journalists and citizens, 
with user-friendly tools to find relevant laws.

# 1.3 Data Mining Goals
To achive the business objectives we need to:

-Build a multiclass text classification model to categorise Zambian legislative documents into predifined legal categories 
such as Constitutional law, Criminal law  and Commercial law.
-Employ text preprocessing techniques e.g tokenization, stopword removal, stemming.
-Use TF-IDF for feature extraction and evaluate advanced models such as BERT for semantic understanding.
-Evaluate model performance using metrics such as F1-score, precision and recall to ensure balanced classification performance.

# 1.4 Success Criteria
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
version control via github, an smooth intergration of the model into a usable interface or workflow. [BU] tag
