# Detecting Fraudulent Job Postings with Support Vector Classification

This project investigates the effectiveness of Support Vector Classification (SVC) models in identifying fraudulent job postings using text and categorical features. We compare two different approaches to preprocessing text data—one that treats each descriptive field independently and another that combines all fields into a single input—using a dataset of 18,000 job postings from the Employment Scam Aegean Dataset (EMSCAD).

## Project Overview

Job posting platforms like LinkedIn and Handshake have facilitated job searches but also opened the door to scammers. This project evaluates whether SVC models, known for their ability to handle high-dimensional data, can accurately detect fake postings. We use TF-IDF vectorization to transform textual data into features and compare two modeling pipelines:

- **Separate-Vectorizer Model**: Each job description field (e.g., title, company profile, requirements) is vectorized separately to retain field-specific semantic distinctions.
- **Combined-Text Model**: All text fields are concatenated and vectorized together, creating a unified feature space.

Each model is evaluated on classification performance using precision, recall, F1, and ROC-AUC scores after hyperparameter tuning and threshold optimization.

## Resources
- Editor Used: Jupyter Notebook / Python
- Libraries: `scikit-learn`, `nltk`, `matplotlib`, `pandas`, `seaborn`

### scikit-learn
Used for building the SVC pipelines, performing TF-IDF vectorization, one-hot encoding, and conducting hyperparameter tuning via cross-validation.

### nltk
Employed for preprocessing steps such as stopword removal, which helps clean textual fields before vectorization.

### Visualization Tools
Confusion matrices and precision-recall curves were used to assess classifier performance. A word cloud was generated to illustrate which terms contributed most strongly to the model’s decision-making.

## Key Findings

- **Performance**: Both models achieved high precision and recall. The combined-text model slightly outperformed the separate-vectorizer model on the F1 metric after threshold tuning (0.941 vs. 0.90).
- **Statistical Significance**: Paired t-tests showed no significant difference in out-of-sample F1 scores but revealed meaningful differences in raw prediction probabilities.
- **Interpretability**: While the separate-vectorizer model allows for field-specific interpretation, the combined-text model offers streamlined preprocessing and slightly better performance.
- **Limitations**: The dataset was highly imbalanced (4.5% fraudulent), and several features (e.g., salary) were excluded due to missingness.

## Future Directions

Future work could:
- Use synthetic oversampling or downsampling to balance the dataset.
- Incorporate more sophisticated textual embeddings (e.g., BERT).
- Reintroduce excluded features via imputation or structured collection.

## Bibliography

If using this work, please cite the original dataset and references:

- Bansal, S. (2020). Real / fake job posting prediction. [Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?resource=download)  
- Awad, M., & Khanna, R. (2015). *Efficient Learning Machines*. Apress.  
- Batuwita, R., & Palade, V. (2013). *Class imbalance learning methods for support vector machines*.  
- Sebastiani, F. (2002). *Machine learning in automated text categorization*. ACM Computing Surveys.  
