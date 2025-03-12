# Research_paper_analysis
# 📄 Research Paper Classification  

This project classifies research papers as **publishable** or **non-publishable** using **feature extraction, clustering, and NLP techniques**. It includes modular Python scripts, Jupyter notebooks for analysis, and pre-trained models for classification.  

---

## 📂 Project Structure  
project/
├── main.py                      # Entry point for execution
├── sample.pdf                   # Test PDF for classification
├── requirements.txt             # Lists dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignored files
│
├── notebooks/                   # Contains original Jupyter notebooks and data
│   ├── Task1.ipynb              # Feature extraction & clustering analysis
│   ├── Task2.ipynb              # NLP-based classification analysis
│   ├── create_csv.ipynb         # Prepares and processes CSV files
│   ├── results.csv              # Output results
│   ├── data/                    # Raw data files used for processing
│
├── mod/                        # Contains models, utilities, and classification logic
│   ├── __init__.py              # Marks as a package
│   ├── classifier.py            # Classification logic for papers
│   ├── KNN_train_data.npy       # Pre-trained KNN data
│   ├── KNN_train_labels.npy     # Labels for KNN training
│   ├── LLM.py                   # Large Language Model processing
│   ├── text_processing.py       # NLP processing for classification
│   ├── utils.py                 # Helper functions


