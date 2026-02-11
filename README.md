# 🟣 News Topic Discovery Dashboard

**An interactive Machine Learning web application built with Streamlit that automatically groups unlabeled news articles into distinct themes using NLP and Hierarchical Clustering.**

This dashboard is designed with an educational, step-by-step flow. It forces users to visually inspect the natural structure of the data (via a Dendrogram) before deciding on the final number of clusters, bridging the gap between mathematical data science and editorial business insights.

---

## ✨ Features

* **📂 Smart Data Handling:** Upload any CSV dataset. The app automatically detects text columns and gracefully handles missing data.
* **📝 Customizable NLP Pipeline:** Fine-tune the `TfidfVectorizer` on the fly by adjusting Max Features, N-gram ranges (Unigrams/Bigrams), and Stopword removal.
* **🌳 Interactive Dendrogram:** Generate a subset dendrogram to visually inspect horizontal cuts and determine the optimal number of clusters before running the final algorithm.
* **🌌 2D PCA Projection:** High-dimensional text data is reduced to 2D using Principal Component Analysis (PCA) and visualized using interactive Plotly scatter plots.
* **📊 Model Validation:** Automatically calculates and displays the **Silhouette Score** to evaluate cluster separation and overlap.
* **🗣️ Business Insights (Dark Mode UI):** Translates raw mathematical clusters into human-readable editorial insights using top TF-IDF keywords, displayed in a sleek, dark-mode-optimized user interface.

---

## 🚀 Installation & Setup

Follow these steps to run the application locally on your machine.

### 1. Prerequisites
Make sure you have Python 3.8+ installed. 

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/news-topic-discovery.git](https://github.com/yourusername/news-topic-discovery.git)
cd news-topic-discovery

# For Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py

news-topic-discovery/
│
├── app.py                # Main Streamlit application file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
│
└── data/                 # Directory for storing local CSV datasets
    └── sample_news.csv   # (Optional) Place a test dataset here

💻 Tech Stack
Frontend: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning (NLP & Clustering): Scikit-Learn (TfidfVectorizer, AgglomerativeClustering, PCA, silhouette_score)

Hierarchical Math: SciPy (linkage, dendrogram)

Data Visualization: Plotly Express, Matplotlib
