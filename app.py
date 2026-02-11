# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import plotly.express as px
# # from scipy.cluster.hierarchy import linkage, dendrogram
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.cluster import AgglomerativeClustering
# # from sklearn.decomposition import PCA
# # from sklearn.metrics import silhouette_score

# # # ==========================================
# # # 1️⃣ App Title & Description
# # # ==========================================
# # st.set_page_config(page_title="News Topic Discovery", layout="wide")
# # st.title("🟣 News Topic Discovery Dashboard")
# # st.markdown("""
# # **This system uses Hierarchical Clustering to automatically group similar news articles based on textual similarity.**

# # 👉 *Purpose: Discover hidden themes without defining categories upfront.*
# # """)

# # # Initialize Session State (Memory for step-by-step flow)
# # if "dendrogram_ready" not in st.session_state:
# #     st.session_state.dendrogram_ready = False

# # # Helper function to auto-detect text column
# # def get_default_text_column(df):
# #     cols = [c.lower() for c in df.columns]
# #     for candidate in ['news', 'text', 'headline', 'content', 'description']:
# #         if candidate in cols:
# #             return df.columns[cols.index(candidate)]
# #     for col in df.columns:
# #         if df[col].dtype == 'object':
# #             return col
# #     return df.columns[0]

# # # ==========================================
# # # 2️⃣ Input Section (Sidebar)
# # # ==========================================
# # st.sidebar.header("📂 Dataset Handling")
# # uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# # if uploaded_file:
# #     # Load and clean data
# #     df = pd.read_csv(uploaded_file, encoding='latin1').dropna(how='all')
    
# #     # Subsample to prevent browser memory crashes
# #     if len(df) > 3000:
# #         st.sidebar.warning("Dataset subsampled to 3000 rows for performance.")
# #         df = df.sample(3000, random_state=42).reset_index(drop=True)

# #     # Column Selection & Preview
# #     all_columns = df.columns.tolist()
# #     default_col = get_default_text_column(df)
# #     default_idx = all_columns.index(default_col) if default_col in all_columns else 0
    
# #     text_col = st.sidebar.selectbox(
# #         "Select the Text Column to Cluster:", 
# #         options=all_columns, 
# #         index=default_idx,
# #         help="Ensure this column contains actual sentences or paragraphs."
# #     )
    
# #     # Fill any empty strings in the selected column to prevent errors
# #     df[text_col] = df[text_col].fillna("").astype(str)

# #     # 🌟 NEW: Show a preview of the text so the user knows they picked the right column
# #     st.sidebar.markdown("**Preview of selected column:**")
# #     st.sidebar.info(f'"{df[text_col].iloc[0][:100]}..."')

# #     # Vectorization Controls
# #     st.sidebar.header("📝 Text Vectorization Controls")
# #     max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000, step=100)
# #     use_stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)
# #     ngram_choice = st.sidebar.selectbox("N-gram Range", ["Unigrams", "Bigrams", "Unigrams + Bigrams"])
    
# #     ngram_map = {"Unigrams": (1, 1), "Bigrams": (2, 2), "Unigrams + Bigrams": (1, 2)}
# #     stopwords_param = 'english' if use_stopwords else None

# #     # Clustering Controls
# #     st.sidebar.header("🌳 Clustering Controls")
# #     linkage_method = st.sidebar.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single']).lower()
    
# #     # Logic constraint: Ward requires Euclidean distance
# #     if linkage_method == 'ward':
# #         dist_metric = st.sidebar.selectbox("Distance Metric", ['euclidean'], disabled=True)
# #     else:
# #         dist_metric = st.sidebar.selectbox("Distance Metric", ['euclidean', 'cosine', 'manhattan'])
        
# #     subset_size = st.sidebar.slider("Articles for Dendrogram", 20, 200, 50, step=10)

# #     # ==========================================
# #     # 3️⃣ Clustering Control - Step 1 (Vectorize & Dendrogram)
# #     # ==========================================
# #     if st.sidebar.button("🟦 Generate Dendrogram", use_container_width=True):
# #         with st.spinner("Vectorizing text..."):
# #             vectorizer = TfidfVectorizer(
# #                 stop_words=stopwords_param, 
# #                 max_features=max_features, 
# #                 ngram_range=ngram_map[ngram_choice]
# #             )
            
# #             # 🛑 Graceful Error Handling for Empty Vocabularies
# #             try:
# #                 X = vectorizer.fit_transform(df[text_col])
                
# #                 # Check if vocabulary is too small
# #                 if X.shape[1] < 5:
# #                     st.error("🚨 **Warning:** The resulting vocabulary is too small. Please select a different text column or uncheck 'Remove English Stopwords'.")
# #                     st.stop()
                    
# #             except ValueError as e:
# #                 if "empty vocabulary" in str(e).lower():
# #                     st.error("🚨 **Vocabulary Error:** The selected column resulted in an empty vocabulary (no usable words found). **Please look at the 'Preview' in the sidebar and select the correct text column.**")
# #                     st.stop()
# #                 else:
# #                     raise e
            
# #             # Save required data to session state
# #             st.session_state.X_matrix = X
# #             st.session_state.df_processed = df
# #             st.session_state.vectorizer = vectorizer
# #             st.session_state.text_col = text_col
# #             st.session_state.dendrogram_ready = True
# #             st.session_state.subset_size = subset_size
# #             st.session_state.linkage_method = linkage_method
# #             st.session_state.dist_metric = dist_metric

# # # ==========================================
# # # 4️⃣ Dendrogram Section (Main Panel)
# # # ==========================================
# # if st.session_state.dendrogram_ready:
# #     st.markdown("---")
# #     st.subheader("🌳 Dendrogram Visual Inspection")
# #     st.write("Inspect the tree below. Look for large vertical lines without horizontal intersections to determine natural groupings.")
    
    

# #     X = st.session_state.X_matrix
# #     subset = min(st.session_state.subset_size, X.shape[0])
# #     X_subset_dense = X[:subset].toarray()
    
# #     Z = linkage(X_subset_dense, method=st.session_state.linkage_method, metric=st.session_state.dist_metric)
    
# #     col1, col2 = st.columns([3, 1])
# #     with col1:
# #         fig, ax = plt.subplots(figsize=(12, 6))
# #         dendro = dendrogram(Z, ax=ax)
# #         plt.title(f"Hierarchical Clustering Dendrogram (Subset: {subset} articles)")
# #         plt.xlabel("Article Index")
# #         plt.ylabel("Distance")
# #         st.pyplot(fig)

# #     with col2:
# #         st.info("💡 **How to read this:**\n\nEach leaf at the bottom is one article. As you move up, similar articles merge. A large vertical gap indicates a distinct change in topic/cluster.")

# #     # ==========================================
# #     # 5️⃣ Apply Clustering (Step 2)
# #     # ==========================================
# #     st.markdown("---")
# #     st.subheader("⚙️ Apply Clustering")
# #     n_clusters = st.number_input("Select Number of Clusters (based on dendrogram):", min_value=2, max_value=20, value=3)
    
# #     if st.button("🟩 Apply Clustering", use_container_width=True):
# #         with st.spinner("Applying clustering to the full dataset..."):
# #             X_dense = X.toarray()
# #             agg = AgglomerativeClustering(
# #                 n_clusters=n_clusters, 
# #                 linkage=st.session_state.linkage_method, 
# #                 metric=st.session_state.dist_metric
# #             )
# #             clusters = agg.fit_predict(X_dense)
            
# #             df_final = st.session_state.df_processed.copy()
# #             df_final['Cluster'] = clusters
# #             text_column = st.session_state.text_col
            
# #             # ==========================================
# #             # 6️⃣ PCA Visualization
# #             # ==========================================
# #             st.markdown("---")
# #             st.subheader("🌌 2D Cluster Projection (PCA)")
            
# #             n_components = min(2, X_dense.shape[1]) 
# #             pca = PCA(n_components=n_components)
# #             X_pca = pca.fit_transform(X_dense)
            
# #             df_final['PCA1'] = X_pca[:, 0]
# #             df_final['PCA2'] = X_pca[:, 1] if n_components == 2 else 0
# #             df_final['Snippet'] = df_final[text_column].str[:80] + "..."
# #             df_final['Cluster_Label'] = df_final['Cluster'].astype(str)
            
# #             fig_pca = px.scatter(
# #                 df_final, x='PCA1', y='PCA2', 
# #                 color='Cluster_Label', 
# #                 hover_data=['Snippet'],
# #                 title="Articles Projected in 2D Space",
# #                 color_discrete_sequence=px.colors.qualitative.Vivid
# #             )
# #             st.plotly_chart(fig_pca, use_container_width=True)

# #             # ==========================================
# #             # 7️⃣ Validation (Silhouette Score)
# #             # ==========================================
# #             st.markdown("---")
# #             st.subheader("📊 Validation: Silhouette Score")
            
# #             if len(set(clusters)) > 1:
# #                 sil_score = silhouette_score(X_dense, clusters, metric=st.session_state.dist_metric)
# #                 sil_col1, sil_col2 = st.columns([1, 3])
# #                 sil_col1.metric("Silhouette Score", f"{sil_score:.3f}")
# #                 sil_col2.info("**Meaning:** Close to 1 → Well-separated. Close to 0 → Overlapping. Negative → Poorly clustered.")
# #             else:
# #                 st.warning("Only 1 cluster found. Silhouette score cannot be calculated.")

# #             # ==========================================
# #             # 8️⃣ Cluster Summary & Business View
# #             # ==========================================
# #             st.markdown("---")
# #             st.subheader("📋 Cluster Summary (Business View)")
            
# #             vectorizer = st.session_state.vectorizer
# #             feature_names = vectorizer.get_feature_names_out()
# #             df_tfidf = pd.DataFrame(X_dense, columns=feature_names)
# #             df_tfidf['Cluster'] = clusters
            
# #             summary_data = []
# #             cluster_interpretations = []
            
# #             for c in range(n_clusters):
# #                 cluster_df = df_final[df_final['Cluster'] == c]
# #                 if len(cluster_df) == 0:
# #                     continue
                
# #                 # Get top words
# #                 mean_tfidf = df_tfidf[df_tfidf['Cluster'] == c].drop('Cluster', axis=1).mean()
# #                 top_words = mean_tfidf.sort_values(ascending=False).head(10).index.tolist()
# #                 rep_snippet = cluster_df.iloc[0][text_column][:150] + "..."
                
# #                 summary_data.append({
# #                     "Cluster ID": c,
# #                     "Articles": len(cluster_df),
# #                     "Top Keywords": ", ".join(top_words),
# #                     "Snippet": rep_snippet
# #                 })
                
# #                 # Generate business interpretations
# #                 if len(top_words) >= 3:
# #                     cluster_interpretations.append(f"**Cluster {c}**: Focuses on **{top_words[0]}**, **{top_words[1]}**, and **{top_words[2]}**.")
# #                 else:
# #                     cluster_interpretations.append(f"**Cluster {c}**: Contains unique/fringe vocabulary.")
                
# #             st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# #             # ==========================================
# #             # 9️⃣ Business Interpretation Section
# #             # ==========================================
# #             st.markdown("---")
# #             st.subheader("🗣️ Business Insights (Human Language)")
# #             for interp in cluster_interpretations:
# #                 st.markdown(f"- 🟣 {interp}")

# # else:
# #     if not uploaded_file:
# #         st.info("👈 Upload a dataset and configure your settings in the sidebar to begin.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score

# # ==========================================
# # 1️⃣ App Configuration & Custom CSS
# # ==========================================
# st.set_page_config(page_title="News Topic Discovery", page_icon="🟣", layout="wide")

# # Inject Custom CSS for a professional look
# st.markdown("""
# <style>
#     /* Reduce top padding */
#     .block-container { padding-top: 2rem; padding-bottom: 2rem; }
#     /* Style the metric text */
#     div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; color: #6C63FF; }
#     /* Hide Streamlit default branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     /* Subtle background for insights */
#     .insight-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #6C63FF; margin-bottom: 10px; }
# </style>
# """, unsafe_allow_html=True)

# # Title Header
# col1, col2 = st.columns([3, 1])
# with col1:
#     st.title("🟣 News Topic Discovery")
#     st.markdown("Automatically group similar news articles based on textual similarity using **Hierarchical Clustering**.")
# with col2:
#     st.info("💡 **Tip:** Upload your data, inspect the tree, and let the algorithm reveal hidden themes.")

# # Initialize Session State
# if "dendrogram_ready" not in st.session_state:
#     st.session_state.dendrogram_ready = False

# def get_default_text_column(df):
#     cols = [c.lower() for c in df.columns]
#     for candidate in ['news', 'text', 'headline', 'content', 'description']:
#         if candidate in cols:
#             return df.columns[cols.index(candidate)]
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             return col
#     return df.columns[0]

# # ==========================================
# # 2️⃣ Sidebar & Configuration
# # ==========================================
# with st.sidebar:
#     st.header("📂 Data & Settings")
#     uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

#     if uploaded_file:
#         df = pd.read_csv(uploaded_file, encoding='latin1').dropna(how='all')
        
#         if len(df) > 3000:
#             st.warning("Dataset subsampled to 3000 rows for performance optimization.")
#             df = df.sample(3000, random_state=42).reset_index(drop=True)

#         all_columns = df.columns.tolist()
#         default_idx = all_columns.index(get_default_text_column(df)) if get_default_text_column(df) in all_columns else 0
        
#         text_col = st.selectbox("Text Column to Cluster", options=all_columns, index=default_idx)
#         df[text_col] = df[text_col].fillna("").astype(str)

#         st.markdown("**Column Preview:**")
#         st.caption(f'"{df[text_col].iloc[0][:100]}..."')
#         st.divider()

#         # Clean Expanders for Advanced Settings
#         with st.expander("🛠️ Text Vectorization Settings"):
#             max_features = st.slider("Max TF-IDF Features", 100, 2000, 1000, step=100)
#             use_stopwords = st.checkbox("Remove English Stopwords", value=True)
#             ngram_choice = st.selectbox("N-gram Range", ["Unigrams", "Bigrams", "Unigrams + Bigrams"])
        
#         with st.expander("🌳 Clustering Settings"):
#             linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single']).lower()
#             dist_metric = st.selectbox("Distance Metric", ['euclidean'], disabled=True) if linkage_method == 'ward' else st.selectbox("Distance Metric", ['euclidean', 'cosine', 'manhattan'])
#             subset_size = st.slider("Articles for Dendrogram", 20, 200, 50, step=10)

#         ngram_map = {"Unigrams": (1, 1), "Bigrams": (2, 2), "Unigrams + Bigrams": (1, 2)}
#         stopwords_param = 'english' if use_stopwords else None

#         st.divider()
#         # Primary Action Button
#         generate_btn = st.button("🟦 Generate Dendrogram", use_container_width=True, type="primary")

# # ==========================================
# # 3️⃣ Processing & Error Handling
# # ==========================================
# if uploaded_file and generate_btn:
#     with st.spinner("Analyzing text and building hierarchy..."):
#         vectorizer = TfidfVectorizer(stop_words=stopwords_param, max_features=max_features, ngram_range=ngram_map[ngram_choice])
#         try:
#             X = vectorizer.fit_transform(df[text_col])
#             if X.shape[1] < 5:
#                 st.error("🚨 Vocabulary too small. Try unchecking 'Remove Stopwords' or pick a different column.")
#                 st.stop()
#         except ValueError as e:
#             if "empty vocabulary" in str(e).lower():
#                 st.error("🚨 **Vocabulary Error:** No usable words found. Check the column preview in the sidebar.")
#                 st.stop()
#             else: raise e
        
#         # Save to Session State
#         st.session_state.update({
#             "X_matrix": X, "df_processed": df, "vectorizer": vectorizer, 
#             "text_col": text_col, "dendrogram_ready": True, 
#             "subset_size": subset_size, "linkage_method": linkage_method, "dist_metric": dist_metric
#         })

# # ==========================================
# # 4️⃣ Dendrogram & Main UI
# # ==========================================
# if st.session_state.dendrogram_ready:
#     with st.container(border=True):
#         st.subheader("🌳 Step 1: Visual Inspection (Dendrogram)")
#         st.write("Inspect the tree below. A large vertical gap indicates a distinct change in topic/cluster.")
        
#         X = st.session_state.X_matrix
#         subset = min(st.session_state.subset_size, X.shape[0])
#         X_subset_dense = X[:subset].toarray()
        
#         Z = linkage(X_subset_dense, method=st.session_state.linkage_method, metric=st.session_state.dist_metric)
        
#         fig, ax = plt.subplots(figsize=(14, 4)) # Wider, shorter plot for modern feel
#         dendro = dendrogram(Z, ax=ax)
#         plt.title(f"Hierarchy (Subset: {subset} articles)", fontsize=10)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         st.pyplot(fig)

#     # ==========================================
#     # 5️⃣ Apply Clustering
#     # ==========================================
#     st.divider()
#     col_a, col_b = st.columns([1, 3])
#     with col_a:
#         st.subheader("⚙️ Step 2: Cluster")
#         n_clusters = st.number_input("Number of Clusters:", min_value=2, max_value=20, value=3)
#         apply_btn = st.button("🟩 Apply & Analyze", use_container_width=True, type="primary")

#     if apply_btn:
#         with st.spinner("Assigning clusters and generating insights..."):
#             X_dense = X.toarray()
#             agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=st.session_state.linkage_method, metric=st.session_state.dist_metric)
#             clusters = agg.fit_predict(X_dense)
            
#             df_final = st.session_state.df_processed.copy()
#             df_final['Cluster'] = clusters
#             text_column = st.session_state.text_col
            
#             # Extract Keywords
#             vectorizer = st.session_state.vectorizer
#             df_tfidf = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())
#             df_tfidf['Cluster'] = clusters
            
#             # ==========================================
#             # 6️⃣ Tabbed Results Dashboard
#             # ==========================================
#             st.divider()
#             tab1, tab2, tab3 = st.tabs(["🌌 2D Projection", "📋 Cluster Details", "🗣️ Business Insights"])
            
#             # --- TAB 1: Visualizations ---
#             with tab1:
#                 col_plot, col_metric = st.columns([3, 1])
#                 with col_plot:
#                     n_components = min(2, X_dense.shape[1]) 
#                     X_pca = PCA(n_components=n_components).fit_transform(X_dense)
#                     df_final['PCA1'] = X_pca[:, 0]
#                     df_final['PCA2'] = X_pca[:, 1] if n_components == 2 else 0
#                     df_final['Snippet'] = df_final[text_column].str[:100] + "..."
#                     df_final['Cluster_Label'] = "Cluster " + df_final['Cluster'].astype(str)
                    
#                     fig_pca = px.scatter(
#                         df_final, x='PCA1', y='PCA2', color='Cluster_Label', hover_data=['Snippet'],
#                         color_discrete_sequence=px.colors.qualitative.Pastel
#                     )
#                     fig_pca.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)")
#                     st.plotly_chart(fig_pca, use_container_width=True)
                
#                 with col_metric:
#                     st.markdown("<br><br>", unsafe_allow_html=True)
#                     if len(set(clusters)) > 1:
#                         sil_score = silhouette_score(X_dense, clusters, metric=st.session_state.dist_metric)
#                         st.metric("Silhouette Score", f"{sil_score:.3f}")
#                         st.caption("🟢 Closer to 1 = Well Separated<br>🔴 Closer to 0 = Overlapping", unsafe_allow_html=True)
#                     else:
#                         st.warning("Needs >1 cluster for scoring.")

#             # --- TAB 2: Table Summary ---
#             with tab2:
#                 summary_data = []
#                 for c in range(n_clusters):
#                     cluster_df = df_final[df_final['Cluster'] == c]
#                     if len(cluster_df) == 0: continue
#                     mean_tfidf = df_tfidf[df_tfidf['Cluster'] == c].drop('Cluster', axis=1).mean()
#                     top_words = mean_tfidf.sort_values(ascending=False).head(10).index.tolist()
#                     summary_data.append({
#                         "Cluster": f"C{c}",
#                         "Count": len(cluster_df),
#                         "Top Keywords": ", ".join(top_words),
#                         "Sample Snippet": cluster_df.iloc[0][text_column][:200] + "..."
#                     })
                
#                 # Professional column formatting
#                 st.dataframe(
#                     pd.DataFrame(summary_data),
#                     use_container_width=True, hide_index=True,
#                     column_config={"Sample Snippet": st.column_config.TextColumn("Sample Snippet", width="large")}
#                 )

#             # --- TAB 3: Insights ---
#             with tab3:
#                 st.write("Based on the top TF-IDF keywords, here is the AI interpretation of your themes:")
#                 for c in range(n_clusters):
#                     mean_tfidf = df_tfidf[df_tfidf['Cluster'] == c].drop('Cluster', axis=1).mean()
#                     top_words = mean_tfidf.sort_values(ascending=False).head(3).index.tolist()
#                     if len(top_words) >= 3:
#                         st.markdown(f'<div class="insight-box"><b>🟣 Cluster {c}:</b> Focuses primarily on topics related to <b>{top_words[0]}</b>, <b>{top_words[1]}</b>, and <b>{top_words[2]}</b>.</div>', unsafe_allow_html=True)

# else:
#     if not uploaded_file:
#         st.info("👈 Please upload a CSV dataset in the sidebar to begin.")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==========================================
# 1️⃣ App Configuration & Custom CSS (DARK MODE OPTIMIZED)
# ==========================================
st.set_page_config(page_title="News Topic Discovery", page_icon="🟣", layout="wide")

# Inject Custom CSS for a professional Dark Mode look
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* 🌟 Make the Silhouette Score highly visible */
    div[data-testid="stMetricValue"] { 
        font-size: 3rem !important; 
        font-weight: 800 !important; 
        color: #9D8CFF !important; /* Bright pastel purple for dark mode */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #E0E0E0 !important;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 🌟 Dark Mode Insight Box */
    .insight-box { 
        background-color: #262730; /* Sleek dark background */
        color: #FFFFFF; /* White text for contrast */
        padding: 18px; 
        border-radius: 8px; 
        border-left: 5px solid #8C52FF; 
        margin-bottom: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.4); /* Subtle shadow */
        font-size: 1.05rem;
    }
    /* Highlight the keywords inside the box */
    .insight-box b {
        color: #B39DFF; /* Lighter purple to make keywords pop */
    }
</style>
""", unsafe_allow_html=True)

# Title Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🟣 News Topic Discovery")
    st.markdown("Automatically group similar news articles based on textual similarity using **Hierarchical Clustering**.")
with col2:
    st.info("💡 **Tip:** Upload your data, inspect the tree, and let the algorithm reveal hidden themes.")

# Initialize Session State
if "dendrogram_ready" not in st.session_state:
    st.session_state.dendrogram_ready = False

def get_default_text_column(df):
    cols = [c.lower() for c in df.columns]
    for candidate in ['news', 'text', 'headline', 'content', 'description']:
        if candidate in cols:
            return df.columns[cols.index(candidate)]
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    return df.columns[0]

# ==========================================
# 2️⃣ Sidebar & Configuration
# ==========================================
with st.sidebar:
    st.header("📂 Data & Settings")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin1').dropna(how='all')
        
        if len(df) > 3000:
            st.warning("Dataset subsampled to 3000 rows for performance optimization.")
            df = df.sample(3000, random_state=42).reset_index(drop=True)

        all_columns = df.columns.tolist()
        default_idx = all_columns.index(get_default_text_column(df)) if get_default_text_column(df) in all_columns else 0
        
        text_col = st.selectbox("Text Column to Cluster", options=all_columns, index=default_idx)
        df[text_col] = df[text_col].fillna("").astype(str)

        st.markdown("**Column Preview:**")
        st.caption(f'"{df[text_col].iloc[0][:100]}..."')
        st.divider()

        # Clean Expanders for Advanced Settings
        with st.expander("🛠️ Text Vectorization Settings"):
            max_features = st.slider("Max TF-IDF Features", 100, 2000, 1000, step=100)
            use_stopwords = st.checkbox("Remove English Stopwords", value=True)
            ngram_choice = st.selectbox("N-gram Range", ["Unigrams", "Bigrams", "Unigrams + Bigrams"])
        
        with st.expander("🌳 Clustering Settings"):
            linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single']).lower()
            dist_metric = st.selectbox("Distance Metric", ['euclidean'], disabled=True) if linkage_method == 'ward' else st.selectbox("Distance Metric", ['euclidean', 'cosine', 'manhattan'])
            subset_size = st.slider("Articles for Dendrogram", 20, 200, 50, step=10)

        ngram_map = {"Unigrams": (1, 1), "Bigrams": (2, 2), "Unigrams + Bigrams": (1, 2)}
        stopwords_param = 'english' if use_stopwords else None

        st.divider()
        # Primary Action Button
        generate_btn = st.button("🟦 Generate Dendrogram", use_container_width=True, type="primary")

# ==========================================
# 3️⃣ Processing & Error Handling
# ==========================================
if uploaded_file and generate_btn:
    with st.spinner("Analyzing text and building hierarchy..."):
        vectorizer = TfidfVectorizer(stop_words=stopwords_param, max_features=max_features, ngram_range=ngram_map[ngram_choice])
        try:
            X = vectorizer.fit_transform(df[text_col])
            if X.shape[1] < 5:
                st.error("🚨 Vocabulary too small. Try unchecking 'Remove Stopwords' or pick a different column.")
                st.stop()
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                st.error("🚨 **Vocabulary Error:** No usable words found. Check the column preview in the sidebar.")
                st.stop()
            else: raise e
        
        # Save to Session State
        st.session_state.update({
            "X_matrix": X, "df_processed": df, "vectorizer": vectorizer, 
            "text_col": text_col, "dendrogram_ready": True, 
            "subset_size": subset_size, "linkage_method": linkage_method, "dist_metric": dist_metric
        })

# ==========================================
# 4️⃣ Dendrogram & Main UI
# ==========================================
if st.session_state.dendrogram_ready:
    with st.container(border=True):
        st.subheader("🌳 Step 1: Visual Inspection (Dendrogram)")
        st.write("Inspect the tree below. A large vertical gap indicates a distinct change in topic/cluster.")
        
        X = st.session_state.X_matrix
        subset = min(st.session_state.subset_size, X.shape[0])
        X_subset_dense = X[:subset].toarray()
        
        Z = linkage(X_subset_dense, method=st.session_state.linkage_method, metric=st.session_state.dist_metric)
        
        fig, ax = plt.subplots(figsize=(14, 4))
        # Style the dendrogram for dark mode compatibility 
        fig.patch.set_facecolor('none') 
        ax.set_facecolor('none')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        
        dendro = dendrogram(Z, ax=ax)
        plt.title(f"Hierarchy (Subset: {subset} articles)", color='gray', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        st.pyplot(fig)

    # ==========================================
    # 5️⃣ Apply Clustering
    # ==========================================
    st.divider()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.subheader("⚙️ Step 2: Cluster")
        n_clusters = st.number_input("Number of Clusters:", min_value=2, max_value=20, value=3)
        apply_btn = st.button("🟩 Apply & Analyze", use_container_width=True, type="primary")

    if apply_btn:
        with st.spinner("Assigning clusters and generating insights..."):
            X_dense = X.toarray()
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=st.session_state.linkage_method, metric=st.session_state.dist_metric)
            clusters = agg.fit_predict(X_dense)
            
            df_final = st.session_state.df_processed.copy()
            df_final['Cluster'] = clusters
            text_column = st.session_state.text_col
            
            # Extract Keywords
            vectorizer = st.session_state.vectorizer
            df_tfidf = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())
            df_tfidf['Cluster'] = clusters
            
            # ==========================================
            # 6️⃣ Tabbed Results Dashboard
            # ==========================================
            st.divider()
            tab1, tab2, tab3 = st.tabs(["🌌 2D Projection", "📋 Cluster Details", "🗣️ Business Insights"])
            
            # --- TAB 1: Visualizations & Score ---
            with tab1:
                col_plot, col_metric = st.columns([3, 1])
                with col_plot:
                    n_components = min(2, X_dense.shape[1]) 
                    X_pca = PCA(n_components=n_components).fit_transform(X_dense)
                    df_final['PCA1'] = X_pca[:, 0]
                    df_final['PCA2'] = X_pca[:, 1] if n_components == 2 else 0
                    df_final['Snippet'] = df_final[text_column].str[:100] + "..."
                    df_final['Cluster_Label'] = "Cluster " + df_final['Cluster'].astype(str)
                    
                    fig_pca = px.scatter(
                        df_final, x='PCA1', y='PCA2', color='Cluster_Label', hover_data=['Snippet'],
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pca.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    fig_pca.update_xaxes(showgrid=False)
                    fig_pca.update_yaxes(showgrid=False)
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                with col_metric:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if len(set(clusters)) > 1:
                        sil_score = silhouette_score(X_dense, clusters, metric=st.session_state.dist_metric)
                        st.metric("Silhouette Score", f"{sil_score:.3f}")
                        st.markdown("""
                        <div style="color:#A0A0A0; font-size:0.9rem; margin-top:-15px;">
                        🟢 Closer to 1 = Well Separated<br>
                        🔴 Closer to 0 = Overlapping
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Needs >1 cluster for scoring.")

            # --- TAB 2: Table Summary ---
            with tab2:
                summary_data = []
                for c in range(n_clusters):
                    cluster_df = df_final[df_final['Cluster'] == c]
                    if len(cluster_df) == 0: continue
                    mean_tfidf = df_tfidf[df_tfidf['Cluster'] == c].drop('Cluster', axis=1).mean()
                    top_words = mean_tfidf.sort_values(ascending=False).head(10).index.tolist()
                    summary_data.append({
                        "Cluster": f"C{c}",
                        "Count": len(cluster_df),
                        "Top Keywords": ", ".join(top_words),
                        "Sample Snippet": cluster_df.iloc[0][text_column][:200] + "..."
                    })
                
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True, hide_index=True,
                    column_config={"Sample Snippet": st.column_config.TextColumn("Sample Snippet", width="large")}
                )

            # --- TAB 3: Insights ---
            with tab3:
                st.write("Based on the top TF-IDF keywords, here is the AI interpretation of your themes:")
                for c in range(n_clusters):
                    mean_tfidf = df_tfidf[df_tfidf['Cluster'] == c].drop('Cluster', axis=1).mean()
                    top_words = mean_tfidf.sort_values(ascending=False).head(3).index.tolist()
                    if len(top_words) >= 3:
                        st.markdown(f'<div class="insight-box"><b>🟣 Cluster {c}:</b> Focuses primarily on topics related to <b>{top_words[0]}</b>, <b>{top_words[1]}</b>, and <b>{top_words[2]}</b>.</div>', unsafe_allow_html=True)

else:
    if not uploaded_file:
        st.info("👈 Please upload a CSV dataset in the sidebar to begin.")