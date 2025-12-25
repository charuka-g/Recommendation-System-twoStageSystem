import streamlit as st
import pandas as pd
import pickle
import joblib
import xgboost as xgb
from pathlib import Path
import numpy as np

# Set up paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Page config
st.set_page_config(
    page_title="All Beauty - Two-Stage Recommendation",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .image-container {
        width: 100%;
        height: 200px;
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        background: #f3f4f6;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.4;
        min-height: 2.5em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .card-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
        padding-top: 0.5rem;
        border-top: 1px solid #e5e7eb;
    }
    
    .price-tag {
        font-size: 1rem;
        font-weight: 700;
        color: #059669;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_ratings():
    """Load cleaned ratings."""
    ratings_path = DATA_DIR / "ratings_clean.parquet"
    return pd.read_parquet(ratings_path)

@st.cache_data
def load_user_features():
    """Load user features."""
    user_features_path = DATA_DIR / "user_features.parquet"
    return pd.read_parquet(user_features_path)

@st.cache_data
def load_item_features():
    """Load item features."""
    item_features_path = DATA_DIR / "item_features.parquet"
    return pd.read_parquet(item_features_path)

@st.cache_resource
def load_svd_model():
    """Load SVD model and helper data."""
    model_path = MODELS_DIR / "svd_surprise.pkl"
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

@st.cache_resource
def load_xgb_artifacts():
    """Load XGBoost ranking artifacts (metadata + model path)."""
    pipeline_path = MODELS_DIR / "ranking_pipeline.joblib"
    return joblib.load(pipeline_path)

# Load data
try:
    df_ratings = load_ratings()
    df_user_features = load_user_features()
    df_item_features = load_item_features()
    svd_data = load_svd_model()
    xgb_artifacts = load_xgb_artifacts()
except FileNotFoundError as e:
    st.error(f"Error loading data or models: {e}")
    st.info("Please run the notebooks first to generate the required files.")
    st.stop()

# Extract model components
svd_model = svd_data['model']
svd_trainset = svd_data['trainset']

# Convert user_seen_items to sets if they're lists
user_seen_items_raw = svd_data['user_seen_items']
user_seen_items = {}
for k, v in user_seen_items_raw.items():
    if isinstance(v, (list, tuple)):
        user_seen_items[k] = set(v)
    elif isinstance(v, set):
        user_seen_items[k] = v
    else:
        user_seen_items[k] = {v} if v else set()

all_items = set(svd_data['all_items'])

# Ranking artifacts
feature_cols = xgb_artifacts["feature_cols"]
user_code_map = xgb_artifacts["user_code_map"]
item_code_map = xgb_artifacts["item_code_map"]
user_stats = xgb_artifacts["user_stats"]
item_stats = xgb_artifacts["item_stats"]
xgb_model_path = xgb_artifacts["xgb_model_path"]

# Load booster
booster = xgb.Booster()
booster.load_model(xgb_model_path)

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Filter to only 500 users
available_users_all = sorted(df_ratings['user_id'].unique())
available_users = available_users_all[:500] if len(available_users_all) > 500 else available_users_all

# User selection
selected_user = st.sidebar.selectbox(
    "👤 Select User ID",
    options=available_users,
    index=0,
    help="Choose a user to generate personalized recommendations"
)

# Display user stats if available
if selected_user in user_stats.index:
    user_stat = user_stats.loc[selected_user]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 User Statistics")
    st.sidebar.metric("Total Ratings", int(user_stat.get('user_num_ratings', 0)))
    st.sidebar.metric("Avg Rating Given", f"{user_stat.get('user_mean_rating', 0):.2f}")

# Retrieval candidates slider
n_candidates = st.sidebar.slider(
    "🔍 Retrieval Candidates",
    min_value=20,
    max_value=300,
    value=100,
    step=10,
    help="Number of candidates to retrieve in the first stage"
)

# Top-K slider
top_k = st.sidebar.slider(
    "📈 Show Top-K Recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
    help="Number of top recommendations to display"
)

# Recommend button
recommend_button = st.sidebar.button("🚀 Generate Recommendations", type="primary", use_container_width=True)

# Helper functions
def get_candidates(user_id: str, n: int = 100):
    """Get top-n candidate items for a user using SVD predictions."""
    seen = user_seen_items.get(user_id, set())
    unseen_items = all_items - seen
    
    if len(unseen_items) == 0:
        return []
    
    try:
        user_inner_id = svd_trainset.to_inner_uid(user_id)
    except ValueError:
        return []
    
    predictions = []
    for item_asin in unseen_items:
        try:
            item_inner_id = svd_trainset.to_inner_iid(item_asin)
            pred = svd_model.predict(user_inner_id, item_inner_id)
            predictions.append((item_asin, pred.est))
        except ValueError:
            continue
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

def rank_candidates(user_id: str, candidates_with_scores: list):
    """Rank candidate items using the trained booster and feature stats."""
    if len(candidates_with_scores) == 0:
        return []

    mf_est_map = dict(candidates_with_scores)
    candidate_asins = [asin for asin, _ in candidates_with_scores]

    df_tmp = pd.DataFrame({
        "user_id": [user_id] * len(candidate_asins),
        "parent_asin": candidate_asins,
    })
    df_tmp["mf_est"] = df_tmp["parent_asin"].map(mf_est_map).fillna(0.0)

    # Merge stats
    df_tmp = df_tmp.merge(user_stats, on="user_id", how="left")
    df_tmp = df_tmp.merge(item_stats, on="parent_asin", how="left")

    for col in ["user_num_ratings", "user_mean_rating", "item_num_ratings", "item_mean_rating"]:
        df_tmp[col] = df_tmp[col].fillna(0)

    df_tmp["user_code"] = df_tmp["user_id"].map(user_code_map).fillna(-1).astype(int)
    df_tmp["item_code"] = df_tmp["parent_asin"].map(item_code_map).fillna(-1).astype(int)

    # Features in the order saved
    X_cand = df_tmp[feature_cols].to_numpy()
    dtest = xgb.DMatrix(X_cand)
    scores = booster.predict(dtest)

    results = list(zip(candidate_asins, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def is_valid_item(title, image_url):
    """Check if item has valid title and image."""
    title_valid = title is not None and pd.notna(title) and str(title).strip()
    image_valid = image_url is not None and pd.notna(image_url) and str(image_url).strip()
    return title_valid and image_valid

# Main header
st.markdown("""
    <div class="main-header">
        <h1>💄 All Beauty — Two-Stage Recommendation System</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Personalized product recommendations powered by Matrix Factorization & XGBoost</p>
    </div>
""", unsafe_allow_html=True)

# Main content
if recommend_button or 'recommendations' not in st.session_state:
    with st.spinner("🔄 Generating personalized recommendations..."):
        # Retrieval stage
        candidates = get_candidates(selected_user, n=n_candidates)
        
        if len(candidates) == 0:
            st.warning(f"⚠️ No candidates found for user {selected_user}")
            st.info("💡 Try selecting a different user or check if the user exists in the training data.")
            st.stop()
        
        # Ranking stage
        ranked_results = rank_candidates(selected_user, candidates)
        
        # Get top-k
        top_results = ranked_results[:top_k]
        
        # Filter out items with null titles or images
        filtered_results = []
        for asin, score in top_results:
            item_info = df_item_features[df_item_features['parent_asin'] == asin]
            if len(item_info) > 0:
                item = item_info.iloc[0]
                title = item.get('title_clean', None)
                image_url = item.get('image_url', None)
                
                if is_valid_item(title, image_url):
                    filtered_results.append((asin, score))
        
        # Store in session state
        st.session_state['recommendations'] = filtered_results
        st.session_state['user_id'] = selected_user
        st.session_state['original_count'] = len(top_results)
        st.session_state['filtered_count'] = len(filtered_results)

# Display recommendations
if 'recommendations' in st.session_state:
    recommendations = st.session_state['recommendations']
    
    if len(recommendations) == 0:
        st.warning("⚠️ No recommendations available after filtering (all items had missing titles or images).")
        st.info("💡 Try increasing the number of candidates or selecting a different user.")
    else:
        # Stats display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Recommendations", len(recommendations))
        with col2:
            if 'original_count' in st.session_state:
                filtered = st.session_state.get('original_count', 0) - len(recommendations)
                st.metric("🔍 Filtered Out", filtered)
        with col3:
            avg_score = np.mean([score for _, score in recommendations]) if recommendations else 0
            st.metric("⭐ Avg Score", f"{avg_score:.3f}")
        
        st.markdown("---")
        
        # Create responsive grid layout (3 columns)
        n_cols = 3
        
        for i in range(0, len(recommendations), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                item_idx = i + j
                if item_idx < len(recommendations):
                    asin, score = recommendations[item_idx]
                    
                    # Get item details
                    item_info = df_item_features[df_item_features['parent_asin'] == asin]
                    
                    if len(item_info) > 0:
                        item = item_info.iloc[0]
                        title = item.get('title_clean', 'N/A')
                        price = item.get('price_float', None)
                        image_url = item.get('image_url', None)
                        
                        # Format price
                        price_display = f"${price:.2f}" if price is not None and pd.notna(price) else "N/A"
                        
                        # Escape title for HTML
                        title_escaped = str(title).replace('"', '&quot;').replace("'", "&#39;")
                        title_short = title_escaped[:50] if len(title_escaped) > 50 else title_escaped
                        
                        with col:
                            # Card container with custom styling
                            card_html = f"""
                            <div class="recommendation-card">
                                <div class="image-container">
                                    <img src="{image_url}" alt="{title_short}" onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'padding: 2rem; text-align: center; color: #6b7280;\\'>🖼️ Image unavailable</div>';">
                                </div>
                                <div class="card-title">{title_escaped}</div>
                                <div class="card-info">
                                    <span class="price-tag">{price_display}</span>
                                    <span class="score-badge">⭐ {score:.3f}</span>
                                </div>
                                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #6b7280;">
                                    ASIN: <code>{asin}</code>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p><strong>Two-Stage Recommendation System:</strong> Retrieval (SVD) → Ranking (XGBoost)</p>
        <p style="font-size: 0.85rem;">Powered by Matrix Factorization & Gradient Boosting</p>
    </div>
""", unsafe_allow_html=True)
