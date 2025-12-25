import streamlit as st
import pandas as pd
import pickle
import joblib
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
    layout="wide"
)

# Title
st.title("💄 All Beauty — Two-Stage Recommendation System")
st.markdown("---")

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
def load_xgb_pipeline():
    """Load XGBoost ranking pipeline."""
    pipeline_path = MODELS_DIR / "xgb_ranker.joblib"
    return joblib.load(pipeline_path)

# Load data
try:
    df_ratings = load_ratings()
    df_user_features = load_user_features()
    df_item_features = load_item_features()
    svd_data = load_svd_model()
    xgb_pipeline = load_xgb_pipeline()
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

preprocessor = xgb_pipeline['preprocessor']
xgb_model = xgb_pipeline['model']
numeric_features = xgb_pipeline['numeric_features']
categorical_features = xgb_pipeline['categorical_features']

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# User selection
available_users = sorted(df_ratings['user_id'].unique())
selected_user = st.sidebar.selectbox(
    "Select User ID",
    options=available_users,
    index=0
)

# Retrieval candidates slider
n_candidates = st.sidebar.slider(
    "Retrieval Candidates",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

# Top-K slider
top_k = st.sidebar.slider(
    "Show Top-K Recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

# Recommend button
recommend_button = st.sidebar.button("🔍 Recommend", type="primary")

# Helper function for retrieval
def get_candidates(user_id: str, n: int = 100):
    """Get top-n candidate items for a user using SVD predictions."""
    # Get items user has NOT interacted with
    seen = user_seen_items.get(user_id, set())
    unseen_items = all_items - seen
    
    if len(unseen_items) == 0:
        return []
    
    # Get user internal ID
    try:
        user_inner_id = svd_trainset.to_inner_uid(user_id)
    except ValueError:
        # User not in training set (cold start)
        return []
    
    # Predict ratings for all unseen items
    predictions = []
    for item_asin in unseen_items:
        try:
            item_inner_id = svd_trainset.to_inner_iid(item_asin)
            pred = svd_model.predict(user_inner_id, item_inner_id)
            predictions.append((item_asin, pred.est))
        except ValueError:
            # Item not in training set, skip
            continue
    
    # Sort by predicted score descending and return top-n
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Helper function for ranking
def rank_candidates(user_id: str, candidate_asins: list):
    """Rank candidate items using XGBoost model."""
    if len(candidate_asins) == 0:
        return []
    
    # Create feature rows for candidates
    candidate_data = []
    for asin in candidate_asins:
        row = {'user_id': user_id, 'parent_asin': asin}
        candidate_data.append(row)
    
    df_candidates = pd.DataFrame(candidate_data)
    
    # Join with user features
    df_candidates = df_candidates.merge(df_user_features, on='user_id', how='left')
    
    # Join with item features
    df_candidates = df_candidates.merge(df_item_features, on='parent_asin', how='left')
    
    # Prepare features
    X = df_candidates[numeric_features + categorical_features].copy()
    
    # Fill NaN
    for col in numeric_features:
        if col in X.columns:
            median_val = df_user_features[col].median() if col in df_user_features.columns else 0.0
            X[col] = X[col].fillna(median_val)
    
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna("unknown")
    
    # Preprocess
    X_processed = preprocessor.transform(X)
    
    # Predict
    scores = xgb_model.predict(X_processed)
    
    # Combine with ASINs
    results = list(zip(candidate_asins, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

# Main content
if recommend_button or 'recommendations' not in st.session_state:
    with st.spinner("Generating recommendations..."):
        # Retrieval stage
        candidates = get_candidates(selected_user, n=n_candidates)
        
        if len(candidates) == 0:
            st.warning(f"No candidates found for user {selected_user}")
            st.stop()
        
        candidate_asins = [asin for asin, _ in candidates]
        
        # Ranking stage
        ranked_results = rank_candidates(selected_user, candidate_asins)
        
        # Get top-k
        top_results = ranked_results[:top_k]
        
        # Store in session state
        st.session_state['recommendations'] = top_results
        st.session_state['user_id'] = selected_user

# Display recommendations
if 'recommendations' in st.session_state:
    st.header(f"📊 Recommendations for User: `{st.session_state['user_id']}`")
    st.markdown(f"Showing top **{len(st.session_state['recommendations'])}** recommendations")
    st.markdown("---")
    
    recommendations = st.session_state['recommendations']
    
    # Create grid layout (5 columns)
    n_cols = 5
    n_rows = (len(recommendations) + n_cols - 1) // n_cols
    
    for row_idx in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            item_idx = row_idx * n_cols + col_idx
            if item_idx < len(recommendations):
                asin, score = recommendations[item_idx]
                
                # Get item details
                item_info = df_item_features[df_item_features['parent_asin'] == asin]
                
                if len(item_info) > 0:
                    item = item_info.iloc[0]
                    title = item.get('title_clean', 'N/A')
                    price = item.get('price_float', None)
                    image_url = item.get('image_url', None)
                else:
                    title = 'N/A'
                    price = None
                    image_url = None
                
                with cols[col_idx]:
                    # Card container
                    with st.container():
                        # Image
                        if image_url and pd.notna(image_url) and str(image_url).strip():
                            try:
                                st.image(str(image_url), use_container_width=True)
                            except Exception as e:
                                st.markdown("🖼️ *Image unavailable*")
                        else:
                            st.markdown("🖼️ *Image unavailable*")
                        
                        # Title
                        st.markdown(f"**{title[:60]}{'...' if len(str(title)) > 60 else ''}**")
                        
                        # Price
                        if price and pd.notna(price):
                            st.markdown(f"💰 ${price:.2f}")
                        else:
                            st.markdown("💰 Price: N/A")
                        
                        # Score
                        st.markdown(f"⭐ Score: `{score:.3f}`")
                        
                        # ASIN
                        st.caption(f"ASIN: `{asin}`")
                        
                        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**Two-Stage Recommendation System:** Retrieval (SVD) → Ranking (XGBoost)")

