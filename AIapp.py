"""
JengaCost AI - Construction Cost Estimation Tool for Kenyan Market
Version: 1.0.0
Author: Senior Principal Software Engineer
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import pytesseract
from typing import Tuple, Dict, List, Optional
import io
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

APP_VERSION = "1.0.0"
APP_TITLE = "JengaCost AI"
CSV_DATABASE = "training_bq.csv"
# Secure Password Retrieval
try:
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except FileNotFoundError:
    # Fallback for local testing if secrets.toml is missing
    ADMIN_PASSWORD = "Karky2003"

LOCATIONS = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
UNITS = ["m³", "m²", "m", "kg", "No.", "Sqm", "Sum", "Tonne", "Ltr"]

LOCATION_MULTIPLIERS = {
    "Nairobi": 1.00,
    "Mombasa": 1.08,
    "Kisumu": 0.98,
    "Nakuru": 0.98,
    "Eldoret": 0.96
}

# ============================================================================
# DATA INITIALIZATION
# ============================================================================

def generate_training_dataset() -> pd.DataFrame:
    """
    Generate a comprehensive Kenyan construction dataset with 60+ realistic items.
    Returns a DataFrame with columns: Description, Unit, Location, Rate
    """
    
    base_items = [
        # EARTHWORKS
        ("Excavation in soft soil", "m³", 450),
        ("Excavation in hard soil/murram", "m³", 650),
        ("Excavation in rock (manual)", "m³", 1200),
        ("Backfilling with selected material", "m³", 380),
        ("Hardcore filling and compaction", "m³", 1800),
        ("Disposal of excavated material off-site", "m³", 320),
        
        # CONCRETE WORKS
        ("Concrete Class 15/20 foundation", "m³", 12500),
        ("Concrete Class 20/25 columns", "m³", 14200),
        ("Concrete Class 25/30 beams and slabs", "m³", 15800),
        ("Blinding concrete 1:3:6", "m³", 11000),
        ("Mass concrete for foundations", "m³", 13500),
        
        # REINFORCEMENT
        ("High tensile steel reinforcement Y8", "kg", 135),
        ("High tensile steel reinforcement Y10", "kg", 128),
        ("High tensile steel reinforcement Y12", "kg", 122),
        ("High tensile steel reinforcement Y16", "kg", 118),
        ("High tensile steel reinforcement Y20", "kg", 115),
        ("Mild steel reinforcement R6", "kg", 145),
        ("Binding wire", "kg", 180),
        
        # BLOCKWORK & MASONRY
        ("150mm solid concrete blocks (machine)", "m²", 1250),
        ("200mm solid concrete blocks (machine)", "m²", 1450),
        ("225mm solid concrete blocks (machine)", "m²", 1650),
        ("Machine cut stones 150mm thick", "m²", 2800),
        ("Machine cut stones 225mm thick", "m²", 3200),
        ("Common burnt bricks 225mm thick", "m²", 1150),
        
        # ROOFING
        ("Galvanized iron sheets 28 gauge", "m²", 650),
        ("Galvanized iron sheets 30 gauge", "m²", 580),
        ("Decra roofing tiles (stone coated)", "m²", 1850),
        ("Clay roof tiles", "m²", 1200),
        ("Timber purlins 2x3 inches", "m", 180),
        ("Timber purlins 2x4 inches", "m", 240),
        ("Ridge capping (metal)", "m", 320),
        
        # CARPENTRY & JOINERY
        ("Hardwood door frame 100x50mm", "m", 850),
        ("Flush door 2100x900mm", "No.", 8500),
        ("Panelled door 2100x900mm", "No.", 12000),
        ("Hardwood window frame 100x75mm", "m", 950),
        ("Aluminum sliding window", "m²", 4500),
        ("Kitchen cabinets (melamine finish)", "m", 12000),
        ("Timber roof trusses", "m²", 1200),
        
        # PLUMBING
        ("UPVC soil pipes 110mm", "m", 850),
        ("UPVC waste pipes 50mm", "m", 380),
        ("GI water pipes 25mm (3/4 inch)", "m", 650),
        ("PPR water pipes 25mm hot/cold", "m", 420),
        ("Water closet (WC) white suite", "No.", 8500),
        ("Wash hand basin with pedestal", "No.", 6500),
        ("Kitchen sink stainless steel double bowl", "No.", 7500),
        ("Water storage tank 5000 litres", "No.", 85000),
        
        # ELECTRICAL
        ("2.5mm² electrical cables (twin)", "m", 85),
        ("4.0mm² electrical cables (twin)", "m", 135),
        ("13A switched socket outlet", "No.", 450),
        ("Distribution board 8-way", "No.", 8500),
        ("Light switches 1-gang", "No.", 280),
        ("LED bulbs 15W", "No.", 350),
        ("PVC conduit 20mm", "m", 95),
        
        # FINISHES
        ("Cement screed 1:3, 25mm thick", "m²", 650),
        ("Ceramic floor tiles 400x400mm", "m²", 1850),
        ("Ceramic wall tiles 300x300mm", "m²", 1650),
        ("Terrazzo floor tiles 400x400mm", "m²", 2200),
        ("Plaster and skim coat internal walls", "m²", 580),
        ("External wall render and skim", "m²", 680),
        ("Emulsion paint 2 coats internal", "m²", 320),
        ("Oil paint 2 coats external", "m²", 420),
        ("Ceiling board (gypsum) 12mm", "m²", 950),
        
        # LABOR
        ("Skilled mason per day", "No.", 1500),
        ("General laborer per day", "No.", 800),
        ("Carpenter per day", "No.", 1800),
        ("Plumber per day", "No.", 2000),
        ("Electrician per day", "No.", 2200),
        ("Painter per day", "No.", 1600),
    ]
    
    # Generate dataset with all locations
    dataset = []
    for description, unit, nairobi_rate in base_items:
        for location in LOCATIONS:
            multiplier = LOCATION_MULTIPLIERS[location]
            rate = round(nairobi_rate * multiplier, 2)
            dataset.append({
                "Description": description,
                "Unit": unit,
                "Location": location,
                "Rate": rate
            })
    
    df = pd.DataFrame(dataset)
    return df


def load_or_create_database() -> pd.DataFrame:
    """
    Load the CSV database or create it if it doesn't exist.
    """
    if os.path.exists(CSV_DATABASE):
        try:
            df = pd.read_csv(CSV_DATABASE)
            # Validate structure
            required_cols = ["Description", "Unit", "Location", "Rate"]
            if all(col in df.columns for col in required_cols):
                return df
            else:
                st.warning("Database structure invalid. Regenerating...")
        except Exception as e:
            st.error(f"Error reading database: {e}. Regenerating...")
    
    # Generate new database
    df = generate_training_dataset()
    save_database(df)
    return df


def save_database(df: pd.DataFrame) -> bool:
    """
    Save the DataFrame to CSV file.
    Returns True if successful, False otherwise.
    """
    try:
        df.to_csv(CSV_DATABASE, index=False)
        return True
    except PermissionError:
        st.error(f"⚠️ Cannot save database. Please close '{CSV_DATABASE}' if it's open in Excel.")
        return False
    except Exception as e:
        st.error(f"Error saving database: {e}")
        return False


# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================

class CostPredictionModel:
    """
    Upgraded ML model with Confidence Scoring.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.rf = None
        self.train_vectors = None
        self.is_trained = False
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the ML model and store vectors for similarity checking.
        """
        # Combine description and location as features
        df['combined_features'] = df['Description'] + " " + df['Location']
        
        X = df['combined_features']
        y = df['Rate']
        
        # 1. Vectorize (Convert text to numbers)
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.train_vectors = self.vectorizer.fit_transform(X)
        
        # 2. Train Regressor (The Price Predictor)
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.rf.fit(self.train_vectors, y)
        
        self.is_trained = True
    
    def predict_with_confidence(self, description: str, location: str) -> Tuple[float, float]:
        """
        Predict rate AND return a confidence score (0.0 to 1.0).
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Vectorize input
        combined = description + " " + location
        input_vec = self.vectorizer.transform([combined])
        
        # 1. Predict Price
        prediction = self.rf.predict(input_vec)[0]
        
        # 2. Calculate Confidence (Cosine Similarity to training data)
        # Finds how close this item is to the closest item we have seen before
        similarity_scores = cosine_similarity(input_vec, self.train_vectors)
        confidence = similarity_scores.max()
        
        return max(0, prediction), confidence

# ============================================================================
# BUSINESS LOGIC FUNCTIONS
# ============================================================================

def calculate_estimate(description: str, location: str, quantity: float, 
                       model: CostPredictionModel) -> Tuple[float, float, float]:
    """
    Calculate rate, total, and confidence.
    """
    rate, conf = model.predict_with_confidence(description, location)
    total = rate * quantity
    return rate, total, conf

def process_batch_bq(df_upload: pd.DataFrame, description_col: str, 
                     quantity_col: str, location: str, 
                     model: CostPredictionModel) -> pd.DataFrame:
    """
    Process BQ and flag low-confidence predictions.
    """
    df_result = df_upload.copy()
    
    rates = []
    totals = []
    confidences = []
    notes = []
    
    progress_bar = st.progress(0)
    
    for idx, row in df_result.iterrows():
        description = str(row[description_col])
        quantity = float(row[quantity_col]) if pd.notna(row[quantity_col]) else 0
        
        try:
            # Use the new function that returns rate AND confidence
            rate, conf = model.predict_with_confidence(description, location)
            total = rate * quantity
            
            # Logic for Low Confidence
            if conf < 0.5: # Less than 50% sure
                note = "⚠️ LOW CONFIDENCE - Verify Rate"
            else:
                note = "✅ OK"
                
        except Exception as e:
            rate = 0
            total = 0
            conf = 0
            note = f"Error: {e}"
        
        rates.append(rate)
        totals.append(total)
        confidences.append(conf)
        notes.append(note)
        
        # Update progress bar
        if idx % 10 == 0:
            progress_bar.progress(min(1.0, idx / len(df_result)))
            
    progress_bar.empty()
    
    df_result['AI_Rate'] = rates
    df_result['Total_Cost'] = totals
    df_result['Confidence_%'] = [round(c * 100, 1) for c in confidences]
    df_result['Status'] = notes
    
    return df_result

def optimize_budget(df_bq: pd.DataFrame, target_budget: float) -> Tuple[pd.DataFrame, str, Dict]:
    """
    Smart Budget Optimizer: Adjust rates to meet target budget.
    """
    df_optimized = df_bq.copy()
    original_total = df_bq['Total_Cost'].sum()
    difference = original_total - target_budget
    
    decision_log = []
    adjustments = {}
    
    if abs(difference) / target_budget < 0.02:  # Within 2%
        decision_log.append("✅ Estimate is within 2% of target budget. No adjustments needed.")
        return df_optimized, "\n".join(decision_log), adjustments
    
    # Categorize items
    finish_keywords = ['paint', 'tile', 'ceiling', 'door', 'window', 'kitchen', 'finish']
    lowvis_keywords = ['excavation', 'concrete', 'hardcore', 'backfill', 'foundation', 'steel']
    
    df_optimized['Category'] = 'Standard'
    for idx, row in df_optimized.iterrows():
        desc_lower = str(row['Description']).lower() if 'Description' in row else ''
        if any(kw in desc_lower for kw in finish_keywords):
            df_optimized.at[idx, 'Category'] = 'Finish'
        elif any(kw in desc_lower for kw in lowvis_keywords):
            df_optimized.at[idx, 'Category'] = 'LowVisibility'
    
    if difference > 0:  # Over budget
        decision_log.append(f"⚠️ Over budget by KES {difference:,.2f}")
        decision_log.append("Strategy: Reduce rates on high-visibility items (Finishes)")
        
        finish_items = df_optimized[df_optimized['Category'] == 'Finish']
        if len(finish_items) > 0:
            # Reduce finish items by up to 8%
            reduction_factor = min(0.08, difference / finish_items['Total_Cost'].sum())
            
            for idx in finish_items.index:
                old_rate = df_optimized.at[idx, 'AI_Predicted_Rate']
                new_rate = old_rate * (1 - reduction_factor)
                df_optimized.at[idx, 'AI_Predicted_Rate'] = new_rate
                df_optimized.at[idx, 'Total_Cost'] = new_rate * df_optimized.at[idx, df_optimized.columns[1]]
                
                adjustments[idx] = {
                    'item': df_optimized.at[idx, 'Description'],
                    'old_rate': old_rate,
                    'new_rate': new_rate,
                    'change_pct': -reduction_factor * 100
                }
            
            decision_log.append(f"Applied {reduction_factor*100:.1f}% reduction to {len(finish_items)} finish items")
    
    else:  # Under budget
        decision_log.append(f"💰 Under budget by KES {abs(difference):,.2f}")
        decision_log.append("Strategy: Increase margins on low-visibility items (Structure)")
        
        lowvis_items = df_optimized[df_optimized['Category'] == 'LowVisibility']
        if len(lowvis_items) > 0:
            # Increase low-vis items by up to 10%
            increase_factor = min(0.10, abs(difference) / lowvis_items['Total_Cost'].sum())
            
            for idx in lowvis_items.index:
                old_rate = df_optimized.at[idx, 'AI_Predicted_Rate']
                new_rate = old_rate * (1 + increase_factor)
                df_optimized.at[idx, 'AI_Predicted_Rate'] = new_rate
                df_optimized.at[idx, 'Total_Cost'] = new_rate * df_optimized.at[idx, df_optimized.columns[1]]
                
                adjustments[idx] = {
                    'item': df_optimized.at[idx, 'Description'],
                    'old_rate': old_rate,
                    'new_rate': new_rate,
                    'change_pct': increase_factor * 100
                }
            
            decision_log.append(f"Applied {increase_factor*100:.1f}% increase to {len(lowvis_items)} structural items")
    
    new_total = df_optimized['Total_Cost'].sum()
    decision_log.append(f"\n📊 Original Total: KES {original_total:,.2f}")
    decision_log.append(f"📊 Optimized Total: KES {new_total:,.2f}")
    decision_log.append(f"📊 Target Budget: KES {target_budget:,.2f}")
    decision_log.append(f"📊 Variance: KES {new_total - target_budget:,.2f}")
    
    return df_optimized, "\n".join(decision_log), adjustments


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar with branding."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1E3A8A/FFFFFF?text=JengaCost+AI", 
                 use_container_width=True)
        st.markdown("---")
        st.markdown(f"**Version:** {APP_VERSION}")
        st.markdown("**Market:** Kenya 🇰🇪")
        st.markdown("---")
        st.markdown("### About")
        st.info("Professional construction cost estimation powered by AI. Trained on Kenyan market rates.")
        st.markdown("---")
        st.markdown("© 2025 JengaCost AI")


def render_ai_pricing_engine(model: CostPredictionModel, df: pd.DataFrame):
    """Module A: AI Pricing Engine"""
    st.header("🤖 AI Pricing Engine")
    st.markdown("Get instant cost predictions for construction items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Searchable dropdown for descriptions
        descriptions = sorted(df['Description'].unique().tolist())
        selected_desc = st.selectbox(
            "Item Description",
            options=descriptions,
            help="Start typing to search"
        )
        
        location = st.selectbox("Location", LOCATIONS)
    
    with col2:
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.2f"
        )
        
        # Get unit from database
        unit = df[df['Description'] == selected_desc]['Unit'].iloc[0]
        st.text_input("Unit", value=unit, disabled=True)
    
    if st.button("Calculate Estimate", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            rate, total = calculate_estimate(selected_desc, location, quantity, model)
            
            st.success("✅ Estimate Generated!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rate per Unit", f"KES {rate:,.2f}")
            col2.metric("Quantity", f"{quantity:,.2f} {unit}")
            col3.metric("Total Estimate", f"KES {total:,.2f}")
            
            # Show historical data
            historical = df[(df['Description'] == selected_desc) & (df['Location'] == location)]
            if not historical.empty:
                st.info(f"📊 Market Rate: KES {historical['Rate'].iloc[0]:,.2f}")


def render_batch_processor(model: CostPredictionModel):
    """Module B: Batch BQ Processor"""
    st.header("📊 Batch BQ Processor")
    st.markdown("Upload your Bill of Quantities and get AI-powered pricing")
    
    uploaded_file = st.file_uploader(
        "Upload Excel File (.xlsx)",
        type=['xlsx'],
        help="Upload your unpriced Bill of Quantities"
    )
    
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df_upload)} rows found")
            
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df_upload.head(10), use_container_width=True)
            
            st.subheader("Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                desc_col = st.selectbox("Description Column", df_upload.columns)
            with col2:
                qty_col = st.selectbox("Quantity Column", df_upload.columns)
            with col3:
                location = st.selectbox("Project Location", LOCATIONS)
            
            if st.button("Process BQ with AI", type="primary", use_container_width=True):
                with st.spinner("Processing... This may take a moment"):
                    df_result = process_batch_bq(df_upload, desc_col, qty_col, location, model)
                    
                    st.success("✅ Processing Complete!")
                    
                    # Show summary
                    total_cost = df_result['Total_Cost'].sum()
                    st.metric("Total Project Cost", f"KES {total_cost:,.2f}")
                    
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Download button
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Priced_BQ')
                    
                    st.download_button(
                        label="📥 Download Priced BQ",
                        data=output.getvalue(),
                        file_name=f"Priced_BQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_budget_optimizer(model: CostPredictionModel):
    """Module C: Smart Budget Optimizer"""
    st.header("💡 Smart Budget Optimizer")
    st.markdown("Optimize your BQ to meet target budgets intelligently")
    
    uploaded_file = st.file_uploader(
        "Upload Priced BQ (.xlsx)",
        type=['xlsx'],
        key="optimizer_upload",
        help="Upload a BQ with rates already calculated"
    )
    
    if uploaded_file:
        try:
            df_bq = pd.read_excel(uploaded_file)
            
            # Try to identify rate and total columns
            rate_cols = [col for col in df_bq.columns if 'rate' in col.lower() or 'price' in col.lower()]
            total_cols = [col for col in df_bq.columns if 'total' in col.lower() or 'amount' in col.lower()]
            desc_cols = [col for col in df_bq.columns if 'desc' in col.lower() or 'item' in col.lower()]
            
            st.subheader("Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                desc_col = st.selectbox("Description Column", df_bq.columns, 
                                       index=df_bq.columns.tolist().index(desc_cols[0]) if desc_cols else 0,
                                       key="opt_desc")
            with col2:
                rate_col = st.selectbox("Rate Column", df_bq.columns,
                                       index=df_bq.columns.tolist().index(rate_cols[0]) if rate_cols else 0,
                                       key="opt_rate")
            with col3:
                total_col = st.selectbox("Total Column", df_bq.columns,
                                        index=df_bq.columns.tolist().index(total_cols[0]) if total_cols else 0,
                                        key="opt_total")
            
            # Rename columns for processing
            df_bq_proc = df_bq.copy()
            df_bq_proc = df_bq_proc.rename(columns={
                desc_col: 'Description',
                rate_col: 'AI_Predicted_Rate',
                total_col: 'Total_Cost'
            })
            
            current_total = df_bq_proc['Total_Cost'].sum()
            
            st.metric("Current Total Cost", f"KES {current_total:,.2f}")
            
            target_budget = st.number_input(
                "Target Budget (KES)",
                min_value=0.0,
                value=float(current_total),
                step=100000.0,
                format="%.2f"
            )
            
            if st.button("🎯 Optimize to Target", type="primary", use_container_width=True):
                with st.spinner("Optimizing budget allocation..."):
                    df_optimized, decision_log, adjustments = optimize_budget(df_bq_proc, target_budget)
                    
                    st.success("✅ Optimization Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Total", f"KES {current_total:,.2f}")
                    with col2:
                        optimized_total = df_optimized['Total_Cost'].sum()
                        st.metric("Optimized Total", f"KES {optimized_total:,.2f}",
                                 delta=f"{optimized_total - current_total:,.2f}")
                    
                    st.subheader("Decision Log")
                    st.text(decision_log)
                    
                    if adjustments:
                        st.subheader("Rate Adjustments")
                        adj_df = pd.DataFrame([
                            {
                                'Item': adj['item'],
                                'Old Rate': f"KES {adj['old_rate']:,.2f}",
                                'New Rate': f"KES {adj['new_rate']:,.2f}",
                                'Change %': f"{adj['change_pct']:+.1f}%"
                            }
                            for adj in adjustments.values()
                        ])
                        st.dataframe(adj_df, use_container_width=True)
                    
                    # Download optimized BQ
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_optimized.to_excel(writer, index=False, sheet_name='Optimized_BQ')
                    
                    st.download_button(
                        label="📥 Download Optimized BQ",
                        data=output.getvalue(),
                        file_name=f"Optimized_BQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def render_admin_panel(df: pd.DataFrame):
    """Module D: Admin Panel with CMS & OCR"""
    st.header("🔐 Admin Panel")
    
    # --- PASSWORD LOGIC ---
    try:
        # Try to load the secure password from Streamlit Cloud Secrets
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    except (FileNotFoundError, KeyError):
        # Fallback ONLY if secrets are missing (e.g. local testing)
        # You can change this to your strong password if testing locally
        ADMIN_PASSWORD = "admin123" 
    
    # --- AUTHENTICATION ---
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid password")
        return
    
    st.success("✅ Authenticated")
    if st.button("Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # --- TABS FOR ENTRY METHODS ---
    tab1, tab2, tab3 = st.tabs(["➕ Add Item (Manual)", "📸 Add via Camera (OCR)", "✏️ Edit Database"])
    
    # TAB 1: MANUAL ENTRY
    with tab1:
        st.subheader("Manual Data Entry")
        with st.form("add_item_form"):
            description = st.text_input("Item Description*")
            unit = st.selectbox("Unit*", UNITS)
            location = st.selectbox("Location*", LOCATIONS)
            rate = st.number_input("Market Rate (KES)*", min_value=0.0, step=10.0)
            
            if st.form_submit_button("Add Item", type="primary"):
                if description and rate > 0:
                    new_row = pd.DataFrame([{"Description": description, "Unit": unit, "Location": location, "Rate": rate}])
                    df_updated = pd.concat([df, new_row], ignore_index=True)
                    if save_database(df_updated):
                        st.success(f"✅ Added: {description}")
                        st.rerun()
    
    # TAB 2: CAMERA / OCR ENTRY
    with tab2:
        st.subheader("📸 AI Receipt Scanner")
        st.info("Snap a photo of a receipt or price tag. The AI will try to read it.")
        
        img_file = st.camera_input("Take a Picture")
        
        if img_file:
            # 1. Show the image
            image = Image.open(img_file)
            st.image(image, caption="Captured Image", width=300)
            
            # 2. Extract Text
            with st.spinner("🤖 AI is reading text..."):
                try:
                    # Configure tesseract path if needed (usually auto-detected on cloud)
                    extracted_text = pytesseract.image_to_string(image)
                    st.text_area("Raw Extracted Text", extracted_text, height=100)
                    
                    # 3. Auto-Fill Form
                    st.markdown("---")
                    st.subheader("Verify & Save")
                    
                    # Try to guess values (Basic logic - improves with time)
                    guessed_desc = extracted_text.split('\n')[0] if extracted_text else ""
                    
                    with st.form("ocr_save"):
                        c1, c2 = st.columns(2)
                        ocr_desc = c1.text_input("Description", value=guessed_desc)
                        ocr_unit = c2.selectbox("Unit", UNITS)
                        
                        c3, c4 = st.columns(2)
                        ocr_loc = c3.selectbox("Location", LOCATIONS)
                        ocr_rate = c4.number_input("Rate (KES)", min_value=0.0)
                        
                        if st.form_submit_button("💾 Save to Database"):
                            new_row = pd.DataFrame([{"Description": ocr_desc, "Unit": ocr_unit, "Location": ocr_loc, "Rate": ocr_rate}])
                            df_updated = pd.concat([df, new_row], ignore_index=True)
                            if save_database(df_updated):
                                st.success("✅ Saved from OCR!")
                except Exception as e:
                    st.error(f"OCR Error: {e}. Make sure 'tesseract-ocr' is in packages.txt")

    # TAB 3: EDIT DATABASE
    with tab3:
        st.subheader("Master Database")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save Changes"):
            if save_database(edited_df):
                st.success("✅ Database updated!")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enterprise look
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        h1 { color: #1E3A8A; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] {
            background-color: #F3F4F6;
            border-radius: 4px 4px 0 0;
            padding: 1rem 2rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    render_sidebar()
    
    # Load database
    df = load_or_create_database()
    
    # Train model
    if 'model' not in st.session_state:
        with st.spinner("Initializing AI model..."):
            model = CostPredictionModel()
            model.train(df)
            st.session_state.model = model
    else:
        model = st.session_state.model
    
    # Main title
    st.title("🏗️ JengaCost AI")
    st.markdown("**Professional Construction Cost Estimation for Kenya** | Powered by Machine Learning")
    st.markdown("---")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI Pricing Engine",
        "📊 Batch BQ Processor",
        "💡 Budget Optimizer",
        "🔐 Admin Panel"
    ])
    
    with tab1:
        render_ai_pricing_engine(model, df)
    
    with tab2:
        render_batch_processor(model)
    
    with tab3:
        render_budget_optimizer(model)
    
    with tab4:
        render_admin_panel(df)


if __name__ == "__main__":
    main()