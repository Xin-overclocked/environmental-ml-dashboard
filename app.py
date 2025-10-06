import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from supabase import create_client, Client
import os
from datetime import datetime
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Environmental Engineering Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Exo 2', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #00ff88;
    }
    
    /* Sidebar navigation */
    .css-17eq0hr {
        background: rgba(0, 255, 136, 0.1);
        border-radius: 10px;
        border: 1px solid #00ff88;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    .css-17eq0hr:hover {
        background: rgba(0, 255, 136, 0.2);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        transform: translateX(5px);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 123, 255, 0.1) 100%);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.4);
        transform: translateY(-5px);
    }
    
    .kpi-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        color: #00ff88;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.7);
    }
    
    .kpi-label {
        font-family: 'Exo 2', sans-serif;
        font-size: 1rem;
        color: #ffffff;
        opacity: 0.8;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ff88, #007bff);
        border: none;
        border-radius: 10px;
        color: white;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #00ff88;
        border-radius: 10px;
        background: rgba(0, 255, 136, 0.05);
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(255, 165, 0, 0.1) 100%);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 2px solid #00ff88;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.2);
        border: 1px solid #00ff88;
        border-radius: 10px;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.2);
        border: 1px solid #ff0000;
        border-radius: 10px;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client with proper credential handling"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets:
            url = st.secrets['SUPABASE_URL']
            key = st.secrets['SUPABASE_ANON_KEY']
        # Fall back to environment variables (for local development)
        else:
            url = os.environ.get('SUPABASE_URL') or os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
            key = os.environ.get('SUPABASE_ANON_KEY') or os.environ.get('NEXT_PUBLIC_SUPABASE_ANON_KEY')
        
        # Validate credentials
        if not url or not key:
            st.error("⚠️ Supabase credentials not found. Please configure them in Streamlit Cloud secrets or local environment variables.")
            st.info(r"""
            **For Streamlit Cloud:**
            1. Go to App Settings → Secrets
            2. Add:
               ```
               SUPABASE_URL = "your_supabase_url"
               SUPABASE_ANON_KEY = "your_anon_key"
               ```
            
            **For Local Development:**
            Create `.streamlit/secrets.toml` with the same format.
            """)
            st.stop()
        
        return create_client(url, key)
    
    except Exception as e:
        st.error(f"❌ Failed to initialize Supabase: {str(e)}")
        st.stop()

supabase = init_supabase()

# Helper functions
def create_kpi_card(title, value, unit="", color="#00ff88"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: {color};">{value}{unit}</div>
        <div class="kpi-label">{title}</div>
    </div>
    """

def load_training_data():
    """Load training data from Supabase"""
    try:
        response = supabase.table('training_data').select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return pd.DataFrame()

def save_training_data(df):
    """Save training data to Supabase"""
    try:
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        for record in records:
            # Convert numpy types to Python types
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        response = supabase.table('training_data').insert(records).execute()
        return True
    except Exception as e:
        error_msg = str(e)
        if 'PGRST205' in error_msg or 'Could not find the table' in error_msg:
            st.error(f"❌ Database table not found. Please run the SQL setup script first.")
            st.info("""
            **Setup Instructions:**
            1. Go to your Supabase dashboard
            2. Navigate to SQL Editor
            3. Run the script from `scripts/001_create_tables.sql`
            4. Refresh this page and try again
            """)
        else:
            st.error(f"Error saving training data: {error_msg}")
        return False

def save_prediction(prediction_data):
    """Save prediction to Supabase"""
    try:
        response = supabase.table('predictions').insert(prediction_data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False

def predict_emissions(vehicle_data, training_df):
    """Predict emissions based on vehicle data and training dataset"""
    if training_df.empty:
        # Default predictions if no training data
        return {
            'cleaning_time': 45.0,
            'hc_reduction': 25.0,
            'co_reduction': 30.0,
            'nox_reduction': 20.0,
            'particulates_reduction': 35.0,
            'co2_saved': 150.0
        }
    
    # Simple prediction model based on engine characteristics
    base_cleaning_time = 30 + (vehicle_data['engine_cc'] / 100) + (vehicle_data['odometer_reading'] / 10000)
    
    # Fuel quality factor
    fuel_factor = 1.2 if vehicle_data['fuel_quality'] == 'Regular' else 0.8
    
    # Age factor
    age_factor = 1 + ((2024 - vehicle_data['vehicle_year']) * 0.05)
    
    cleaning_time = base_cleaning_time * fuel_factor * age_factor
    
    # Emission reductions (percentages)
    hc_reduction = min(40, 15 + (cleaning_time / 2))
    co_reduction = min(45, 20 + (cleaning_time / 2.5))
    nox_reduction = min(35, 10 + (cleaning_time / 3))
    particulates_reduction = min(50, 25 + (cleaning_time / 2))
    
    # CO2 equivalent saved (g/km)
    co2_saved = (hc_reduction + co_reduction + nox_reduction) * 2.5
    
    return {
        'cleaning_time': round(cleaning_time, 1),
        'hc_reduction': round(hc_reduction, 1),
        'co_reduction': round(co_reduction, 1),
        'nox_reduction': round(nox_reduction, 1),
        'particulates_reduction': round(particulates_reduction, 1),
        'co2_saved': round(co2_saved, 1)
    }

# Define the page variable for navigation
page = "Main Dashboard"  # Default page

# Sidebar Navigation
st.sidebar.markdown("# 🌱 Environmental Dashboard")
st.sidebar.markdown("---")

# Add a new tab for the vehicle dashboard
selected_tab = st.sidebar.radio("Select Tab", ["Main Dashboard", "Vehicle Dashboard", "Training Process"])

if selected_tab == "Vehicle Dashboard":
    page = "Vehicle Dashboard"
    st.title("Vehicle Dashboard")
    
    # Vehicle Details Section with validation
    st.subheader("🚗 Vehicle Details")
    with st.form("vehicle_form"):
        col1, col2 = st.columns(2)
        with col1:
            vehicle_model = st.text_input("Vehicle Model*", key="vehicle_model_dashboard_tab")
            vehicle_year = st.number_input("Vehicle Year*", min_value=1900, max_value=2025, value=2020, key="vehicle_year_dashboard_tab")
            vehicle_meter = st.number_input("Vehicle Meter Reading (KM)*", min_value=0, value=0, key="vehicle_meter_dashboard_tab")
        with col2:
            vehicle_cc = st.number_input("Vehicle CC*", min_value=0, value=1500, key="vehicle_cc_dashboard_tab")
            fuel_quality = st.selectbox("Fuel Quality*", ["Regular", "Premium", "Super"], key="fuel_quality_dashboard_tab")
            fuel_brand = st.text_input("Fuel Brand*", key="fuel_brand_dashboard_tab")

        # Emissions Section
        st.subheader("📊 Emissions Measurements")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Before Treatment**")
            hc_b = st.number_input("HC (ppm) - Before*", min_value=0, key="hc_b_dashboard_tab")
            co_b = st.number_input("CO (ppm) - Before*", min_value=0, key="co_b_dashboard_tab")
            o2_b = st.number_input("O2% - Before*", min_value=0.0, max_value=100.0, key="o2_b_dashboard_tab")
            nox_b = st.number_input("NOx (ppm) - Before*", min_value=0, key="nox_b_dashboard_tab")
        with col4:
            st.markdown("**After Treatment**")
            hc_a = st.number_input("HC (ppm) - After*", min_value=0, key="hc_a_dashboard_tab")
            co_a = st.number_input("CO (ppm) - After*", min_value=0, key="co_a_dashboard_tab")
            o2_a = st.number_input("O2% - After*", min_value=0.0, max_value=100.0, key="o2_a_dashboard_tab")
            nox_a = st.number_input("NOx (ppm) - After*", min_value=0, key="nox_a_dashboard_tab")

        # Performance Section
        st.subheader("⚡ Performance Metrics")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Before Treatment**")
            afr_b = st.number_input("AFR - Before*", min_value=0.0, key="afr_b_dashboard_tab")
            lambda_b = st.number_input("Lambda - Before*", min_value=0.0, key="lambda_b_dashboard_tab")
            rpm_b = st.number_input("RPM - Before*", min_value=0, key="rpm_b_dashboard_tab")
            engine_load_b = st.number_input("Engine Load (%) - Before*", min_value=0.0, max_value=100.0, key="engine_load_b_dashboard_tab")
        with col6:
            st.markdown("**After Treatment**")
            afr_a = st.number_input("AFR - After*", min_value=0.0, key="afr_a_dashboard_tab")
            lambda_a = st.number_input("Lambda - After*", min_value=0.0, key="lambda_a_dashboard_tab")
            rpm_a = st.number_input("RPM - After*", min_value=0, key="rpm_a_dashboard_tab")
            engine_load_a = st.number_input("Engine Load (%) - After*", min_value=0.0, max_value=100.0, key="engine_load_a_dashboard_tab")

        submitted = st.form_submit_button("Save and Analyze Results", type="primary")

        if submitted:
            # Calculate reductions and improvements
            hc_reduction = ((hc_b - hc_a) / hc_b * 100) if hc_b > 0 else 0
            co_reduction = ((co_b - co_a) / co_b * 100) if co_b > 0 else 0
            nox_reduction = ((nox_b - nox_a) / nox_b * 100) if nox_b > 0 else 0

            # Display metrics
            st.subheader("📈 Results Analysis")
            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("HC Reduction", f"{hc_reduction:.1f}%", 
                         f"{hc_b - hc_a:.1f} ppm")
            with met2:
                st.metric("CO Reduction", f"{co_reduction:.1f}%",
                         f"{co_b - co_a:.1f} ppm")
            with met3:
                st.metric("NOx Reduction", f"{nox_reduction:.1f}%",
                         f"{nox_b - nox_a:.1f} ppm")

            # Create visualization
            st.subheader("📊 Emissions Comparison")
            emissions_data = {
                'Parameter': ['HC (ppm)', 'CO (ppm)', 'NOx (ppm)'],
                'Before': [hc_b, co_b, nox_b],
                'After': [hc_a, co_a, nox_a]
            }
            df = pd.DataFrame(emissions_data)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Before',
                x=df['Parameter'],
                y=df['Before'],
                marker_color='#ff6b6b'
            ))
            fig.add_trace(go.Bar(
                name='After',
                x=df['Parameter'],
                y=df['After'],
                marker_color='#00ff88'
            ))

            fig.update_layout(
                title="Emissions Before vs After Treatment",
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=True
            )

            st.plotly_chart(fig, width="stretch")

            # Save to database
            try:
                data = {
                    "vehicle_model": vehicle_model,
                    "vehicle_year": vehicle_year,
                    "meter_reading": vehicle_meter,
                    "vehicle_cc": vehicle_cc,
                    "fuel_quality": fuel_quality,
                    "fuel_brand": fuel_brand,
                    "hc_before": hc_b,
                    "hc_after": hc_a,
                    "co_before": co_b,
                    "co_after": co_a,
                    "nox_before": nox_b,
                    "nox_after": nox_a,
                    "o2_before": o2_b,
                    "o2_after": o2_a,
                    "afr_before": afr_b,
                    "afr_after": afr_a,
                    "lambda_before": lambda_b,
                    "lambda_after": lambda_a,
                    "rpm_before": rpm_b,
                    "rpm_after": rpm_a,
                    "engine_load_before": engine_load_b,
                    "engine_load_after": engine_load_a,
                    "timestamp": datetime.now().isoformat()
                }
                
                response = supabase.table('vehicle_measurements').insert(data).execute()
                st.success("✅ Data saved successfully!")
            except Exception as e:
                st.error(f"❌ Error saving data: {str(e)}")

    # Help section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. Fill in all required fields marked with *
        2. Enter measurements before and after treatment
        3. Click 'Save and Analyze Results' to:
           - View reduction percentages
           - See comparison charts
           - Save data to database
        """)

elif selected_tab == "Training Process":
    page = "Training Process"

# Main content based on selected page
if page == "Training Process":
    st.markdown("# 🔬 Training Process")
    st.markdown("### Research & Data Preparation")
    
    # File upload section
    st.markdown("## 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV containing vehicle and emissions data",
        type=['csv'],
        help="CSV should contain: Vehicle Info, Emissions (HC, CO, NOx, Smoke), OBD Metrics (AFR, Lambda, O₂, RPM, Engine Load)"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            # Display dataset summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_kpi_card("Total Rows", len(df)), unsafe_allow_html=True)
            
            with col2:
                missing_values = df.isnull().sum().sum()
                st.markdown(create_kpi_card("Missing Values", missing_values, color="#ff6b6b"), unsafe_allow_html=True)
            
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    min_val = df[numeric_cols].min().min()
                    st.markdown(create_kpi_card("Min Value", f"{min_val:.2f}"), unsafe_allow_html=True)
            
            with col4:
                if len(numeric_cols) > 0:
                    max_val = df[numeric_cols].max().max()
                    st.markdown(create_kpi_card("Max Value", f"{max_val:.2f}"), unsafe_allow_html=True)
            
            # Editable data table with pagination
            st.markdown("## 📊 Dataset Preview & Editing")
            
            # Pagination
            rows_per_page = 50
            total_pages = (len(df) - 1) // rows_per_page + 1
            
            if total_pages > 1:
                page_num = st.selectbox("Select Page", range(1, total_pages + 1))
                start_idx = (page_num - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(df))
                df_page = df.iloc[start_idx:end_idx]
            else:
                df_page = df
            
            # Editable dataframe
            edited_df = st.data_editor(
                df_page,
                width="stretch",
                num_rows="dynamic",
                key="data_editor"
            )
            
            # Save to database
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("💾 Save to Database", type="primary"):
                    if save_training_data(edited_df):
                        st.success("✅ Data saved to database successfully!")
                    else:
                        st.error("❌ Failed to save data to database.")
            
            with col2:
                # Export options
                csv = edited_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Correlation Analysis
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                st.markdown("## 📈 Correlation Analysis")
                
                # Scatter plots with regression
                emissions_cols = [col for col in df.columns if any(x in col.lower() for x in ['hc', 'co', 'nox', 'smoke'])]
                obd_cols = [col for col in df.columns if any(x in col.lower() for x in ['afr', 'lambda', 'o2', 'rpm', 'load'])]
                
                if emissions_cols and obd_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        emission_col = st.selectbox("Select Emission", emissions_cols)
                    
                    with col2:
                        obd_col = st.selectbox("Select OBD Metric", obd_cols)
                    
                    if emission_col in df.columns and obd_col in df.columns:
                        # Create scatter plot with regression
                        fig = px.scatter(
                            df, 
                            x=obd_col, 
                            y=emission_col,
                            trendline="ols",
                            title=f"{emission_col} vs {obd_col}",
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                st.markdown("### 🔥 Correlation Heatmap")
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix: Emissions vs OBD Metrics",
                        template="plotly_dark",
                        color_continuous_scale="RdBu"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Load existing data from database
    st.markdown("## 🗄️ Existing Training Data")
    existing_data = load_training_data()
    
    if not existing_data.empty:
        st.info(f"📊 {len(existing_data)} records found in database")
        
        # Show summary of existing data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_kpi_card("Total Records", len(existing_data)), unsafe_allow_html=True)
        
        with col2:
            unique_models = existing_data['vehicle_model'].nunique() if 'vehicle_model' in existing_data.columns else 0
            st.markdown(create_kpi_card("Vehicle Models", unique_models), unsafe_allow_html=True)
        
        with col3:
            date_range = "N/A"
            if 'created_at' in existing_data.columns:
                try:
                    dates = pd.to_datetime(existing_data['created_at'])
                    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                except:
                    pass
            st.markdown(create_kpi_card("Date Range", date_range), unsafe_allow_html=True)
        
        # Preview existing data
        if st.checkbox("Show existing data preview"):
            st.dataframe(existing_data.head(10), use_container_width=True)
    else:
        st.info("📝 No training data found in database. Upload CSV files to get started.")

elif page == "Vehicle Dashboard":
    pass  # Vehicle Dashboard is already defined in the selected_tab section

else:  # Main Dashboard
    st.title("Main Dashboard")
    st.markdown("Welcome to the Environmental Engineering Dashboard.")
    st.markdown("Use the sidebar to navigate between different sections.")
    st.markdown("### Quick Insights")
    
    # Sample KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_kpi_card("Total Vehicles Analyzed", "1,250"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card("Average CO₂ Saved per Vehicle", "120 g/km"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card("Total Cleaning Sessions", "3,500"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Latest Predictions")
    
    # Sample table of latest predictions
    sample_predictions = pd.DataFrame({
        'Vehicle Model': ['Toyota Camry', 'Honda Accord', 'Ford Focus'],
        'Predicted Cleaning Time (mins)': [45, 50, 40],
        'CO₂ Equivalent Saved (g/km)': [130, 110, 150]
    })
    
    st.dataframe(sample_predictions, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #00ff88; font-family: Orbitron;'>
        🌱 Environmental Engineering Dashboard | Powered by Streamlit & Supabase
    </div>
    """, 
    unsafe_allow_html=True
)
