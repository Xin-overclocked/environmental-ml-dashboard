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
            st.info("""
            **For Streamlit Cloud:**
            1. Go to App Settings → Secrets
            2. Add:
               \`\`\`
               SUPABASE_URL = "your_supabase_url"
               SUPABASE_ANON_KEY = "your_anon_key"
               \`\`\`
            
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

# Sidebar Navigation
st.sidebar.markdown("# 🌱 Environmental Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    ["Training Process", "Prediction Process", "Reports & Analytics"],
    key="navigation"
)

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
                use_container_width=True,
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

elif page == "Prediction Process":
    st.markdown("# 🔮 Prediction Process")
    st.markdown("### Vehicle Emissions Prediction")
    
    # Load training data for predictions
    training_data = load_training_data()
    
    # Vehicle input form
    st.markdown("## 🚗 Vehicle Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_model = st.text_input("Vehicle Model", placeholder="e.g., Toyota Camry")
        vehicle_year = st.number_input("Vehicle Year", min_value=1990, max_value=2024, value=2020)
        engine_cc = st.number_input("Engine CC", min_value=800, max_value=8000, value=2000)
    
    with col2:
        odometer_reading = st.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=50000)
        fuel_quality = st.selectbox("Fuel Quality", ["Premium", "Regular", "Super"])
        fuel_brand = st.text_input("Fuel Brand", placeholder="e.g., Shell, BP, Exxon")
    
    if st.button("🔍 Generate Prediction", type="primary"):
        if vehicle_model and fuel_brand:
            # Prepare vehicle data
            vehicle_data = {
                'vehicle_model': vehicle_model,
                'vehicle_year': vehicle_year,
                'engine_cc': engine_cc,
                'odometer_reading': odometer_reading,
                'fuel_quality': fuel_quality,
                'fuel_brand': fuel_brand
            }
            
            # Generate predictions
            predictions = predict_emissions(vehicle_data, training_data)
            
            # Display KPI cards
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_kpi_card(
                    "Cleaning Time", 
                    predictions['cleaning_time'], 
                    " mins",
                    "#00ff88"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_kpi_card(
                    "CO₂ Equivalent Saved", 
                    predictions['co2_saved'], 
                    " g/km",
                    "#007bff"
                ), unsafe_allow_html=True)
            
            with col3:
                avg_reduction = (predictions['hc_reduction'] + predictions['co_reduction'] + 
                               predictions['nox_reduction'] + predictions['particulates_reduction']) / 4
                st.markdown(create_kpi_card(
                    "Avg. Reduction", 
                    f"{avg_reduction:.1f}", 
                    "%",
                    "#ffa500"
                ), unsafe_allow_html=True)
            
            # Emissions reduction chart
            st.markdown("## 📈 Emissions Reduction Analysis")
            
            # Before vs After bar chart
            emissions = ['HC (ppm)', 'CO (ppm)', 'NOx (ppm)', 'Particulates (mg/m³)']
            before_values = [100, 100, 100, 100]  # Baseline 100%
            after_values = [
                100 - predictions['hc_reduction'],
                100 - predictions['co_reduction'],
                100 - predictions['nox_reduction'],
                100 - predictions['particulates_reduction']
            ]
            
            fig = go.Figure(data=[
                go.Bar(name='Before Cleaning', x=emissions, y=before_values, marker_color='#ff6b6b'),
                go.Bar(name='After Cleaning', x=emissions, y=after_values, marker_color='#00ff88')
            ])
            
            fig.update_layout(
                title='Before vs After Emissions Comparison',
                xaxis_title='Emission Type',
                yaxis_title='Relative Level (%)',
                barmode='group',
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Real-time style chart (simulated RPM vs HC)
            st.markdown("## ⚡ Real-time Simulation: RPM vs HC Levels")
            
            # Generate simulated real-time data
            rpm_values = np.linspace(800, 6000, 50)
            hc_before = 150 + (rpm_values - 800) * 0.02 + np.random.normal(0, 10, 50)
            hc_after = hc_before * (1 - predictions['hc_reduction']/100)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rpm_values, 
                y=hc_before,
                mode='lines+markers',
                name='Before Cleaning',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=rpm_values, 
                y=hc_after,
                mode='lines+markers',
                name='After Cleaning',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title='RPM vs HC Emissions (Real-time Simulation)',
                xaxis_title='RPM',
                yaxis_title='HC Levels (ppm)',
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save prediction and export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Prediction"):
                    prediction_record = {
                        **vehicle_data,
                        'predicted_cleaning_time': predictions['cleaning_time'],
                        'predicted_hc_reduction': predictions['hc_reduction'],
                        'predicted_co_reduction': predictions['co_reduction'],
                        'predicted_nox_reduction': predictions['nox_reduction'],
                        'predicted_particulates_reduction': predictions['particulates_reduction'],
                        'co2_equivalent_saved': predictions['co2_saved']
                    }
                    
                    if save_prediction(prediction_record):
                        st.success("✅ Prediction saved successfully!")
                    else:
                        st.error("❌ Failed to save prediction.")
            
            with col2:
                # Export CSV
                export_data = pd.DataFrame([{
                    **vehicle_data,
                    **predictions
                }])
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Export Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_data.to_excel(writer, sheet_name='Prediction', index=False)
                
                st.download_button(
                    label="📊 Export Excel",
                    data=buffer.getvalue(),
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        else:
            st.error("❌ Please fill in all required fields (Vehicle Model and Fuel Brand)")

else:  # Reports & Analytics
    st.markdown("# 📊 Reports & Analytics")
    st.markdown("### Comprehensive Data Analysis")
    
    # Load data
    training_data = load_training_data()
    
    try:
        predictions_response = supabase.table('predictions').select('*').execute()
        predictions_data = pd.DataFrame(predictions_response.data) if predictions_response.data else pd.DataFrame()
    except:
        predictions_data = pd.DataFrame()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_kpi_card("Training Records", len(training_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card("Predictions Made", len(predictions_data)), unsafe_allow_html=True)
    
    with col3:
        if not predictions_data.empty and 'predicted_cleaning_time' in predictions_data.columns:
            avg_cleaning_time = predictions_data['predicted_cleaning_time'].mean()
            st.markdown(create_kpi_card("Avg Cleaning Time", f"{avg_cleaning_time:.1f}", " mins"), unsafe_allow_html=True)
        else:
            st.markdown(create_kpi_card("Avg Cleaning Time", "N/A"), unsafe_allow_html=True)
    
    with col4:
        if not predictions_data.empty and 'co2_equivalent_saved' in predictions_data.columns:
            total_co2_saved = predictions_data['co2_equivalent_saved'].sum()
            st.markdown(create_kpi_card("Total CO₂ Saved", f"{total_co2_saved:.1f}", " g/km"), unsafe_allow_html=True)
        else:
            st.markdown(create_kpi_card("Total CO₂ Saved", "N/A"), unsafe_allow_html=True)
    
    # Analytics charts
    if not predictions_data.empty:
        st.markdown("## 📈 Prediction Analytics")
        
        # Predictions over time
        if 'created_at' in predictions_data.columns:
            predictions_data['created_at'] = pd.to_datetime(predictions_data['created_at'])
            daily_predictions = predictions_data.groupby(predictions_data['created_at'].dt.date).size().reset_index()
            daily_predictions.columns = ['date', 'count']
            
            fig = px.line(
                daily_predictions, 
                x='date', 
                y='count',
                title='Predictions Over Time',
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle model distribution
        if 'vehicle_model' in predictions_data.columns:
            model_counts = predictions_data['vehicle_model'].value_counts().head(10)
            
            fig = px.bar(
                x=model_counts.values,
                y=model_counts.index,
                orientation='h',
                title='Top 10 Vehicle Models Analyzed',
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Emissions reduction distribution
        reduction_cols = ['predicted_hc_reduction', 'predicted_co_reduction', 
                         'predicted_nox_reduction', 'predicted_particulates_reduction']
        
        available_cols = [col for col in reduction_cols if col in predictions_data.columns]
        
        if available_cols:
            fig = go.Figure()
            
            for col in available_cols:
                fig.add_trace(go.Box(
                    y=predictions_data[col],
                    name=col.replace('predicted_', '').replace('_reduction', '').upper(),
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title='Emissions Reduction Distribution',
                yaxis_title='Reduction Percentage (%)',
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📝 No prediction data available yet. Make some predictions to see analytics!")
    
    # Training data analytics
    if not training_data.empty:
        st.markdown("## 🔬 Training Data Analytics")
        
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(training_data[numeric_cols].describe(), use_container_width=True)
            
            # Distribution plots
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col1 = st.selectbox("Select variable for distribution", numeric_cols, key="dist1")
                
                with col2:
                    selected_col2 = st.selectbox("Select variable for comparison", numeric_cols, key="dist2")
                
                if selected_col1 and selected_col2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            training_data, 
                            x=selected_col1,
                            title=f'Distribution of {selected_col1}',
                            template='plotly_dark'
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.scatter(
                            training_data, 
                            x=selected_col1, 
                            y=selected_col2,
                            title=f'{selected_col1} vs {selected_col2}',
                            template='plotly_dark'
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Export all data
    st.markdown("## 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not training_data.empty:
            csv_training = training_data.to_csv(index=False)
            st.download_button(
                label="📊 Export Training Data (CSV)",
                data=csv_training,
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not predictions_data.empty:
            csv_predictions = predictions_data.to_csv(index=False)
            st.download_button(
                label="🔮 Export Predictions (CSV)",
                data=csv_predictions,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

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
