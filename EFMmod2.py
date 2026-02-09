"""
================================================================================
STREAMLIT APP: HYBRID PHYSICS-ML VELOCITY PREDICTION FOR CARBONATE ROCKS
================================================================================
Complete application with Plotly visualizations and data table download
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning components
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.special import iv

# Set page configuration
st.set_page_config(
    page_title="Carbonate Velocity Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #374151;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .data-table {
        font-size: 0.9rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .download-button {
        background-color: #3B82F6;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
    }
    .download-button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FIXED PHYSICS-BASED MODEL
# ==============================================================================

class EffectiveFieldMethod:
    """
    Fixed implementation of the effective field method
    """
    
    def __init__(self, matrix_props, crack_props):
        self.matrix = matrix_props
        self.crack = crack_props
        
        # Calculate moduli
        self._calculate_moduli()
    
    def _calculate_moduli(self):
        """Calculate elastic moduli with unit consistency"""
        mat = self.matrix
        
        # Convert density from g/cc to kg/m³ for calculations
        if 'rho' in mat:
            mat['rho_kgm3'] = mat['rho'] * 1000
        else:
            mat['rho_kgm3'] = 2650  # Default limestone density
        
        # Calculate moduli in Pa
        if 'Vp' in mat and 'Vs' in mat:
            # G = ρ * Vs² (in Pa)
            mat['G'] = mat['rho_kgm3'] * mat['Vs']**2
            
            # K = ρ * (Vp² - 4/3 * Vs²)
            mat['K'] = mat['rho_kgm3'] * (mat['Vp']**2 - (4/3) * mat['Vs']**2)
            
            # Poisson's ratio
            mat['nu'] = (mat['Vp']**2 - 2*mat['Vs']**2) / (2*(mat['Vp']**2 - mat['Vs']**2))
    
    def orientation_distribution_function(self, beta, distribution='uniform'):
        """Calculate F(β) with safe numerical handling"""
        beta = float(beta)
        
        if distribution == 'uniform':
            if beta == 0 or np.isnan(beta):
                return 1.0
            return (beta + np.sin(beta) * np.cos(beta)) / (2 * beta)
        
        elif distribution == 'von_mises':
            if beta == 0 or np.isnan(beta):
                return 1.0, 1.0
            
            sigma = max(beta, 0.001)  # Avoid zero
            z = 1.0 / (sigma ** 2)
            
            try:
                I0 = iv(0, z)
                I1 = iv(1, z)
                I2 = iv(2, z)
                
                F1 = (sigma**2 * I1 + I2) / I0
                F2 = (sigma**2 * I1) / I0
                return F1, F2
            except:
                return 1.0, 1.0
        
        return 1.0
    
    def estimate_crack_parameters(self, porosity, sw=1.0, rt=1.0, vclay=0):
        """
        Estimate crack parameters from well logs with safe handling
        """
        # Ensure valid inputs
        porosity = max(float(porosity), 0.001)
        sw = min(max(float(sw), 0), 1.0)
        rt = max(float(rt), 0.1)
        vclay = min(max(float(vclay), 0), 100)
        
        # Estimate crack density
        aspect_ratio = 0.01  # Typical for microcracks
        crack_density = (3 * porosity * sw) / (4 * np.pi * aspect_ratio)
        
        # Adjust for clay content
        clay_factor = 1.0 - min(vclay / 100.0, 0.7)
        crack_density *= clay_factor
        
        # Estimate orientation from resistivity
        rt_factor = np.log10(max(rt, 0.1)) / 3.0
        beta = np.pi/4 * (1 - 0.3 * rt_factor)  # 45° for isotropic
        
        # Bound values
        crack_density = min(max(crack_density, 0), 0.5)
        beta = min(max(beta, 0), np.pi/2)
        
        return crack_density, beta
    
    def calculate_effective_properties(self, crack_density, beta):
        """
        Calculate effective properties with simplified model
        """
        # Matrix properties
        K0 = self.matrix.get('K', 50e9)  # Default 50 GPa
        G0 = self.matrix.get('G', 30e9)  # Default 30 GPa
        rho0 = self.matrix.get('rho_kgm3', 2650)
        
        # Get orientation factor
        F_val = self.orientation_distribution_function(beta, 'uniform')
        
        # Simplified effective moduli calculation
        # Based on crack density and orientation
        K_eff = K0 * (1 - crack_density * 0.8 * F_val)
        G_eff = G0 * (1 - crack_density * 0.6 * F_val)
        
        # Calculate velocities
        Vp_eff = np.sqrt((K_eff + 4*G_eff/3) / rho0)
        Vs_eff = np.sqrt(G_eff / rho0)
        
        return {
            'Vp': Vp_eff,
            'Vs': Vs_eff,
            'K': K_eff,
            'G': G_eff,
            'crack_density': crack_density,
            'beta': beta,
            'F': F_val
        }

# ==============================================================================
# FIXED HYBRID MODEL WITH PROPER NaN HANDLING
# ==============================================================================

class HybridVelocityPredictor:
    """
    Fixed hybrid model with proper data handling
    """
    
    def __init__(self, matrix_props=None):
        self.efm_model = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.matrix_props = matrix_props
        self.feature_importances = {}
        
    def create_safe_features(self, df):
        """
        Create features with safe numerical handling
        """
        features = {}
        
        # Basic features with NaN handling
        basic_cols = ['porosity', 'rho', 'sw']
        for col in basic_cols:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
        
        # Additional features if available
        if 'Vclay' in df.columns:
            features['Vclay'] = pd.to_numeric(df['Vclay'], errors='coerce').fillna(0)
        
        if 'RT' in df.columns:
            features['RT'] = pd.to_numeric(df['RT'], errors='coerce').fillna(df['RT'].median())
            features['RT_log'] = np.log10(np.maximum(features['RT'], 0.1))
        
        if 'GR' in df.columns:
            features['GR'] = pd.to_numeric(df['GR'], errors='coerce').fillna(df['GR'].median())
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Add physics-based features if EFM model is available
        if self.efm_model and self.matrix_props:
            crack_densities = []
            betas = []
            F_values = []
            vp_efm = []
            vs_efm = []
            
            for idx in range(len(df)):
                # Get values with safe access
                porosity_val = features_df.iloc[idx]['porosity'] if 'porosity' in features_df.columns else 0.1
                sw_val = features_df.iloc[idx]['sw'] if 'sw' in features_df.columns else 1.0
                rt_val = features_df.iloc[idx]['RT'] if 'RT' in features_df.columns else 1.0
                vclay_val = features_df.iloc[idx]['Vclay'] if 'Vclay' in features_df.columns else 0
                
                # Estimate crack parameters
                crack_density, beta = self.efm_model.estimate_crack_parameters(
                    porosity_val, sw_val, rt_val, vclay_val
                )
                
                # Calculate effective properties
                eff_props = self.efm_model.calculate_effective_properties(crack_density, beta)
                
                crack_densities.append(crack_density)
                betas.append(beta)
                F_values.append(eff_props['F'])
                vp_efm.append(eff_props['Vp'])
                vs_efm.append(eff_props['Vs'])
            
            # Add to features
            features_df['crack_density_efm'] = crack_densities
            features_df['orientation_beta'] = betas
            features_df['F_beta'] = F_values
            features_df['Vp_efm'] = vp_efm
            features_df['Vs_efm'] = vs_efm
        
        # Feature engineering with safe operations
        if 'porosity' in features_df.columns:
            features_df['porosity_sq'] = features_df['porosity'] ** 2
            features_df['porosity_sqrt'] = np.sqrt(np.maximum(features_df['porosity'], 0))
        
        if 'rho' in features_df.columns:
            features_df['density_porosity'] = features_df['rho'] * features_df['porosity']
        
        # Interaction features
        if 'crack_density_efm' in features_df.columns:
            if 'porosity' in features_df.columns:
                features_df['crack_porosity'] = features_df['crack_density_efm'] * features_df['porosity']
            
            if 'Vclay' in features_df.columns:
                features_df['crack_clay'] = features_df['crack_density_efm'] * features_df['Vclay']
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(features_df.median())
        
        # Replace infinities
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
        
        return features_df
    
    def train(self, X_train, y_train_vp, y_train_vs):
        """
        Train hybrid model with proper data preprocessing
        """
        # Impute missing values
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Train Vp model
        self.ml_models['Vp'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        
        # Train Vs model
        self.ml_models['Vs'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        
        # Train models
        self.ml_models['Vp'].fit(X_train_scaled, y_train_vp)
        self.ml_models['Vs'].fit(X_train_scaled, y_train_vs)
        
        # Store feature importances
        self.feature_importances['Vp'] = self.ml_models['Vp'].feature_importances_
        self.feature_importances['Vs'] = self.ml_models['Vs'].feature_importances_
    
    def predict(self, X):
        """
        Make predictions with proper preprocessing
        """
        # Impute and scale
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make predictions
        vp_pred = self.ml_models['Vp'].predict(X_scaled)
        vs_pred = self.ml_models['Vs'].predict(X_scaled)
        
        return vp_pred, vs_pred
    
    def hybrid_predict(self, X, physics_weight=0.3):
        """
        Combine ML predictions with physics-based predictions
        """
        # ML predictions
        vp_ml, vs_ml = self.predict(X)
        
        # Physics-based predictions if available
        if 'Vp_efm' in X.columns and 'Vs_efm' in X.columns:
            vp_physics = X['Vp_efm'].values
            vs_physics = X['Vs_efm'].values
            
            # Weighted combination
            vp_hybrid = (1 - physics_weight) * vp_ml + physics_weight * vp_physics
            vs_hybrid = (1 - physics_weight) * vs_ml + physics_weight * vs_physics
            
            return vp_hybrid, vs_hybrid
        
        return vp_ml, vs_ml

# ==============================================================================
# PLOTLY VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plotly_scatter(x_actual, y_predicted, title, x_label, y_label, color='blue'):
    """Create Plotly scatter plot with R² and perfect fit line"""
    
    # Calculate R² and correlation
    r2 = r2_score(x_actual, y_predicted)
    correlation = np.corrcoef(x_actual, y_predicted)[0, 1]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter trace
    fig.add_trace(go.Scatter(
        x=x_actual,
        y=y_predicted,
        mode='markers',
        marker=dict(
            color=color,
            size=8,
            opacity=0.6,
            line=dict(width=0.5, color='white')
        ),
        name='Predictions',
        text=[f'Actual: {a:.0f}<br>Predicted: {p:.0f}<br>Error: {p-a:.0f}' 
              for a, p in zip(x_actual, y_predicted)],
        hoverinfo='text'
    ))
    
    # Add perfect fit line
    min_val = min(min(x_actual), min(y_predicted))
    max_val = max(max(x_actual), max(y_predicted))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Perfect Fit'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{title}<br>R² = {r2:.4f}, Correlation = {correlation:.4f}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig, r2, correlation

def create_plotly_histogram(errors, title, color='blue'):
    """Create Plotly histogram of errors"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        marker_color=color,
        opacity=0.7,
        name='Error Distribution',
        histnorm='probability density'
    ))
    
    # Add vertical lines
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", 
                  annotation_text="Zero Error", annotation_position="top right")
    fig.add_vline(x=mean_error, line_width=2, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean_error:.1f}%", annotation_position="top left")
    
    # Add normal distribution curve
    if len(errors) > 10:
        x_norm = np.linspace(errors.min(), errors.max(), 100)
        y_norm = (1/(std_error * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mean_error)/std_error)**2)
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            line=dict(color='black', width=2),
            name='Normal Distribution'
        ))
    
    fig.update_layout(
        title=f"{title}<br>Mean: {mean_error:.1f}%, Std: {std_error:.1f}%",
        xaxis_title='Error (%)',
        yaxis_title='Density',
        template='plotly_white',
        height=400,
        showlegend=True,
        bargap=0.1
    )
    
    return fig, mean_error, std_error

def create_plotly_feature_importance(feature_names, importances, title, color='steelblue'):
    """Create Plotly horizontal bar plot for feature importance"""
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[-10:]  # Top 10 features
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = [importances[i] for i in sorted_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features,
        x=top_importances,
        orientation='h',
        marker_color=color,
        text=[f'{imp:.4f}' for imp in top_importances],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

def create_prediction_table(df, results, include_features=True):
    """Create a comprehensive prediction table with actual vs predicted values"""
    
    # Create results table
    results_table = pd.DataFrame({
        'Actual_Vp_m_s': results['y_vp_test'],
        'Predicted_Vp_m_s': results['vp_test_pred'],
        'Vp_Error_%': 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test'],
        'Vp_Absolute_Error_m_s': results['vp_test_pred'] - results['y_vp_test'],
        'Actual_Vs_m_s': results['y_vs_test'],
        'Predicted_Vs_m_s': results['vs_test_pred'],
        'Vs_Error_%': 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test'],
        'Vs_Absolute_Error_m_s': results['vs_test_pred'] - results['y_vs_test'],
        'Actual_Vp_Vs_Ratio': results['y_vp_test'] / results['y_vs_test'],
        'Predicted_Vp_Vs_Ratio': results['vp_test_pred'] / results['vs_test_pred'],
        'Vp_Vs_Error_%': 100 * ((results['vp_test_pred'] / results['vs_test_pred']) - 
                               (results['y_vp_test'] / results['y_vs_test'])) / 
                               (results['y_vp_test'] / results['y_vs_test'])
    })
    
    # Add original features if requested
    if include_features:
        original_df = df
        test_indices = results['X_test'].index
        
        # Add commonly used features
        feature_cols = ['porosity', 'rho', 'Vclay', 'RT', 'sw', 'GR']
        for col in feature_cols:
            if col in original_df.columns:
                results_table[col] = original_df.loc[test_indices, col].values
    
    # Sort by absolute Vp error
    results_table['Abs_Vp_Error_%'] = np.abs(results_table['Vp_Error_%'])
    results_table = results_table.sort_values('Abs_Vp_Error_%')
    
    return results_table

def create_interactive_data_table(df, results):
    """Create an interactive data table with sorting and filtering"""
    
    # Create prediction table
    prediction_table = create_prediction_table(df, results, include_features=True)
    
    # Create a styled dataframe
    styled_df = prediction_table.style.format({
        'Actual_Vp_m_s': '{:.0f}',
        'Predicted_Vp_m_s': '{:.0f}',
        'Vp_Error_%': '{:.2f}',
        'Vp_Absolute_Error_m_s': '{:.0f}',
        'Actual_Vs_m_s': '{:.0f}',
        'Predicted_Vs_m_s': '{:.0f}',
        'Vs_Error_%': '{:.2f}',
        'Vs_Absolute_Error_m_s': '{:.0f}',
        'Actual_Vp_Vs_Ratio': '{:.3f}',
        'Predicted_Vp_Vs_Ratio': '{:.3f}',
        'Vp_Vs_Error_%': '{:.2f}',
        'porosity': '{:.3f}',
        'rho': '{:.3f}',
        'Vclay': '{:.1f}',
        'RT': '{:.1f}',
        'sw': '{:.2f}',
        'GR': '{:.1f}'
    })
    
    return styled_df, prediction_table

# ==============================================================================
# STREAMLIT APP MAIN FUNCTION
# ==============================================================================

def streamlit_app():
    """Streamlit application main function"""
    
    st.markdown('<h1 class="main-header">🎯 Hybrid Physics-ML Velocity Prediction for Carbonate Rocks</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        app_mode = st.radio(
            "Select Mode:",
            ["🏠 Home", "📁 Data Upload", "⚙️ Model Configuration", "🚀 Run Analysis", "📈 Results", "📊 Data Table", "🔮 Single Prediction"]
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        # Global settings
        physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.3, 0.1,
                                 help="Weight for physics prediction (0=pure ML, 1=pure physics)")
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5,
                            help="Percentage of data for testing")
        
        st.markdown("---")
        st.markdown("## 📚 About")
        st.info("""
        This app predicts acoustic velocities (Vp, Vs) in carbonate rocks using:
        
        - **Physics**: Effective Field Method (EFM)
        - **Machine Learning**: Gradient Boosting
        - **Hybrid**: Weighted combination
        
        Target: Correlation > 0.75
        """)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'hybrid_model' not in st.session_state:
        st.session_state.hybrid_model = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'prediction_table' not in st.session_state:
        st.session_state.prediction_table = None
    
    # ==========================================================================
    # HOME PAGE
    # ==========================================================================
    if app_mode == "🏠 Home":
        st.markdown("## Welcome to Carbonate Velocity Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 About This App")
            st.markdown("""
            This application predicts acoustic velocities (Vp, Vs) in carbonate rocks using:
            
            1. **Physics Model**: Effective Field Method based on micromechanics
            2. **Machine Learning**: Gradient Boosting for complex relationships
            3. **Hybrid Integration**: Weighted combination for robust predictions
            
            ### 🎯 Key Features
            - Interactive Plotly visualizations
            - R² and correlation metrics
            - **Complete prediction tables** with actual vs predicted values
            - Download results as CSV
            - Feature importance analysis
            """)
        
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. **Upload** your CSV data
            2. **Configure** model parameters
            3. **Run** the analysis
            4. **Explore** interactive results
            5. **View & Download** prediction tables
            
            ### 📋 Data Requirements
            **Required:**
            - Vp (P-wave velocity in m/s)
            - Vs (S-wave velocity in m/s)
            - Porosity (fraction)
            
            **Optional:**
            - Density (g/cc)
            - Clay content
            - Resistivity
            - Water saturation
            """)
        
        # Sample data
        st.markdown("### 📋 Sample Data Format")
        sample_data = pd.DataFrame({
            'Vp': [4500, 4400, 4300, 4200, 4100],
            'Vs': [2500, 2450, 2400, 2350, 2300],
            'porosity': [0.05, 0.08, 0.12, 0.15, 0.18],
            'rho': [2.68, 2.65, 2.62, 2.60, 2.57],
            'Vclay': [5, 8, 12, 15, 20],
            'RT': [100, 80, 60, 50, 40],
            'sw': [1.0, 1.0, 0.9, 0.8, 0.7]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        # Download template
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv,
            file_name="carbonate_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ==========================================================================
    # DATA UPLOAD PAGE
    # ==========================================================================
    elif app_mode == "📁 Data Upload":
        st.markdown("## 📁 Upload Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Display metrics
                st.markdown("### 📊 Data Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Samples", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    missing = df.isnull().sum().sum()
                    st.metric("Missing Values", missing)
                with col4:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # Data preview
                st.markdown("### 👁️ Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check required columns
                required_cols = ['Vp', 'Vs', 'porosity']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.success("✅ All required columns present")
                    
                    # Show statistics for key columns
                    st.markdown("### 📏 Data Statistics")
                    
                    if 'Vp' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Vp Range", f"{df['Vp'].min():.0f} - {df['Vp'].max():.0f} m/s")
                        with col2:
                            st.metric("Vp Mean", f"{df['Vp'].mean():.0f} m/s")
                    
                    if 'porosity' in df.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Porosity Range", f"{df['porosity'].min():.3f} - {df['porosity'].max():.3f}")
                        with col2:
                            st.metric("Porosity Mean", f"{df['porosity'].mean():.3f}")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # ==========================================================================
    # MODEL CONFIGURATION PAGE
    # ==========================================================================
    elif app_mode == "⚙️ Model Configuration":
        st.markdown("## ⚙️ Model Configuration")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first.")
            return
        
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Target Configuration")
            
            # Select target columns
            vp_col = st.selectbox("Vp column", df.columns,
                                 index=list(df.columns).index('Vp') if 'Vp' in df.columns else 0)
            vs_col = st.selectbox("Vs column", df.columns,
                                 index=list(df.columns).index('Vs') if 'Vs' in df.columns else 1)
            
            st.markdown("### 🔧 Feature Selection")
            
            # Available features (exclude targets)
            exclude_cols = [vp_col, vs_col, 'DEPTH', 'Depth', 'depth']
            available_cols = [col for col in df.columns if col not in exclude_cols]
            
            selected_features = st.multiselect(
                "Select features for modeling",
                available_cols,
                default=[col for col in ['porosity', 'rho', 'Vclay', 'RT', 'sw'] 
                        if col in available_cols]
            )
            
            st.session_state.selected_features = selected_features
        
        with col2:
            st.markdown("### ⚛️ Physics Model Settings")
            
            # Matrix properties
            st.markdown("#### Matrix Properties")
            
            # Estimate from data or manual input
            if 'porosity' in df.columns:
                low_porosity = df.nsmallest(10, 'porosity')
                estimated_vp = low_porosity[vp_col].mean() if len(low_porosity) > 0 else 5500
                estimated_vs = low_porosity[vs_col].mean() if len(low_porosity) > 0 else 3000
                estimated_rho = low_porosity['rho'].mean() if 'rho' in low_porosity.columns else 2.71
            else:
                estimated_vp = 5500
                estimated_vs = 3000
                estimated_rho = 2.71
            
            matrix_vp = st.number_input("Matrix Vp (m/s)", 3000, 7000, int(estimated_vp), 100)
            matrix_vs = st.number_input("Matrix Vs (m/s)", 1500, 4000, int(estimated_vs), 100)
            matrix_rho = st.number_input("Matrix ρ (g/cc)", 2.0, 3.0, float(estimated_rho), 0.01)
            
            matrix_props = {
                'Vp': matrix_vp,
                'Vs': matrix_vs,
                'rho': matrix_rho
            }
            
            # Crack properties
            st.markdown("#### Crack Properties")
            
            aspect_ratio = st.slider("Aspect Ratio", 0.001, 0.1, 0.01, 0.001)
            fluid_k = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 10.0, 2.25, 0.1) * 1e9
            
            crack_props = {
                'aspect_ratio': aspect_ratio,
                'fluid_K': fluid_k,
                'fluid_rho': 1000
            }
            
            st.session_state.matrix_props = matrix_props
            st.session_state.crack_props = crack_props
        
        # Save configuration
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("✅ Configuration saved!")
    
    # ==========================================================================
    # RUN ANALYSIS PAGE
    # ==========================================================================
    elif app_mode == "🚀 Run Analysis":
        st.markdown("## 🚀 Run Analysis")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first.")
            return
        
        if 'selected_features' not in st.session_state:
            st.warning("⚠️ Please configure the model first.")
            return
        
        df = st.session_state.df
        selected_features = st.session_state.selected_features
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize models
        status_text.text("Step 1/6: Initializing models...")
        progress_bar.progress(10)
        
        matrix_props = st.session_state.get('matrix_props', {
            'Vp': 5500, 'Vs': 3000, 'rho': 2.71
        })
        
        crack_props = st.session_state.get('crack_props', {
            'aspect_ratio': 0.01,
            'fluid_K': 2.25e9,
            'fluid_rho': 1000
        })
        
        # Initialize physics model
        efm_model = EffectiveFieldMethod(matrix_props, crack_props)
        
        # Initialize hybrid model
        hybrid_model = HybridVelocityPredictor(matrix_props)
        hybrid_model.efm_model = efm_model
        
        # Step 2: Create features
        status_text.text("Step 2/6: Creating features...")
        progress_bar.progress(30)
        
        df_features = hybrid_model.create_safe_features(df[selected_features])
        
        # Select feature columns
        exclude_cols = ['Vp', 'Vs']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df_features[col])]
        
        X = df_features[feature_cols]
        y_vp = df['Vp'].values
        y_vs = df['Vs'].values
        
        # Step 3: Split data
        status_text.text("Step 3/6: Splitting data...")
        progress_bar.progress(50)
        
        X_train, X_test, y_vp_train, y_vp_test, y_vs_train, y_vs_test = train_test_split(
            X, y_vp, y_vs, test_size=test_size/100, random_state=42
        )
        
        # Step 4: Train model
        status_text.text("Step 4/6: Training model...")
        progress_bar.progress(70)
        
        hybrid_model.train(X_train, y_vp_train, y_vs_train)
        
        # Step 5: Make predictions
        status_text.text("Step 5/6: Making predictions...")
        progress_bar.progress(90)
        
        # Test predictions
        vp_test_pred, vs_test_pred = hybrid_model.hybrid_predict(X_test, physics_weight)
        
        # Full predictions
        vp_full_pred, vs_full_pred = hybrid_model.hybrid_predict(X, physics_weight)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        st.session_state.results = {
            'X': X,
            'y_vp': y_vp,
            'y_vs': y_vs,
            'X_test': X_test,
            'y_vp_test': y_vp_test,
            'y_vs_test': y_vs_test,
            'vp_test_pred': vp_test_pred,
            'vs_test_pred': vs_test_pred,
            'vp_full_pred': vp_full_pred,
            'vs_full_pred': vs_full_pred,
            'feature_cols': feature_cols,
            'model': hybrid_model
        }
        
        # Create prediction table
        prediction_table = create_prediction_table(df, st.session_state.results, include_features=True)
        st.session_state.prediction_table = prediction_table
        
        st.session_state.hybrid_model = hybrid_model
        st.session_state.analysis_complete = True
        
        # Display metrics
        st.markdown("---")
        st.markdown("### 📊 Performance Metrics")
        
        # Calculate metrics
        vp_r2 = r2_score(y_vp_test, vp_test_pred)
        vs_r2 = r2_score(y_vs_test, vs_test_pred)
        vp_corr = np.corrcoef(y_vp_test, vp_test_pred)[0, 1]
        vs_corr = np.corrcoef(y_vs_test, vs_test_pred)[0, 1]
        vp_mae = mean_absolute_error(y_vp_test, vp_test_pred)
        vs_mae = mean_absolute_error(y_vs_test, vs_test_pred)
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Vp R²", f"{vp_r2:.4f}")
            st.metric("Vp Correlation", f"{vp_corr:.4f}")
        
        with col2:
            st.metric("Vs R²", f"{vs_r2:.4f}")
            st.metric("Vs Correlation", f"{vs_corr:.4f}")
        
        with col3:
            st.metric("Vp MAE", f"{vp_mae:.0f} m/s")
            st.metric("Vp RMSE", f"{np.sqrt(mean_squared_error(y_vp_test, vp_test_pred)):.0f} m/s")
        
        with col4:
            st.metric("Vs MAE", f"{vs_mae:.0f} m/s")
            st.metric("Vs RMSE", f"{np.sqrt(mean_squared_error(y_vs_test, vs_test_pred)):.0f} m/s")
        
        # Target achievement
        st.markdown("---")
        st.markdown("### 🎯 Target Achievement")
        
        target = 0.75
        if vp_corr >= target and vs_corr >= target:
            st.markdown('<div class="success-box">🎉 SUCCESS: Both Vp and Vs correlations exceed 0.75!</div>', 
                       unsafe_allow_html=True)
        elif vp_corr >= target:
            st.markdown(f'<div class="warning-box">⚠️ PARTIAL: Vp achieved target ({vp_corr:.4f}) but Vs needs improvement ({vs_corr:.4f})</div>', 
                       unsafe_allow_html=True)
        elif vs_corr >= target:
            st.markdown(f'<div class="warning-box">⚠️ PARTIAL: Vs achieved target ({vs_corr:.4f}) but Vp needs improvement ({vp_corr:.4f})</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ NEEDS IMPROVEMENT: Both Vp ({vp_corr:.4f}) and Vs ({vs_corr:.4f}) below target 0.75</div>', 
                       unsafe_allow_html=True)
    
    # ==========================================================================
    # RESULTS PAGE
    # ==========================================================================
    elif app_mode == "📈 Results":
        st.markdown("## 📈 Results & Visualizations")
        
        if st.session_state.results is None:
            st.warning("⚠️ Please run the analysis first")
            return
        
        results = st.session_state.results
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Error Analysis", "🔍 Feature Importance"])
        
        with tab1:
            st.markdown("### 📊 Prediction Results")
            
            # Vp plot
            fig_vp, r2_vp, corr_vp = create_plotly_scatter(
                results['y_vp_test'], results['vp_test_pred'],
                "Vp Prediction",
                "Measured Vp (m/s)",
                "Predicted Vp (m/s)",
                'blue'
            )
            
            st.plotly_chart(fig_vp, use_container_width=True)
            
            # Vs plot
            fig_vs, r2_vs, corr_vs = create_plotly_scatter(
                results['y_vs_test'], results['vs_test_pred'],
                "Vs Prediction",
                "Measured Vs (m/s)",
                "Predicted Vs (m/s)",
                'green'
            )
            
            st.plotly_chart(fig_vs, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Error Analysis")
            
            # Calculate errors
            vp_error = 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test']
            vs_error = 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_vp_error, vp_mean_error, vp_std_error = create_plotly_histogram(
                    vp_error, "Vp Error Distribution", 'blue'
                )
                st.plotly_chart(fig_vp_error, use_container_width=True)
            
            with col2:
                fig_vs_error, vs_mean_error, vs_std_error = create_plotly_histogram(
                    vs_error, "Vs Error Distribution", 'green'
                )
                st.plotly_chart(fig_vs_error, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Feature Importance")
            
            hybrid_model = results['model']
            if hasattr(hybrid_model, 'feature_importances'):
                # Vp feature importance
                fig_vp_imp = create_plotly_feature_importance(
                    results['feature_cols'],
                    hybrid_model.feature_importances['Vp'],
                    "Top Features for Vp Prediction",
                    'blue'
                )
                
                st.plotly_chart(fig_vp_imp, use_container_width=True)
                
                # Vs feature importance
                fig_vs_imp = create_plotly_feature_importance(
                    results['feature_cols'],
                    hybrid_model.feature_importances['Vs'],
                    "Top Features for Vs Prediction",
                    'green'
                )
                
                st.plotly_chart(fig_vs_imp, use_container_width=True)
    
    # ==========================================================================
    # DATA TABLE PAGE
    # ==========================================================================
    elif app_mode == "📊 Data Table":
        st.markdown("## 📊 Prediction Data Table")
        
        if st.session_state.results is None:
            st.warning("⚠️ Please run the analysis first")
            return
        
        if st.session_state.prediction_table is None:
            st.warning("⚠️ No prediction table available. Please run analysis first.")
            return
        
        results = st.session_state.results
        prediction_table = st.session_state.prediction_table
        
        st.markdown("### 📋 Complete Prediction Results")
        st.markdown("This table shows actual vs predicted Vp and Vs values for the test set.")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(prediction_table))
        with col2:
            vp_mean_error = prediction_table['Vp_Error_%'].mean()
            st.metric("Avg Vp Error", f"{vp_mean_error:.2f}%")
        with col3:
            vs_mean_error = prediction_table['Vs_Error_%'].mean()
            st.metric("Avg Vs Error", f"{vs_mean_error:.2f}%")
        with col4:
            best_samples = len(prediction_table[prediction_table['Abs_Vp_Error_%'] < 5])
            st.metric("Best Predictions (<5% error)", best_samples)
        
        # Create interactive table with sorting
        st.markdown("### 🔍 Interactive Data Table")
        st.markdown("Click on column headers to sort. Scroll horizontally to see all columns.")
        
        # Display the table with all columns
        st.dataframe(
            prediction_table,
            use_container_width=True,
            height=600
        )
        
        # Filter options
        st.markdown("### 🔎 Filter Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            error_threshold = st.slider("Maximum Vp Error (%)", 0, 50, 10, 1)
        with col2:
            show_best_only = st.checkbox("Show only best predictions (error < 5%)", value=False)
        
        # Filter the table
        if show_best_only:
            filtered_table = prediction_table[prediction_table['Abs_Vp_Error_%'] < 5]
        else:
            filtered_table = prediction_table[prediction_table['Abs_Vp_Error_%'] <= error_threshold]
        
        st.markdown(f"**Showing {len(filtered_table)} of {len(prediction_table)} samples**")
        
        # Display filtered table
        if len(filtered_table) > 0:
            st.dataframe(
                filtered_table,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No samples match the current filter criteria.")
        
        # Download section
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download complete results
            complete_csv = prediction_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Table",
                data=complete_csv,
                file_name="complete_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download filtered results
            filtered_csv = filtered_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Table",
                data=filtered_csv,
                file_name="filtered_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=len(filtered_table) == 0
            )
        
        with col3:
            # Download best predictions only
            best_predictions = prediction_table[prediction_table['Abs_Vp_Error_%'] < 5]
            best_csv = best_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Best Predictions",
                data=best_csv,
                file_name="best_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=len(best_predictions) == 0
            )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Summary Statistics")
        
        summary_stats = pd.DataFrame({
            'Statistic': [
                'Vp Mean Absolute Error (%)',
                'Vp Root Mean Square Error (m/s)',
                'Vs Mean Absolute Error (%)',
                'Vs Root Mean Square Error (m/s)',
                'Vp/Vs Ratio Mean Error (%)',
                'Best Predictions Count (<5% error)',
                'Good Predictions Count (<10% error)',
                'Fair Predictions Count (<20% error)'
            ],
            'Value': [
                f"{np.abs(prediction_table['Vp_Error_%']).mean():.2f}",
                f"{np.sqrt(np.mean(prediction_table['Vp_Absolute_Error_m_s']**2)):.0f}",
                f"{np.abs(prediction_table['Vs_Error_%']).mean():.2f}",
                f"{np.sqrt(np.mean(prediction_table['Vs_Absolute_Error_m_s']**2)):.0f}",
                f"{np.abs(prediction_table['Vp_Vs_Error_%']).mean():.2f}",
                f"{len(prediction_table[prediction_table['Abs_Vp_Error_%'] < 5])}",
                f"{len(prediction_table[prediction_table['Abs_Vp_Error_%'] < 10])}",
                f"{len(prediction_table[prediction_table['Abs_Vp_Error_%'] < 20])}"
            ]
        })
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Best and worst predictions
        st.markdown("### 🏆 Best & Worst Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Top 5 Best Predictions")
            best_5 = prediction_table.nsmallest(5, 'Abs_Vp_Error_%')
            display_cols = ['Actual_Vp_m_s', 'Predicted_Vp_m_s', 'Vp_Error_%', 
                          'Actual_Vs_m_s', 'Predicted_Vs_m_s', 'Vs_Error_%']
            st.dataframe(best_5[display_cols].round(2), use_container_width=True)
        
        with col2:
            st.markdown("#### ❌ Top 5 Worst Predictions")
            worst_5 = prediction_table.nlargest(5, 'Abs_Vp_Error_%')
            st.dataframe(worst_5[display_cols].round(2), use_container_width=True)
    
    # ==========================================================================
    # SINGLE PREDICTION PAGE
    # ==========================================================================
    elif app_mode == "🔮 Single Prediction":
        st.markdown("## 🔮 Single Sample Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Input Parameters")
            
            porosity = st.slider("Porosity", 0.0, 0.5, 0.15, 0.01)
            clay_content = st.slider("Clay Content (%)", 0.0, 100.0, 10.0, 1.0)
            saturation = st.slider("Water Saturation", 0.0, 1.0, 1.0, 0.1)
            density = st.number_input("Bulk Density (g/cc)", 2.0, 3.0, 2.65, 0.01)
            resistivity = st.number_input("Resistivity (ohm-m)", 0.1, 1000.0, 100.0, 10.0)
        
        with col2:
            st.markdown("### 🎯 Prediction")
            
            if st.button("Predict", use_container_width=True):
                # Get matrix properties
                matrix_props = st.session_state.get('matrix_props', {
                    'Vp': 5500, 'Vs': 3000, 'rho': 2.71
                })
                
                crack_props = st.session_state.get('crack_props', {
                    'aspect_ratio': 0.01,
                    'fluid_K': 2.25e9
                })
                
                # Initialize physics model
                efm_model = EffectiveFieldMethod(matrix_props, crack_props)
                
                # Make prediction
                crack_density, beta = efm_model.estimate_crack_parameters(
                    porosity, saturation, resistivity, clay_content
                )
                
                eff_props = efm_model.calculate_effective_properties(crack_density, beta)
                
                # Display results in a table
                st.markdown("#### 📊 Prediction Results")
                
                results_table = pd.DataFrame({
                    'Parameter': ['P-wave Velocity (Vp)', 'S-wave Velocity (Vs)', 
                                 'Vp/Vs Ratio', 'Crack Density', 'Orientation Angle (β)'],
                    'Value': [
                        f"{eff_props['Vp']:.0f} m/s",
                        f"{eff_props['Vs']:.0f} m/s",
                        f"{eff_props['Vp']/eff_props['Vs']:.2f}",
                        f"{crack_density:.3f}",
                        f"{beta:.3f} rad"
                    ],
                    'Units': ['m/s', 'm/s', '-', '-', 'rad']
                })
                
                st.dataframe(results_table, use_container_width=True)
                
                # Create comparison visualization
                st.markdown("#### 📈 Velocity Comparison")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Vp', 'Vs'],
                    y=[eff_props['Vp'], eff_props['Vs']],
                    marker_color=['blue', 'green'],
                    text=[f'{eff_props["Vp"]:.0f} m/s', f'{eff_props["Vs"]:.0f} m/s'],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Predicted Velocities',
                    xaxis_title='Velocity Type',
                    yaxis_title='Velocity (m/s)',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download single prediction
                single_result = pd.DataFrame({
                    'Porosity': [porosity],
                    'Clay_Content_%': [clay_content],
                    'Saturation': [saturation],
                    'Density_g_cc': [density],
                    'Resistivity_ohm_m': [resistivity],
                    'Predicted_Vp_m_s': [eff_props['Vp']],
                    'Predicted_Vs_m_s': [eff_props['Vs']],
                    'Predicted_Vp_Vs_Ratio': [eff_props['Vp']/eff_props['Vs']],
                    'Crack_Density': [crack_density],
                    'Orientation_Angle_rad': [beta]
                })
                
                csv = single_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download This Prediction",
                    data=csv,
                    file_name="single_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================================
# RUN THE APP
# ==============================================================================

if __name__ == "__main__":
    streamlit_app()
