"""
================================================================================
STREAMLIT APP: HYBRID PHYSICS-ML VELOCITY PREDICTION FOR CARBONATE ROCKS
================================================================================

Interactive web application for predicting acoustic velocities in carbonate
rocks using a hybrid physics-machine learning approach.

Author: Geophysics/ML Specialist
Version: 2.0
Date: 2024
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import iv
import warnings
warnings.filterwarnings('ignore')

# Machine Learning components
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SCIENTIFIC IMPLEMENTATION: EFFECTIVE FIELD METHOD (EFM)
# ==============================================================================

class EffectiveFieldMethod:
    """Physics-based model for cracked elastic media."""
    
    def __init__(self, matrix_props, crack_props):
        self.matrix = matrix_props.copy()
        self.crack = crack_props.copy()
        self._calculate_elastic_moduli()
    
    def _calculate_elastic_moduli(self):
        """Calculate elastic moduli from velocities."""
        mat = self.matrix
        
        # Convert density to kg/m³
        if 'rho' in mat:
            mat['rho_kgm3'] = mat['rho'] * 1000
        else:
            mat['rho_kgm3'] = 2650
        
        # Calculate moduli if velocities are provided
        if 'Vp' in mat and 'Vs' in mat:
            mat['G'] = mat['rho_kgm3'] * mat['Vs']**2
            mat['K'] = mat['rho_kgm3'] * (mat['Vp']**2 - (4/3) * mat['Vs']**2)
            vp2, vs2 = mat['Vp']**2, mat['Vs']**2
            mat['nu'] = (vp2 - 2*vs2) / (2*(vp2 - vs2))
    
    def orientation_distribution_function(self, beta, distribution='uniform'):
        """Calculate orientation distribution factor F(β)."""
        beta = float(beta)
        
        if distribution == 'uniform':
            if beta <= 1e-10 or np.isnan(beta):
                return 1.0
            return (beta + np.sin(beta) * np.cos(beta)) / (2 * beta)
        
        elif distribution == 'von_mises':
            if beta <= 1e-10 or np.isnan(beta):
                return 1.0, 1.0
            
            sigma = max(beta, 0.001)
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
        """Estimate crack parameters from well log data."""
        porosity = max(float(porosity), 0.001)
        sw = min(max(float(sw), 0.0), 1.0)
        rt = max(float(rt), 0.1)
        
        # Convert vclay to fraction
        if vclay > 1.0:
            vclay_frac = vclay / 100.0
        else:
            vclay_frac = vclay
        vclay_frac = min(max(vclay_frac, 0.0), 0.7)
        
        # Estimate crack density
        aspect_ratio = self.crack.get('aspect_ratio', 0.01)
        crack_density = (3.0 * porosity * sw) / (4.0 * np.pi * aspect_ratio)
        
        # Clay reduction factor
        clay_factor = 1.0 - vclay_frac * 1.5
        crack_density *= max(clay_factor, 0.3)
        
        # Estimate orientation from resistivity
        rt_factor = np.log10(max(rt, 0.1)) / 3.0
        rt_factor = min(max(rt_factor, 0.1), 1.0)
        
        beta_base = np.pi / 4
        beta = beta_base * (1.0 - 0.3 * rt_factor)
        beta *= (1.0 + 0.2 * vclay_frac)
        
        # Physical bounds
        crack_density = min(max(crack_density, 0.0), 0.5)
        beta = min(max(beta, 0.0), np.pi / 2)
        
        return crack_density, beta
    
    def calculate_effective_properties(self, crack_density, beta, distribution='uniform'):
        """Calculate effective elastic properties."""
        # Matrix properties
        K0 = self.matrix.get('K', 50e9)
        G0 = self.matrix.get('G', 30e9)
        rho0 = self.matrix.get('rho_kgm3', 2650)
        
        # Get orientation factor
        if distribution == 'uniform':
            F_val = self.orientation_distribution_function(beta, 'uniform')
            F1, F2 = F_val, F_val
        else:
            F1, F2 = self.orientation_distribution_function(beta, 'von_mises')
        
        # Crack compliance parameters
        aspect_ratio = self.crack.get('aspect_ratio', 0.01)
        fluid_K = self.crack.get('fluid_K', 2.25e9)
        
        C1 = 0.8
        C2 = 0.6
        
        if fluid_K > 0:
            fluid_factor = K0 / (K0 + fluid_K)
            C1 *= fluid_factor
        
        # Effective moduli
        K_eff = K0 * (1.0 - crack_density * C1 * F1)
        G_eff = G0 * (1.0 - crack_density * C2 * F2)
        
        K_eff = max(K_eff, 0.1 * K0)
        G_eff = max(G_eff, 0.1 * G0)
        
        # Calculate velocities
        Vp_eff = np.sqrt((K_eff + 4.0 * G_eff / 3.0) / rho0)
        Vs_eff = np.sqrt(G_eff / rho0)
        
        # Anisotropy
        epsilon = 0.0
        gamma = 0.0
        
        if distribution != 'uniform' and beta < np.pi/4:
            epsilon = crack_density * 0.1 * (1 - beta/(np.pi/4))
            gamma = crack_density * 0.15 * (1 - beta/(np.pi/4))
        
        return {
            'Vp': Vp_eff,
            'Vs': Vs_eff,
            'K': K_eff,
            'G': G_eff,
            'crack_density': crack_density,
            'beta': beta,
            'F_normal': F1,
            'F_shear': F2,
            'anisotropy_epsilon': epsilon,
            'anisotropy_gamma': gamma,
            'Vp/Vs': Vp_eff / Vs_eff if Vs_eff > 0 else 0
        }

# ==============================================================================
# HYBRID PHYSICS-MACHINE LEARNING MODEL
# ==============================================================================

class HybridVelocityPredictor:
    """Hybrid model combining physics-based EFM with machine learning."""
    
    def __init__(self, matrix_props=None):
        self.efm_model = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.matrix_props = matrix_props
        self.feature_importances = {}
        self.feature_names = []
    
    def create_safe_features(self, df):
        """Create feature matrix with robust handling."""
        features = {}
        
        # Basic features
        basic_cols = ['porosity', 'rho', 'sw']
        for col in basic_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                if series.isna().any():
                    series = series.fillna(series.median())
                features[col] = series
        
        # Additional features
        if 'Vclay' in df.columns:
            features['Vclay'] = pd.to_numeric(df['Vclay'], errors='coerce').fillna(0)
        
        if 'RT' in df.columns:
            features['RT'] = pd.to_numeric(df['RT'], errors='coerce')
            features['RT'] = features['RT'].fillna(features['RT'].median())
            features['RT_log'] = np.log10(np.maximum(features['RT'], 0.1))
        
        if 'GR' in df.columns:
            features['GR'] = pd.to_numeric(df['GR'], errors='coerce')
            features['GR'] = features['GR'].fillna(features['GR'].median())
        
        # Physics-based features if EFM available
        if self.efm_model and self.matrix_props:
            crack_densities = []
            betas = []
            F_values = []
            vp_efm = []
            vs_efm = []
            
            n_samples = len(df)
            
            for idx in range(n_samples):
                porosity_val = features.get('porosity', pd.Series([0.1]*n_samples)).iloc[idx] \
                              if 'porosity' in features else 0.1
                sw_val = features.get('sw', pd.Series([1.0]*n_samples)).iloc[idx] \
                        if 'sw' in features else 1.0
                rt_val = features.get('RT', pd.Series([1.0]*n_samples)).iloc[idx] \
                        if 'RT' in features else 1.0
                vclay_val = features.get('Vclay', pd.Series([0.0]*n_samples)).iloc[idx] \
                           if 'Vclay' in features else 0.0
                
                crack_density, beta = self.efm_model.estimate_crack_parameters(
                    porosity_val, sw_val, rt_val, vclay_val
                )
                
                eff_props = self.efm_model.calculate_effective_properties(
                    crack_density, beta, 'uniform'
                )
                
                crack_densities.append(crack_density)
                betas.append(beta)
                F_values.append(eff_props['F_normal'])
                vp_efm.append(eff_props['Vp'])
                vs_efm.append(eff_props['Vs'])
            
            features['crack_density_efm'] = crack_densities
            features['orientation_beta'] = betas
            features['F_beta'] = F_values
            features['Vp_efm'] = vp_efm
            features['Vs_efm'] = vs_efm
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Feature engineering
        if 'porosity' in features_df.columns:
            features_df['porosity_sq'] = features_df['porosity'] ** 2
            features_df['porosity_sqrt'] = np.sqrt(np.maximum(features_df['porosity'], 0))
        
        if 'rho' in features_df.columns and 'porosity' in features_df.columns:
            features_df['density_porosity'] = features_df['rho'] * features_df['porosity']
        
        if 'crack_density_efm' in features_df.columns:
            if 'porosity' in features_df.columns:
                features_df['crack_porosity'] = features_df['crack_density_efm'] * features_df['porosity']
            
            if 'Vclay' in features_df.columns:
                features_df['crack_clay'] = features_df['crack_density_efm'] * features_df['Vclay']
        
        # Clean data
        features_df = features_df.fillna(features_df.median())
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
        
        return features_df
    
    def train(self, X_train, y_train_vp, y_train_vs, model_type='Gradient Boosting'):
        """Train ML models."""
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Impute and scale
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Select model
        if model_type == 'Gradient Boosting':
            vp_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=2, random_state=42
            )
            vs_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=2, random_state=42
            )
        else:  # Random Forest
            vp_model = RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
            vs_model = RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
        
        # Train models
        vp_model.fit(X_train_scaled, y_train_vp)
        vs_model.fit(X_train_scaled, y_train_vs)
        
        self.ml_models['Vp'] = vp_model
        self.ml_models['Vs'] = vs_model
        
        # Store feature importances
        self.feature_importances['Vp'] = vp_model.feature_importances_
        self.feature_importances['Vs'] = vs_model.feature_importances_
    
    def predict(self, X):
        """Make predictions using trained ML models."""
        if not self.ml_models:
            raise ValueError("Models not trained.")
        
        # Reorder columns
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0
        
        X = X[self.feature_names]
        
        # Preprocess
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Predict
        vp_pred = self.ml_models['Vp'].predict(X_scaled)
        vs_pred = self.ml_models['Vs'].predict(X_scaled)
        
        return vp_pred, vs_pred
    
    def hybrid_predict(self, X, physics_weight=0.3):
        """Combine ML predictions with physics-based predictions."""
        # ML predictions
        vp_ml, vs_ml = self.predict(X)
        
        # Physics predictions if available
        if physics_weight > 0 and 'Vp_efm' in X.columns and 'Vs_efm' in X.columns:
            vp_physics = X['Vp_efm'].values
            vs_physics = X['Vs_efm'].values
            
            vp_hybrid = (1 - physics_weight) * vp_ml + physics_weight * vp_physics
            vs_hybrid = (1 - physics_weight) * vs_ml + physics_weight * vs_physics
            
            return vp_hybrid, vs_hybrid
        
        return vp_ml, vs_ml

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Hybrid Physics-ML Velocity Prediction for Carbonate Rocks</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        app_mode = st.radio(
            "Select Mode:",
            ["🏠 Home", "📁 Data Upload", "⚙️ Model Configuration", "🚀 Run Analysis", "📈 Results", "🔮 Single Prediction"]
        )
        
        st.markdown("---")
        st.markdown("## 📚 About")
        st.info("""
        This app predicts acoustic velocities (Vp, Vs) in carbonate rocks using:
        
        - **Physics**: Effective Field Method (EFM)
        - **Machine Learning**: Gradient Boosting/Random Forest
        - **Hybrid**: Weighted combination
        
        Target: Correlation > 0.75
        """)
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.3, 0.1,
                                 help="Weight for physics prediction (0=pure ML, 1=pure physics)")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5,
                            help="Percentage of data for testing")
        
        model_type = st.selectbox(
            "ML Model",
            ["Gradient Boosting", "Random Forest"],
            help="Select machine learning algorithm"
        )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'hybrid_model' not in st.session_state:
        st.session_state.hybrid_model = None
    
    # ==========================================================================
    # HOME PAGE
    # ==========================================================================
    if app_mode == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to Carbonate Velocity Predictor</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Data Requirements")
            st.markdown("""
            Required columns:
            - **Vp**: P-wave velocity (m/s)
            - **Vs**: S-wave velocity (m/s)
            - **porosity**: Fraction or percentage
            - **rho**: Density (g/cc)
            
            Optional columns:
            - sw, Vclay, RT, GR, DEPTH
            """)
        
        with col2:
            st.markdown("### 🔬 Scientific Foundation")
            st.markdown("""
            Based on:
            1. **Effective Field Method** (Hudson, 1980)
            2. **Crack density theory** (Budiansky & O'Connell, 1976)
            3. **Machine learning** (Gradient Boosting)
            4. **Hybrid integration** for robustness
            """)
        
        with col3:
            st.markdown("### 🎯 Performance Targets")
            st.markdown("""
            Success criteria:
            - **Vp correlation > 0.75**
            - **Vs correlation > 0.75**
            - **Mean error < 5%**
            - **Physical consistency**
            """)
        
        st.markdown("---")
        
        # Quick start guide
        st.markdown('<h3 class="section-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
        
        steps = st.columns(4)
        with steps[0]:
            st.markdown("### 1. Upload")
            st.markdown("Go to **Data Upload** and upload your CSV file")
        
        with steps[1]:
            st.markdown("### 2. Configure")
            st.markdown("Set model parameters in **Model Configuration**")
        
        with steps[2]:
            st.markdown("### 3. Run")
            st.markdown("Execute analysis in **Run Analysis**")
        
        with steps[3]:
            st.markdown("### 4. Results")
            st.markdown("View predictions in **Results**")
        
        st.markdown("---")
        
        # Example data
        st.markdown('<h3 class="section-header">📋 Sample Data Format</h3>', unsafe_allow_html=True)
        
        sample_data = pd.DataFrame({
            'DEPTH': [1000, 1001, 1002, 1003, 1004],
            'Vp': [4500, 4450, 4400, 4350, 4300],
            'Vs': [2500, 2480, 2450, 2420, 2400],
            'porosity': [0.05, 0.08, 0.10, 0.12, 0.15],
            'rho': [2.65, 2.63, 2.60, 2.58, 2.55],
            'sw': [1.0, 1.0, 0.9, 0.8, 0.7],
            'Vclay': [5, 8, 10, 12, 15],
            'RT': [100, 80, 60, 50, 40],
            'GR': [30, 35, 40, 45, 50]
        })
        
        st.dataframe(sample_data)
        
        # Download sample template
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv,
            file_name="carbonate_template.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # DATA UPLOAD PAGE
    # ==========================================================================
    elif app_mode == "📁 Data Upload":
        st.markdown('<h2 class="sub-header">📁 Data Upload & Exploration</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="Upload carbonate rock data with required columns"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                    
                    # Display basic info
                    st.markdown("### 📊 Data Overview")
                    
                    info_cols = st.columns(4)
                    with info_cols[0]:
                        st.metric("Samples", len(df))
                    with info_cols[1]:
                        st.metric("Features", len(df.columns))
                    with info_cols[2]:
                        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                    with info_cols[3]:
                        missing = df.isnull().sum().sum()
                        st.metric("Missing Values", missing)
                    
                    # Data preview
                    st.markdown("### 👁️ Data Preview")
                    st.dataframe(df.head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.markdown("### 🔍 Data Quality Check")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                
                # Check required columns
                required_cols = ['Vp', 'Vs', 'porosity', 'rho']
                missing_required = [col for col in required_cols if col not in df.columns]
                
                if missing_required:
                    st.error(f"❌ Missing required columns: {missing_required}")
                else:
                    st.success("✅ All required columns present")
                
                # Check data ranges
                st.markdown("#### 📏 Data Ranges")
                
                if 'Vp' in df.columns:
                    vp_min, vp_max = df['Vp'].min(), df['Vp'].max()
                    st.info(f"Vp: {vp_min:.0f} - {vp_max:.0f} m/s")
                
                if 'Vs' in df.columns:
                    vs_min, vs_max = df['Vs'].min(), df['Vs'].max()
                    st.info(f"Vs: {vs_min:.0f} - {vs_max:.0f} m/s")
                
                if 'porosity' in df.columns:
                    phi_min, phi_max = df['porosity'].min(), df['porosity'].max()
                    st.info(f"Porosity: {phi_min:.3f} - {phi_max:.3f}")
                
                # Column selector for detailed view
                st.markdown("#### 📋 Column Details")
                selected_col = st.selectbox("Select column", df.columns)
                if selected_col:
                    col_data = df[selected_col]
                    st.write(f"**Type:** {col_data.dtype}")
                    st.write(f"**Non-null:** {col_data.count()}/{len(col_data)}")
                    st.write(f"**Mean:** {col_data.mean():.4f}")
                    st.write(f"**Std:** {col_data.std():.4f}")
        
        # Data visualization
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown('<h3 class="section-header">📈 Data Visualization</h3>', unsafe_allow_html=True)
            
            df = st.session_state.df
            
            plot_type = st.selectbox("Select plot type", 
                                    ["Histogram", "Scatter Plot", "Box Plot", "Correlation Matrix"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", df.columns, index=min(0, len(df.columns)-1))
            
            with col2:
                if plot_type == "Scatter Plot":
                    y_axis = st.selectbox("Y-axis", df.columns, index=min(1, len(df.columns)-1))
            
            if plot_type == "Histogram":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[x_axis].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {x_axis}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif plot_type == "Scatter Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_axis], df[y_axis], alpha=0.5, s=20)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'{y_axis} vs {x_axis}')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(df) > 1:
                    mask = ~df[[x_axis, y_axis]].isna().any(axis=1)
                    if mask.sum() > 2:
                        x_vals = df.loc[mask, x_axis]
                        y_vals = df.loc[mask, y_axis]
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(x_range, p(x_range), 'r-', linewidth=2, alpha=0.8)
                
                st.pyplot(fig)
            
            elif plot_type == "Box Plot":
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.boxplot(df[x_axis].dropna())
                ax.set_ylabel(x_axis)
                ax.set_title(f'Box Plot of {x_axis}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif plot_type == "Correlation Matrix":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    ax.set_xticks(range(len(numeric_cols)))
                    ax.set_yticks(range(len(numeric_cols)))
                    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                    ax.set_yticklabels(numeric_cols)
                    
                    # Add correlation values
                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black")
                    
                    plt.colorbar(im)
                    ax.set_title('Correlation Matrix')
                    st.pyplot(fig)
    
    # ==========================================================================
    # MODEL CONFIGURATION PAGE
    # ==========================================================================
    elif app_mode == "⚙️ Model Configuration":
        st.markdown('<h2 class="sub-header">⚙️ Model Configuration</h2>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first in the Data Upload section.")
            return
        
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Target Configuration")
            
            # Target variable selection
            vp_col = st.selectbox("Vp column", df.columns, 
                                 index=list(df.columns).index('Vp') if 'Vp' in df.columns else 0)
            vs_col = st.selectbox("Vs column", df.columns,
                                 index=list(df.columns).index('Vs') if 'Vs' in df.columns else 1)
            
            # Feature selection
            st.markdown("### 🔧 Feature Selection")
            
            exclude_default = ['Vp', 'Vs', 'DEPTH', 'VPVSMOD', 'PIMPMOD', 'SIMPMOD']
            available_cols = [col for col in df.columns if col not in [vp_col, vs_col]]
            
            selected_features = st.multiselect(
                "Select features for modeling",
                available_cols,
                default=[col for col in ['porosity', 'rho', 'sw', 'Vclay', 'RT', 'GR'] 
                        if col in available_cols]
            )
            
            st.session_state.selected_features = selected_features
        
        with col2:
            st.markdown("### ⚛️ Physics Model Configuration")
            
            # Matrix properties
            st.markdown("#### Matrix Properties")
            
            col_a, col_b = st.columns(2)
            with col_a:
                matrix_vp = st.number_input("Matrix Vp (m/s)", min_value=1000, max_value=8000, 
                                          value=5500, step=100)
                matrix_vs = st.number_input("Matrix Vs (m/s)", min_value=500, max_value=5000,
                                          value=3000, step=100)
            with col_b:
                matrix_rho = st.number_input("Matrix ρ (g/cc)", min_value=2.0, max_value=3.0,
                                           value=2.71, step=0.01)
            
            matrix_props = {
                'Vp': matrix_vp,
                'Vs': matrix_vs,
                'rho': matrix_rho
            }
            
            # Crack properties
            st.markdown("#### Crack Properties")
            
            aspect_ratio = st.slider("Aspect Ratio (α)", 0.001, 0.1, 0.01, 0.001,
                                   help="Crack thickness/length ratio")
            
            fluid_k = st.number_input("Fluid Bulk Modulus (GPa)", min_value=0.1, max_value=10.0,
                                    value=2.25, step=0.1) * 1e9
            
            crack_props = {
                'aspect_ratio': aspect_ratio,
                'fluid_K': fluid_k,
                'fluid_rho': 1000
            }
            
            st.session_state.matrix_props = matrix_props
            st.session_state.crack_props = crack_props
        
        # Model parameters
        st.markdown("---")
        st.markdown('<h3 class="section-header">🤖 Machine Learning Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Estimators", 50, 500, 200, 50)
            max_depth = st.slider("Max Depth", 3, 15, 6, 1)
        
        with col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
        
        with col3:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1)
            subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05)
        
        st.session_state.ml_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample
        }
        
        # Save configuration
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("✅ Configuration saved!")
            st.info("Go to **Run Analysis** to train the model.")
    
    # ==========================================================================
    # RUN ANALYSIS PAGE
    # ==========================================================================
    elif app_mode == "🚀 Run Analysis":
        st.markdown('<h2 class="sub-header">🚀 Run Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first.")
            return
        
        if 'selected_features' not in st.session_state:
            st.warning("⚠️ Please configure the model first.")
            return
        
        df = st.session_state.df
        
        # Analysis progress
        st.markdown("### 📊 Analysis Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Prepare data
        status_text.text("Step 1/5: Preparing data...")
        progress_bar.progress(20)
        
        # Get matrix properties
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
        status_text.text("Step 2/5: Creating features...")
        progress_bar.progress(40)
        
        df_features = hybrid_model.create_safe_features(df)
        
        # Select features
        selected_features = st.session_state.selected_features
        exclude_cols = ['Vp', 'Vs', 'VPVSMOD', 'PIMPMOD', 'SIMPMOD']
        if 'DEPTH' in df.columns:
            exclude_cols.append('DEPTH')
        
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df_features[col])]
        
        X = df_features[feature_cols]
        y_vp = df['Vp'].values
        y_vs = df['Vs'].values
        
        # Step 3: Split data
        status_text.text("Step 3/5: Splitting data...")
        progress_bar.progress(60)
        
        X_train, X_test, y_vp_train, y_vp_test, y_vs_train, y_vs_test = train_test_split(
            X, y_vp, y_vs, test_size=test_size/100, random_state=42
        )
        
        # Step 4: Train model
        status_text.text("Step 4/5: Training model...")
        progress_bar.progress(80)
        
        hybrid_model.train(X_train, y_vp_train, y_vs_train, model_type)
        
        # Step 5: Make predictions
        status_text.text("Step 5/5: Making predictions...")
        
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
            'X_train': X_train,
            'X_test': X_test,
            'y_vp_train': y_vp_train,
            'y_vp_test': y_vp_test,
            'y_vs_train': y_vs_train,
            'y_vs_test': y_vs_test,
            'vp_test_pred': vp_test_pred,
            'vs_test_pred': vs_test_pred,
            'vp_full_pred': vp_full_pred,
            'vs_full_pred': vs_full_pred,
            'feature_cols': feature_cols,
            'df_features': df_features
        }
        
        st.session_state.hybrid_model = hybrid_model
        
        # Calculate and display metrics
        st.markdown("---")
        st.markdown('<h3 class="section-header">📈 Quick Results</h3>', unsafe_allow_html=True)
        
        # Test set metrics
        vp_test_corr = np.corrcoef(y_vp_test, vp_test_pred)[0, 1]
        vs_test_corr = np.corrcoef(y_vs_test, vs_test_pred)[0, 1]
        
        vp_test_r2 = r2_score(y_vp_test, vp_test_pred)
        vs_test_r2 = r2_score(y_vs_test, vs_test_pred)
        
        vp_test_mae = mean_absolute_error(y_vp_test, vp_test_pred)
        vs_test_mae = mean_absolute_error(y_vs_test, vs_test_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Vp Correlation", f"{vp_test_corr:.4f}")
        with col2:
            st.metric("Vs Correlation", f"{vs_test_corr:.4f}")
        with col3:
            st.metric("Vp R² Score", f"{vp_test_r2:.4f}")
        with col4:
            st.metric("Vs R² Score", f"{vs_test_r2:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vp MAE", f"{vp_test_mae:.0f} m/s")
        with col2:
            st.metric("Vs MAE", f"{vs_test_mae:.0f} m/s")
        
        # Success assessment
        st.markdown("---")
        st.markdown('<h3 class="section-header">🎯 Target Achievement</h3>', unsafe_allow_html=True)
        
        target = 0.75
        vp_success = vp_test_corr >= target
        vs_success = vs_test_corr >= target
        
        if vp_success and vs_success:
            st.markdown('<div class="success-box">🎉 SUCCESS: Both Vp and Vs correlations exceed 0.75!</div>', unsafe_allow_html=True)
        elif vp_success:
            st.markdown('<div class="warning-box">⚠️ PARTIAL: Vp achieved target but Vs needs improvement.</div>', unsafe_allow_html=True)
        elif vs_success:
            st.markdown('<div class="warning-box">⚠️ PARTIAL: Vs achieved target but Vp needs improvement.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ NEEDS IMPROVEMENT: Both Vp and Vs below target.</div>', unsafe_allow_html=True)
        
        # Show feature importance
        if hasattr(hybrid_model, 'feature_importances'):
            st.markdown("### 🔍 Top Features")
            
            importances = hybrid_model.feature_importances['Vp']
            top_n = min(10, len(feature_cols))
            sorted_idx = np.argsort(importances)[-top_n:][::-1]
            
            importance_df = pd.DataFrame({
                'Feature': [feature_cols[i] for i in sorted_idx],
                'Importance': [importances[i] for i in sorted_idx]
            })
            
            st.dataframe(importance_df, use_container_width=True)
        
        # Export results
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Vp_predicted'] = vp_full_pred
        result_df['Vs_predicted'] = vs_full_pred
        result_df['Vp_error_%'] = 100 * (vp_full_pred - y_vp) / y_vp
        result_df['Vs_error_%'] = 100 * (vs_full_pred - y_vs) / y_vs
        result_df['VpVs_predicted'] = vp_full_pred / vs_full_pred
        result_df['VpVs_actual'] = y_vp / y_vs
        
        csv = result_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="carbonate_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("📊 View Detailed Results", use_container_width=True):
                st.session_state.show_results = True
                st.experimental_rerun()
    
    # ==========================================================================
    # RESULTS PAGE
    # ==========================================================================
    elif app_mode == "📈 Results":
        st.markdown('<h2 class="sub-header">📈 Detailed Results</h2>', unsafe_allow_html=True)
        
        if st.session_state.results is None:
            st.warning("⚠️ Please run the analysis first.")
            return
        
        results = st.session_state.results
        hybrid_model = st.session_state.hybrid_model
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Cross-Plots", "📈 Error Analysis", "🔍 Feature Importance", "📋 Prediction Table"])
        
        with tab1:
            st.markdown("### 📊 Predicted vs Actual Velocities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vp cross-plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(results['y_vp_test'], results['vp_test_pred'], 
                          alpha=0.5, s=20, color='blue')
                
                vp_min = min(results['y_vp_test'].min(), results['vp_test_pred'].min())
                vp_max = max(results['y_vp_test'].max(), results['vp_test_pred'].max())
                ax.plot([vp_min, vp_max], [vp_min, vp_max], 'r--', linewidth=2)
                
                ax.set_xlabel('Measured Vp (m/s)')
                ax.set_ylabel('Predicted Vp (m/s)')
                ax.set_title('Vp: Test Set Predictions')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Vs cross-plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(results['y_vs_test'], results['vs_test_pred'], 
                          alpha=0.5, s=20, color='green')
                
                vs_min = min(results['y_vs_test'].min(), results['vs_test_pred'].min())
                vs_max = max(results['y_vs_test'].max(), results['vs_test_pred'].max())
                ax.plot([vs_min, vs_max], [vs_min, vs_max], 'r--', linewidth=2)
                
                ax.set_xlabel('Measured Vs (m/s)')
                ax.set_ylabel('Predicted Vs (m/s)')
                ax.set_title('Vs: Test Set Predictions')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Vp/Vs ratio
            st.markdown("### 📐 Vp/Vs Ratio Comparison")
            
            actual_vpvs = results['y_vp_test'] / results['y_vs_test']
            pred_vpvs = results['vp_test_pred'] / results['vs_test_pred']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(actual_vpvs, pred_vpvs, alpha=0.5, s=20, color='purple')
            
            vpvs_min = min(actual_vpvs.min(), pred_vpvs.min())
            vpvs_max = max(actual_vpvs.max(), pred_vpvs.max())
            ax.plot([vpvs_min, vpvs_max], [vpvs_min, vpvs_max], 'r--', linewidth=2)
            
            ax.set_xlabel('Actual Vp/Vs')
            ax.set_ylabel('Predicted Vp/Vs')
            ax.set_title('Vp/Vs Ratio Prediction')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### 📈 Error Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vp error histogram
                vp_error = 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(vp_error, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax.axvline(x=vp_error.mean(), color='green', linestyle='-', linewidth=2)
                
                ax.set_xlabel('Vp Error (%)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Vp Error Distribution\nMean: {vp_error.mean():.2f}%, Std: {vp_error.std():.2f}%')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Vs error histogram
                vs_error = 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(vs_error, bins=30, alpha=0.7, color='green', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax.axvline(x=vs_error.mean(), color='orange', linestyle='-', linewidth=2)
                
                ax.set_xlabel('Vs Error (%)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Vs Error Distribution\nMean: {vs_error.mean():.2f}%, Std: {vs_error.std():.2f}%')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Error vs porosity
            st.markdown("### 🧪 Error vs Porosity Relationship")
            
            if 'porosity' in st.session_state.df.columns:
                porosity_test = st.session_state.df.loc[results['X_test'].index, 'porosity']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.scatter(porosity_test, vp_error, alpha=0.5, s=20, color='blue')
                ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax1.set_xlabel('Porosity')
                ax1.set_ylabel('Vp Error (%)')
                ax1.set_title('Vp Error vs Porosity')
                ax1.grid(True, alpha=0.3)
                
                ax2.scatter(porosity_test, vs_error, alpha=0.5, s=20, color='green')
                ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
                ax2.set_xlabel('Porosity')
                ax2.set_ylabel('Vs Error (%)')
                ax2.set_title('Vs Error vs Porosity')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        with tab3:
            st.markdown("### 🔍 Feature Importance Analysis")
            
            if hasattr(hybrid_model, 'feature_importances'):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Vp feature importance
                    importances_vp = hybrid_model.feature_importances['Vp']
                    top_n = min(15, len(results['feature_cols']))
                    sorted_idx_vp = np.argsort(importances_vp)[-top_n:][::-1]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    y_pos = np.arange(top_n)
                    ax.barh(y_pos, importances_vp[sorted_idx_vp])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([results['feature_cols'][i] for i in sorted_idx_vp])
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Top Features for Vp Prediction')
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                
                with col2:
                    # Vs feature importance
                    importances_vs = hybrid_model.feature_importances['Vs']
                    sorted_idx_vs = np.argsort(importances_vs)[-top_n:][::-1]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    y_pos = np.arange(top_n)
                    ax.barh(y_pos, importances_vs[sorted_idx_vs], color='green')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([results['feature_cols'][i] for i in sorted_idx_vs])
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Top Features for Vs Prediction')
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                
                # Compare Vp vs Vs importance
                st.markdown("### 🔄 Vp vs Vs Feature Importance Comparison")
                
                comparison_data = []
                for i, feat in enumerate(results['feature_cols']):
                    if i < len(importances_vp) and i < len(importances_vs):
                        comparison_data.append({
                            'Feature': feat,
                            'Vp_Importance': importances_vp[i],
                            'Vs_Importance': importances_vs[i]
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Vp_Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(comparison_df))
                width = 0.35
                
                ax.bar(x - width/2, comparison_df['Vp_Importance'], width, label='Vp', color='blue')
                ax.bar(x + width/2, comparison_df['Vs_Importance'], width, label='Vs', color='green')
                
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance Score')
                ax.set_title('Feature Importance Comparison: Vp vs Vs')
                ax.set_xticks(x)
                ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
        
        with tab4:
            st.markdown("### 📋 Detailed Predictions Table")
            
            # Create results table
            result_table = pd.DataFrame({
                'Actual_Vp': results['y_vp_test'],
                'Predicted_Vp': results['vp_test_pred'],
                'Vp_Error_%': 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test'],
                'Actual_Vs': results['y_vs_test'],
                'Predicted_Vs': results['vs_test_pred'],
                'Vs_Error_%': 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
            })
            
            # Add original features if available
            if st.session_state.df is not None:
                original_df = st.session_state.df
                test_indices = results['X_test'].index
                
                for col in ['porosity', 'rho', 'sw', 'Vclay', 'RT', 'GR']:
                    if col in original_df.columns:
                        result_table[col] = original_df.loc[test_indices, col].values
            
            # Sort by Vp error
            result_table = result_table.sort_values('Vp_Error_%', key=abs)
            
            # Display table
            st.dataframe(result_table.round(2), use_container_width=True)
            
            # Statistics
            st.markdown("#### 📊 Prediction Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Vp Mean Error", f"{result_table['Vp_Error_%'].mean():.2f}%")
            with col2:
                st.metric("Vp Std Error", f"{result_table['Vp_Error_%'].std():.2f}%")
            with col3:
                st.metric("Vs Mean Error", f"{result_table['Vs_Error_%'].mean():.2f}%")
            with col4:
                st.metric("Vs Std Error", f"{result_table['Vs_Error_%'].std():.2f}%")
    
    # ==========================================================================
    # SINGLE PREDICTION PAGE
    # ==========================================================================
    elif app_mode == "🔮 Single Prediction":
        st.markdown('<h2 class="sub-header">🔮 Single Sample Prediction</h2>', unsafe_allow_html=True)
        
        if st.session_state.hybrid_model is None:
            st.warning("⚠️ Please train the model first in the Run Analysis section.")
            return
        
        hybrid_model = st.session_state.hybrid_model
        
        st.markdown("### 📝 Enter Rock Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            porosity = st.number_input("Porosity (fraction)", min_value=0.0, max_value=0.5, 
                                     value=0.1, step=0.01)
            rho = st.number_input("Density (g/cc)", min_value=2.0, max_value=3.0,
                                value=2.65, step=0.01)
            sw = st.number_input("Water Saturation", min_value=0.0, max_value=1.0,
                               value=1.0, step=0.1)
        
        with col2:
            vclay = st.number_input("Clay Volume (%)", min_value=0.0, max_value=100.0,
                                  value=10.0, step=1.0)
            rt = st.number_input("Resistivity (ohm-m)", min_value=0.1, max_value=1000.0,
                               value=100.0, step=10.0)
            gr = st.number_input("Gamma Ray (API)", min_value=0.0, max_value=200.0,
                               value=50.0, step=5.0)
        
        with col3:
            # Get matrix properties from session state
            matrix_props = st.session_state.get('matrix_props', {
                'Vp': 5500, 'Vs': 3000, 'rho': 2.71
            })
            
            crack_props = st.session_state.get('crack_props', {
                'aspect_ratio': 0.01,
                'fluid_K': 2.25e9
            })
            
            # Display current physics model parameters
            st.markdown("#### ⚛️ Physics Parameters")
            st.info(f"Matrix Vp: {matrix_props['Vp']} m/s")
            st.info(f"Matrix Vs: {matrix_props['Vs']} m/s")
            st.info(f"Aspect Ratio: {crack_props['aspect_ratio']}")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'porosity': [porosity],
            'rho': [rho],
            'sw': [sw],
            'Vclay': [vclay],
            'RT': [rt],
            'GR': [gr]
        })
        
        # Add physics-based features
        efm_model = EffectiveFieldMethod(matrix_props, crack_props)
        
        crack_density, beta = efm_model.estimate_crack_parameters(
            porosity, sw, rt, vclay
        )
        
        eff_props = efm_model.calculate_effective_properties(crack_density, beta)
        
        input_data['crack_density_efm'] = crack_density
        input_data['orientation_beta'] = beta
        input_data['F_beta'] = eff_props['F_normal']
        input_data['Vp_efm'] = eff_props['Vp']
        input_data['Vs_efm'] = eff_props['Vs']
        
        # Feature engineering
        input_data['porosity_sq'] = porosity ** 2
        input_data['porosity_sqrt'] = np.sqrt(max(porosity, 0))
        input_data['density_porosity'] = rho * porosity
        
        if crack_density > 0:
            input_data['crack_porosity'] = crack_density * porosity
            input_data['crack_clay'] = crack_density * (vclay/100.0)
        
        # Ensure all required features are present
        if hasattr(hybrid_model, 'feature_names'):
            missing_features = set(hybrid_model.feature_names) - set(input_data.columns)
            for feat in missing_features:
                input_data[feat] = 0
        
        # Make prediction
        if st.button("🔮 Predict Velocities", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    # Get physics prediction
                    physics_vp = eff_props['Vp']
                    physics_vs = eff_props['Vs']
                    
                    # Get hybrid prediction
                    hybrid_vp, hybrid_vs = hybrid_model.hybrid_predict(input_data, physics_weight)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h3 class="section-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### ⚛️ Physics Model")
                        st.metric("Vp", f"{physics_vp:.0f} m/s")
                        st.metric("Vs", f"{physics_vs:.0f} m/s")
                        st.metric("Vp/Vs", f"{physics_vp/physics_vs:.2f}")
                    
                    with col2:
                        st.markdown("#### 🤖 Hybrid Model")
                        st.metric("Vp", f"{hybrid_vp[0]:.0f} m/s", 
                                 delta=f"{(hybrid_vp[0]-physics_vp):.0f} m/s")
                        st.metric("Vs", f"{hybrid_vs[0]:.0f} m/s",
                                 delta=f"{(hybrid_vs[0]-physics_vs):.0f} m/s")
                        st.metric("Vp/Vs", f"{hybrid_vp[0]/hybrid_vs[0]:.2f}")
                    
                    with col3:
                        st.markdown("#### 📊 Crack Parameters")
                        st.metric("Crack Density", f"{crack_density:.3f}")
                        st.metric("Orientation (β)", f"{beta:.3f} rad")
                        st.metric("Orientation Factor", f"{eff_props['F_normal']:.3f}")
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("### 📈 Velocity Comparison")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Bar chart
                    models = ['Physics', 'Hybrid']
                    vp_values = [physics_vp, hybrid_vp[0]]
                    vs_values = [physics_vs, hybrid_vs[0]]
                    
                    x = np.arange(len(models))
                    width = 0.35
                    
                    ax1.bar(x - width/2, vp_values, width, label='Vp', color='blue', alpha=0.7)
                    ax1.bar(x + width/2, vs_values, width, label='Vs', color='green', alpha=0.7)
                    
                    ax1.set_xlabel('Model')
                    ax1.set_ylabel('Velocity (m/s)')
                    ax1.set_title('Physics vs Hybrid Predictions')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(models)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3, axis='y')
                    
                    # Vp/Vs ratio
                    vpvs_physics = physics_vp / physics_vs
                    vpvs_hybrid = hybrid_vp[0] / hybrid_vs[0]
                    
                    ax2.bar(['Physics', 'Hybrid'], [vpvs_physics, vpvs_hybrid], 
                           color=['red', 'orange'], alpha=0.7)
                    ax2.set_ylabel('Vp/Vs Ratio')
                    ax2.set_title('Vp/Vs Ratio Comparison')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Geological Interpretation")
                    
                    interpretation_cols = st.columns(2)
                    
                    with interpretation_cols[0]:
                        st.markdown("#### 🪨 Rock Properties")
                        st.info(f"**Porosity**: {porosity:.3f} ({'Low' if porosity < 0.1 else 'Moderate' if porosity < 0.2 else 'High'})")
                        st.info(f"**Density**: {rho:.2f} g/cc ({'Typical carbonate' if 2.6 < rho < 2.8 else 'Check mineralogy'})")
                        st.info(f"**Clay Content**: {vclay:.1f}% ({'Clean' if vclay < 15 else 'Slightly shaly' if vclay < 35 else 'Shaly'})")
                    
                    with interpretation_cols[1]:
                        st.markdown("#### 🔍 Crack Analysis")
                        st.info(f"**Crack Density**: {crack_density:.3f} ({'Low' if crack_density < 0.1 else 'Moderate' if crack_density < 0.3 else 'High'})")
                        st.info(f"**Crack Orientation**: {'Random' if beta > 1.0 else 'Moderately aligned' if beta > 0.5 else 'Well-aligned'}")
                        st.info(f"**Fluid Saturation**: {'Fully water-saturated' if sw > 0.9 else 'Partially saturated'}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

# ==============================================================================
# RUN THE APP
# ==============================================================================

if __name__ == "__main__":
    main()
