"""
================================================================================
STREAMLIT APP: HYBRID PHYSICS-ML VELOCITY PREDICTION FOR CARBONATE ROCKS
================================================================================
Complete application with Plotly visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Scientific imports
from scipy.special import iv

# Set page configuration
st.set_page_config(
    page_title="Carbonate Velocity Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin: 0.5rem 0;
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
    .plot-container {
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# EFFECTIVE FIELD METHOD (PHYSICS MODEL)
# ==============================================================================

class EffectiveFieldMethod:
    """Physics-based model for cracked elastic media."""
    
    def __init__(self, matrix_props, crack_props):
        self.matrix = matrix_props.copy()
        self.crack = crack_props.copy()
        self._calculate_moduli()
    
    def _calculate_moduli(self):
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
    
    def orientation_factor(self, beta, distribution='uniform'):
        """Calculate orientation distribution factor."""
        beta = float(beta)
        
        if distribution == 'uniform':
            if beta <= 1e-10:
                return 1.0
            return (beta + np.sin(beta) * np.cos(beta)) / (2 * beta)
        
        elif distribution == 'von_mises':
            if beta <= 1e-10:
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
    
    def estimate_crack_params(self, porosity, sw=1.0, rt=1.0, vclay=0):
        """Estimate crack parameters from well logs."""
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
        
        # Clay reduction
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
    
    def calculate_effective(self, crack_density, beta):
        """Calculate effective elastic properties."""
        # Matrix properties
        K0 = self.matrix.get('K', 50e9)
        G0 = self.matrix.get('G', 30e9)
        rho0 = self.matrix.get('rho_kgm3', 2650)
        
        # Get orientation factor
        F_val = self.orientation_factor(beta, 'uniform')
        
        # Simplified effective moduli
        K_eff = K0 * (1.0 - crack_density * 0.8 * F_val)
        G_eff = G0 * (1.0 - crack_density * 0.6 * F_val)
        
        # Ensure positive moduli
        K_eff = max(K_eff, 0.1 * K0)
        G_eff = max(G_eff, 0.1 * G0)
        
        # Calculate velocities
        Vp_eff = np.sqrt((K_eff + 4.0 * G_eff / 3.0) / rho0)
        Vs_eff = np.sqrt(G_eff / rho0)
        
        return {
            'Vp': Vp_eff,
            'Vs': Vs_eff,
            'K': K_eff,
            'G': G_eff,
            'crack_density': crack_density,
            'beta': beta,
            'F': F_val,
            'Vp/Vs': Vp_eff / Vs_eff if Vs_eff > 0 else 0
        }

# ==============================================================================
# HYBRID PHYSICS-ML MODEL
# ==============================================================================

class HybridVelocityPredictor:
    """Hybrid model combining physics and machine learning."""
    
    def __init__(self, matrix_props=None):
        self.efm_model = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.matrix_props = matrix_props
        self.feature_importances = {}
        self.feature_names = []
    
    def create_features(self, df):
        """Create feature matrix from input data."""
        features = pd.DataFrame()
        
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
                porosity_val = features['porosity'].iloc[idx] if 'porosity' in features.columns else 0.1
                sw_val = features['sw'].iloc[idx] if 'sw' in features.columns else 1.0
                rt_val = features['RT'].iloc[idx] if 'RT' in features.columns else 1.0
                vclay_val = features['Vclay'].iloc[idx] if 'Vclay' in features.columns else 0.0
                
                crack_density, beta = self.efm_model.estimate_crack_params(
                    porosity_val, sw_val, rt_val, vclay_val
                )
                
                eff_props = self.efm_model.calculate_effective(crack_density, beta)
                
                crack_densities.append(crack_density)
                betas.append(beta)
                F_values.append(eff_props['F'])
                vp_efm.append(eff_props['Vp'])
                vs_efm.append(eff_props['Vs'])
            
            features['crack_density'] = crack_densities
            features['beta'] = betas
            features['F'] = F_values
            features['Vp_efm'] = vp_efm
            features['Vs_efm'] = vs_efm
        
        # Feature engineering
        if 'porosity' in features.columns:
            features['porosity_sq'] = features['porosity'] ** 2
        
        if 'rho' in features.columns and 'porosity' in features.columns:
            features['density_porosity'] = features['rho'] * features['porosity']
        
        if 'crack_density' in features.columns:
            if 'porosity' in features.columns:
                features['crack_porosity'] = features['crack_density'] * features['porosity']
        
        # Clean data
        features = features.fillna(features.median())
        features = features.replace([np.inf, -np.inf], np.nan).fillna(features.median())
        
        return features
    
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
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_samples_split=5, random_state=42
            )
            vs_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_samples_split=5, random_state=42
            )
        else:
            vp_model = RandomForestRegressor(
                n_estimators=100, max_depth=4, min_samples_split=5,
                random_state=42
            )
            vs_model = RandomForestRegressor(
                n_estimators=100, max_depth=4, min_samples_split=5,
                random_state=42
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
        """Make predictions using trained models."""
        if not self.ml_models:
            raise ValueError("Models not trained.")
        
        # Handle missing features
        missing_cols = set(self.feature_names) - set(X.columns)
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
        """Combine ML and physics predictions."""
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
# PLOTLY VISUALIZATION FUNCTIONS
# ==============================================================================

def create_scatter_with_r2(x_actual, y_predicted, title, x_label, y_label, color='blue'):
    """Create scatter plot with R² value."""
    # Calculate metrics
    if len(x_actual) > 1:
        r2 = r2_score(x_actual, y_predicted)
        corr = np.corrcoef(x_actual, y_predicted)[0, 1]
    else:
        r2 = 0
        corr = 0
    
    # Create plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_actual,
        y=y_predicted,
        mode='markers',
        marker=dict(color=color, size=8, opacity=0.6),
        name='Predictions',
        text=[f'Actual: {a:.0f}<br>Predicted: {p:.0f}' 
              for a, p in zip(x_actual, y_predicted)]
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
    
    # Add regression line
    if len(x_actual) > 1:
        z = np.polyfit(x_actual, y_predicted, 1)
        p = np.poly1d(z)
        reg_line = p([min_val, max_val])
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=reg_line,
            mode='lines',
            line=dict(color='green', width=2),
            name=f'Regression (slope={z[0]:.3f})'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{title}<br>R² = {r2:.4f}, Correlation = {corr:.4f}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig, r2, corr

def create_error_histogram(errors, title, color='blue'):
    """Create histogram of prediction errors."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        marker_color=color,
        opacity=0.7,
        name='Error Distribution'
    ))
    
    # Add vertical lines
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=mean_error, line_width=2, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title=f"{title}<br>Mean: {mean_error:.2f}, Std: {std_error:.2f}",
        xaxis_title='Error',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        bargap=0.1
    )
    
    return fig

def create_feature_importance_plot(feature_names, importances, title, color='steelblue'):
    """Create horizontal bar plot for feature importance."""
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(10, len(feature_names))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[feature_names[i] for i in sorted_idx[:top_n]],
        x=[importances[i] for i in sorted_idx[:top_n]],
        orientation='h',
        marker_color=color
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================

def main():
    """Main application function."""
    
    st.markdown('<h1 class="main-header">🎯 Hybrid Physics-ML Velocity Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📁 Upload Data", "⚙️ Configure", "🚀 Run Analysis", "📈 Results"]
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        # Global settings
        physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.3, 0.1)
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
        model_type = st.selectbox("ML Model", ["Gradient Boosting", "Random Forest"])
        
        st.markdown("---")
        st.markdown("## 📚 About")
        st.info("""
        **Hybrid Approach:**
        - Physics: Effective Field Method
        - ML: Gradient Boosting / Random Forest
        - Target: Correlation > 0.75
        """)
    
    # ==========================================================================
    # HOME PAGE
    # ==========================================================================
    if page == "🏠 Home":
        st.markdown("## Welcome to Carbonate Velocity Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 About This App")
            st.markdown("""
            This application predicts acoustic velocities in carbonate rocks using:
            
            1. **Physics Model**: Effective Field Method based on micromechanics
            2. **Machine Learning**: Ensemble methods for complex relationships
            3. **Hybrid Integration**: Weighted combination for robust predictions
            
            ### 🎯 Key Features
            - Interactive Plotly visualizations
            - R² and correlation metrics
            - Feature importance analysis
            - Single sample prediction
            - Results export
            """)
        
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. **Upload** your CSV data
            2. **Configure** model parameters
            3. **Run** the analysis
            4. **Explore** interactive results
            
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
            mime="text/csv"
        )
    
    # ==========================================================================
    # UPLOAD DATA PAGE
    # ==========================================================================
    elif page == "📁 Upload Data":
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
        
        # Data visualization if data is loaded
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("### 📈 Data Visualization")
            
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, 
                                         index=numeric_cols.index('porosity') if 'porosity' in numeric_cols else 0)
                with col2:
                    y_axis = st.selectbox("Y-axis", numeric_cols,
                                         index=numeric_cols.index('Vp') if 'Vp' in numeric_cols else 1)
                
                # Create scatter plot
                fig = px.scatter(df, x=x_axis, y=y_axis, 
                               title=f'{y_axis} vs {x_axis}',
                               template='plotly_white')
                
                fig.update_traces(marker=dict(size=8, opacity=0.6))
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plot
            st.markdown("### 📊 Distribution Analysis")
            
            hist_col = st.selectbox("Select column for histogram", numeric_cols)
            
            fig = px.histogram(df, x=hist_col, nbins=30,
                             title=f"Distribution of {hist_col}",
                             template='plotly_white')
            
            fig.update_layout(height=400, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # CONFIGURE PAGE
    # ==========================================================================
    elif page == "⚙️ Configure":
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
            
            col_a, col_b = st.columns(2)
            with col_a:
                matrix_vp = st.number_input("Matrix Vp (m/s)", 3000, 7000, int(estimated_vp), 100)
                matrix_vs = st.number_input("Matrix Vs (m/s)", 1500, 4000, int(estimated_vs), 100)
            with col_b:
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
    elif page == "🚀 Run Analysis":
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
        
        df_features = hybrid_model.create_features(df[selected_features])
        
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
        
        hybrid_model.train(X_train, y_vp_train, y_vs_train, model_type)
        
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
        
        # Quick visualization
        st.markdown("---")
        st.markdown("### 👁️ Quick Preview")
        
        # Create comparison plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Vp (R²={vp_r2:.3f})', f'Vs (R²={vs_r2:.3f})'),
            horizontal_spacing=0.15
        )
        
        # Vp plot
        fig.add_trace(
            go.Scatter(x=y_vp_test, y=vp_test_pred, mode='markers',
                      marker=dict(color='blue', opacity=0.6), name='Vp'),
            row=1, col=1
        )
        
        # Vs plot
        fig.add_trace(
            go.Scatter(x=y_vs_test, y=vs_test_pred, mode='markers',
                      marker=dict(color='green', opacity=0.6), name='Vs'),
            row=1, col=2
        )
        
        # Add perfect fit lines
        for col in [1, 2]:
            data = y_vp_test if col == 1 else y_vs_test
            pred = vp_test_pred if col == 1 else vs_test_pred
            min_val = min(min(data), min(pred))
            max_val = max(max(data), max(pred))
            
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', line=dict(color='red', dash='dash'),
                          showlegend=False),
                row=1, col=col
            )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text='Measured (m/s)', row=1, col=1)
        fig.update_xaxes(title_text='Measured (m/s)', row=1, col=2)
        fig.update_yaxes(title_text='Predicted (m/s)', row=1, col=1)
        fig.update_yaxes(title_text='Predicted (m/s)', row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # RESULTS PAGE
    # ==========================================================================
    elif page == "📈 Results":
        st.markdown("## 📈 Detailed Results")
        
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please run the analysis first.")
            return
        
        results = st.session_state.results
        hybrid_model = results['model']
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Prediction Plots", 
            "📈 Error Analysis", 
            "🔍 Features",
            "📋 Data"
        ])
        
        with tab1:
            st.markdown("### 📊 Prediction Results")
            
            # Vp prediction plot
            fig_vp, r2_vp, corr_vp = create_scatter_with_r2(
                results['y_vp_test'], results['vp_test_pred'],
                "P-wave Velocity (Vp) Prediction",
                "Measured Vp (m/s)",
                "Predicted Vp (m/s)",
                'blue'
            )
            
            st.plotly_chart(fig_vp, use_container_width=True)
            
            # Vs prediction plot
            fig_vs, r2_vs, corr_vs = create_scatter_with_r2(
                results['y_vs_test'], results['vs_test_pred'],
                "S-wave Velocity (Vs) Prediction",
                "Measured Vs (m/s)",
                "Predicted Vs (m/s)",
                'green'
            )
            
            st.plotly_chart(fig_vs, use_container_width=True)
            
            # Vp/Vs ratio
            st.markdown("### 📐 Vp/Vs Ratio")
            
            vpvs_actual = results['y_vp_test'] / results['y_vs_test']
            vpvs_pred = results['vp_test_pred'] / results['vs_test_pred']
            
            fig_vpvs, r2_vpvs, corr_vpvs = create_scatter_with_r2(
                vpvs_actual, vpvs_pred,
                "Vp/Vs Ratio Prediction",
                "Actual Vp/Vs",
                "Predicted Vp/Vs",
                'purple'
            )
            
            st.plotly_chart(fig_vpvs, use_container_width=True)
        
        with tab2:
            st.markdown("### 📈 Error Analysis")
            
            # Calculate errors
            vp_error = 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test']
            vs_error = 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_vp_error = create_error_histogram(
                    vp_error, "Vp Error Distribution", 'blue'
                )
                st.plotly_chart(fig_vp_error, use_container_width=True)
            
            with col2:
                fig_vs_error = create_error_histogram(
                    vs_error, "Vs Error Distribution", 'green'
                )
                st.plotly_chart(fig_vs_error, use_container_width=True)
            
            # Error statistics
            st.markdown("### 📊 Error Statistics")
            
            error_stats = pd.DataFrame({
                'Metric': ['Mean Error (%)', 'Std Error (%)', 'MAE (m/s)', 'RMSE (m/s)'],
                'Vp': [
                    f"{vp_error.mean():.2f}",
                    f"{vp_error.std():.2f}",
                    f"{mean_absolute_error(results['y_vp_test'], results['vp_test_pred']):.0f}",
                    f"{np.sqrt(mean_squared_error(results['y_vp_test'], results['vp_test_pred'])):.0f}"
                ],
                'Vs': [
                    f"{vs_error.mean():.2f}",
                    f"{vs_error.std():.2f}",
                    f"{mean_absolute_error(results['y_vs_test'], results['vs_test_pred']):.0f}",
                    f"{np.sqrt(mean_squared_error(results['y_vs_test'], results['vs_test_pred'])):.0f}"
                ]
            })
            
            st.dataframe(error_stats, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Feature Importance")
            
            if hasattr(hybrid_model, 'feature_importances'):
                # Vp feature importance
                fig_vp_imp = create_feature_importance_plot(
                    results['feature_cols'],
                    hybrid_model.feature_importances['Vp'],
                    "Top Features for Vp Prediction",
                    'blue'
                )
                
                st.plotly_chart(fig_vp_imp, use_container_width=True)
                
                # Vs feature importance
                fig_vs_imp = create_feature_importance_plot(
                    results['feature_cols'],
                    hybrid_model.feature_importances['Vs'],
                    "Top Features for Vs Prediction",
                    'green'
                )
                
                st.plotly_chart(fig_vs_imp, use_container_width=True)
            
            # Feature table
            st.markdown("### 📋 Feature Summary")
            
            feature_info = pd.DataFrame({
                'Feature': results['feature_cols'],
                'Type': 'Input' if len(results['feature_cols']) <= 10 else 'Engineered'
            })
            
            st.dataframe(feature_info, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Prediction Results")
            
            # Create results table
            results_df = pd.DataFrame({
                'Actual_Vp': results['y_vp_test'],
                'Predicted_Vp': results['vp_test_pred'],
                'Vp_Error_%': 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test'],
                'Actual_Vs': results['y_vs_test'],
                'Predicted_Vs': results['vs_test_pred'],
                'Vs_Error_%': 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
            })
            
            # Add some original features if available
            original_df = st.session_state.df
            test_indices = results['X_test'].index
            
            for col in ['porosity', 'rho', 'Vclay']:
                if col in original_df.columns:
                    results_df[col] = original_df.loc[test_indices, col].values
            
            # Sort by absolute Vp error
            results_df['Abs_Vp_Error'] = np.abs(results_df['Vp_Error_%'])
            results_df = results_df.sort_values('Abs_Vp_Error')
            
            st.dataframe(results_df.round(2), use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="velocity_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Best and worst predictions
            st.markdown("### 🏆 Best & Worst Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Best Predictions (Lowest Error)")
                best = results_df.nsmallest(5, 'Abs_Vp_Error')
                st.dataframe(best[['Actual_Vp', 'Predicted_Vp', 'Vp_Error_%']].round(2))
            
            with col2:
                st.markdown("#### ❌ Worst Predictions (Highest Error)")
                worst = results_df.nlargest(5, 'Abs_Vp_Error')
                st.dataframe(worst[['Actual_Vp', 'Predicted_Vp', 'Vp_Error_%']].round(2))
            
            # Overall summary
            st.markdown("### 📊 Overall Summary")
            
            summary_data = {
                'Total Samples': len(results['y_vp']),
                'Test Samples': len(results['y_vp_test']),
                'Vp R²': f"{r2_score(results['y_vp_test'], results['vp_test_pred']):.4f}",
                'Vs R²': f"{r2_score(results['y_vs_test'], results['vs_test_pred']):.4f}",
                'Vp Correlation': f"{np.corrcoef(results['y_vp_test'], results['vp_test_pred'])[0, 1]:.4f}",
                'Vs Correlation': f"{np.corrcoef(results['y_vs_test'], results['vs_test_pred'])[0, 1]:.4f}",
                'Features Used': len(results['feature_cols']),
                'Physics Weight': physics_weight
            }
            
            summary_df = pd.DataFrame(list(summary_data.items()), 
                                    columns=['Metric', 'Value'])
            
            st.dataframe(summary_df, use_container_width=True)

# ==============================================================================
# RUN APPLICATION
# ==============================================================================

if __name__ == "__main__":
    main()
