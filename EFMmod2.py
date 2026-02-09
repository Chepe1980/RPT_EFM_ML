"""
================================================================================
STREAMLIT APP: HYBRID PHYSICS-ML VELOCITY PREDICTION FOR CARBONATE ROCKS
================================================================================
Enhanced with Plotly for interactive visualizations
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
from sklearn.model_selection import train_test_split, cross_val_score
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
    .plotly-plot {
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
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
            
            st.session_state.debug_info = f"""
            Matrix Properties Calculated:
            - Shear modulus (G): {mat['G']/1e9:.1f} GPa
            - Bulk modulus (K): {mat['K']/1e9:.1f} GPa
            - Poisson's ratio (ν): {mat['nu']:.3f}
            """
    
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
        self.training_history = {'Vp': {}, 'Vs': {}}
    
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
            features_df['porosity_log'] = np.log1p(features_df['porosity'])
        
        if 'rho' in features_df.columns and 'porosity' in features_df.columns:
            features_df['density_porosity'] = features_df['rho'] * features_df['porosity']
            features_df['density_over_porosity'] = features_df['rho'] / (features_df['porosity'] + 0.01)
        
        if 'crack_density_efm' in features_df.columns:
            if 'porosity' in features_df.columns:
                features_df['crack_porosity'] = features_df['crack_density_efm'] * features_df['porosity']
                features_df['crack_porosity_ratio'] = features_df['crack_density_efm'] / (features_df['porosity'] + 0.01)
            
            if 'Vclay' in features_df.columns:
                features_df['crack_clay'] = features_df['crack_density_efm'] * features_df['Vclay']
                features_df['crack_clay_ratio'] = features_df['crack_density_efm'] / (features_df['Vclay'] + 0.01)
        
        if 'RT' in features_df.columns and 'porosity' in features_df.columns:
            features_df['F_formation_factor'] = features_df['RT'] / (features_df['porosity'] ** (-2) + 0.01)
        
        # Clean data
        features_df = features_df.fillna(features_df.median())
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.median())
        
        return features_df
    
    def train(self, X_train, y_train_vp, y_train_vs, model_type='Gradient Boosting', cv_folds=5):
        """Train ML models with cross-validation."""
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Impute and scale
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Select model
        if model_type == 'Gradient Boosting':
            vp_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=2, random_state=42,
                subsample=0.8, validation_fraction=0.1, n_iter_no_change=10
            )
            vs_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                min_samples_split=5, min_samples_leaf=2, random_state=42,
                subsample=0.8, validation_fraction=0.1, n_iter_no_change=10
            )
        else:  # Random Forest
            vp_model = RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            vs_model = RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
        
        # Cross-validation
        if cv_folds > 1:
            cv_scores_vp = cross_val_score(vp_model, X_train_scaled, y_train_vp, 
                                          cv=cv_folds, scoring='r2')
            cv_scores_vs = cross_val_score(vs_model, X_train_scaled, y_train_vs,
                                          cv=cv_folds, scoring='r2')
            
            self.training_history['Vp']['cv_scores'] = cv_scores_vp
            self.training_history['Vs']['cv_scores'] = cv_scores_vs
            self.training_history['Vp']['cv_mean'] = cv_scores_vp.mean()
            self.training_history['Vs']['cv_mean'] = cv_scores_vs.mean()
            self.training_history['Vp']['cv_std'] = cv_scores_vp.std()
            self.training_history['Vs']['cv_std'] = cv_scores_vs.std()
        
        # Train models
        vp_model.fit(X_train_scaled, y_train_vp)
        vs_model.fit(X_train_scaled, y_train_vs)
        
        self.ml_models['Vp'] = vp_model
        self.ml_models['Vs'] = vs_model
        
        # Store feature importances
        self.feature_importances['Vp'] = vp_model.feature_importances_
        self.feature_importances['Vs'] = vs_model.feature_importances_
        
        # Store training scores
        self.training_history['Vp']['train_score'] = vp_model.score(X_train_scaled, y_train_vp)
        self.training_history['Vs']['train_score'] = vs_model.score(X_train_scaled, y_train_vs)
    
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
    
    def get_feature_importance_df(self, top_n=20):
        """Get feature importance as DataFrame."""
        if not self.feature_importances or not self.feature_names:
            return pd.DataFrame()
        
        importance_data = []
        for i, feat in enumerate(self.feature_names):
            if i < len(self.feature_importances['Vp']):
                importance_data.append({
                    'Feature': feat,
                    'Vp_Importance': self.feature_importances['Vp'][i],
                    'Vs_Importance': self.feature_importances['Vs'][i],
                    'Total_Importance': (self.feature_importances['Vp'][i] + 
                                        self.feature_importances['Vs'][i]) / 2
                })
        
        df = pd.DataFrame(importance_data)
        df = df.sort_values('Total_Importance', ascending=False).head(top_n)
        return df

# ==============================================================================
# PLOTLY VISUALIZATION FUNCTIONS
# ==============================================================================

def create_scatter_plot_with_r2(x_actual, y_predicted, title, x_label, y_label, 
                               color='blue', symbol='circle', show_r2=True):
    """Create a scatter plot with R² value and perfect fit line."""
    
    # Calculate R²
    r2 = r2_score(x_actual, y_predicted)
    correlation = np.corrcoef(x_actual, y_predicted)[0, 1]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_actual,
        y=y_predicted,
        mode='markers',
        marker=dict(
            color=color,
            size=8,
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        name='Predictions',
        text=[f'Actual: {a:.0f}<br>Predicted: {p:.0f}' 
              for a, p in zip(x_actual, y_predicted)],
        hoverinfo='text'
    ))
    
    # Add perfect fit line
    min_val = min(x_actual.min(), y_predicted.min())
    max_val = max(x_actual.max(), y_predicted.max())
    line_range = [min_val, max_val]
    
    fig.add_trace(go.Scatter(
        x=line_range,
        y=line_range,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Perfect Fit'
    ))
    
    # Calculate regression line
    if len(x_actual) > 1:
        z = np.polyfit(x_actual, y_predicted, 1)
        p = np.poly1d(z)
        reg_line = p(line_range)
        
        fig.add_trace(go.Scatter(
            x=line_range,
            y=reg_line,
            mode='lines',
            line=dict(color='green', width=2),
            name=f'Regression (slope={z[0]:.3f})'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{title}<br>R² = {r2:.4f}, Correlation = {correlation:.4f}" if show_r2 else title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
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
    
    # Add vertical lines for mean and zero
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", 
                  annotation_text="Zero Error", annotation_position="top right")
    fig.add_vline(x=mean_error, line_width=2, line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {mean_error:.2f}", annotation_position="top left")
    
    # Add normal distribution curve
    if len(errors) > 10:
        x_norm = np.linspace(errors.min(), errors.max(), 100)
        y_norm = (1/(std_error * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mean_error)/std_error)**2)
        y_norm = y_norm * len(errors) * (errors.max() - errors.min()) / 30  # Scale to histogram
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            line=dict(color='black', width=2),
            name='Normal Distribution'
        ))
    
    fig.update_layout(
        title=f"{title}<br>Mean: {mean_error:.2f}, Std: {std_error:.2f}",
        xaxis_title='Error',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=True,
        bargap=0.1
    )
    
    return fig, mean_error, std_error

def create_feature_importance_plot(feature_names, importances, title, color='steelblue'):
    """Create horizontal bar plot for feature importance."""
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[feature_names[i] for i in sorted_idx[:top_n]],
        x=[importances[i] for i in sorted_idx[:top_n]],
        orientation='h',
        marker_color=color,
        text=[f'{importances[i]:.4f}' for i in sorted_idx[:top_n]],
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

def create_vp_vs_comparison_plot(vp_actual, vp_pred, vs_actual, vs_pred):
    """Create side-by-side comparison of Vp and Vs predictions."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P-wave Velocity (Vp)', 'S-wave Velocity (Vs)'),
        horizontal_spacing=0.15
    )
    
    # Vp plot
    r2_vp = r2_score(vp_actual, vp_pred)
    fig.add_trace(
        go.Scatter(
            x=vp_actual,
            y=vp_pred,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name=f'Vp (R²={r2_vp:.3f})',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Vs plot
    r2_vs = r2_score(vs_actual, vs_pred)
    fig.add_trace(
        go.Scatter(
            x=vs_actual,
            y=vs_pred,
            mode='markers',
            marker=dict(color='green', size=8, opacity=0.6),
            name=f'Vs (R²={r2_vs:.3f})',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add perfect fit lines
    for col in [1, 2]:
        min_val = min(vp_actual.min(), vp_pred.min()) if col == 1 else min(vs_actual.min(), vs_pred.min())
        max_val = max(vp_actual.max(), vp_pred.max()) if col == 1 else max(vs_actual.max(), vs_pred.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Perfect Fit',
                showlegend=(col == 1)
            ),
            row=1, col=col
        )
    
    # Update layout
    fig.update_layout(
        title='Velocity Prediction Comparison',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text='Measured (m/s)', row=1, col=1)
    fig.update_xaxes(title_text='Measured (m/s)', row=1, col=2)
    fig.update_yaxes(title_text='Predicted (m/s)', row=1, col=1)
    fig.update_yaxes(title_text='Predicted (m/s)', row=1, col=2)
    
    return fig, r2_vp, r2_vs

def create_porosity_velocity_plot(porosity, vp_actual, vp_pred, vs_actual, vs_pred):
    """Create porosity vs velocity plots."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Porosity vs Vp', 'Porosity vs Vs'),
        horizontal_spacing=0.15
    )
    
    # Vp vs Porosity
    fig.add_trace(
        go.Scatter(
            x=porosity,
            y=vp_actual,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Actual Vp'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=porosity,
            y=vp_pred,
            mode='markers',
            marker=dict(color='red', size=8, opacity=0.6, symbol='x'),
            name='Predicted Vp'
        ),
        row=1, col=1
    )
    
    # Vs vs Porosity
    fig.add_trace(
        go.Scatter(
            x=porosity,
            y=vs_actual,
            mode='markers',
            marker=dict(color='green', size=8, opacity=0.6),
            name='Actual Vs',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=porosity,
            y=vs_pred,
            mode='markers',
            marker=dict(color='orange', size=8, opacity=0.6, symbol='x'),
            name='Predicted Vs',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add trend lines if enough data
    if len(porosity) > 10:
        for col, (y_actual, y_pred, color_actual, color_pred) in enumerate([
            (vp_actual, vp_pred, 'darkblue', 'darkred'),
            (vs_actual, vs_pred, 'darkgreen', 'darkorange')
        ], 1):
            
            # Actual trend
            z_actual = np.polyfit(porosity, y_actual, 2)
            p_actual = np.poly1d(z_actual)
            porosity_sorted = np.sort(porosity)
            
            fig.add_trace(
                go.Scatter(
                    x=porosity_sorted,
                    y=p_actual(porosity_sorted),
                    mode='lines',
                    line=dict(color=color_actual, width=2),
                    name='Actual Trend',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            
            # Predicted trend
            z_pred = np.polyfit(porosity, y_pred, 2)
            p_pred = np.poly1d(z_pred)
            
            fig.add_trace(
                go.Scatter(
                    x=porosity_sorted,
                    y=p_pred(porosity_sorted),
                    mode='lines',
                    line=dict(color=color_pred, width=2, dash='dash'),
                    name='Predicted Trend',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
    
    # Update layout
    fig.update_layout(
        title='Porosity-Velocity Relationships',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text='Porosity', row=1, col=1)
    fig.update_xaxes(title_text='Porosity', row=1, col=2)
    fig.update_yaxes(title_text='Vp (m/s)', row=1, col=1)
    fig.update_yaxes(title_text='Vs (m/s)', row=1, col=2)
    
    return fig

def create_cross_validation_plot(cv_scores, title):
    """Create plot of cross-validation scores."""
    
    fig = go.Figure()
    
    # Add bars for each fold
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv_scores))],
        y=cv_scores,
        marker_color='lightblue',
        name='Fold Score'
    ))
    
    # Add mean line
    mean_score = np.mean(cv_scores)
    fig.add_hline(y=mean_score, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_score:.4f}", 
                  annotation_position="top right")
    
    fig.update_layout(
        title=f"{title}<br>Mean R²: {mean_score:.4f} ± {np.std(cv_scores):.4f}",
        xaxis_title='Cross-Validation Fold',
        yaxis_title='R² Score',
        template='plotly_white',
        height=400,
        showlegend=False,
        yaxis=dict(range=[max(0, min(cv_scores) - 0.1), min(1, max(cv_scores) + 0.1)])
    )
    
    return fig

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Hybrid Physics-ML Velocity Prediction for Carbonate Rocks</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        app_mode = st.radio(
            "Select Mode:",
            ["🏠 Home", "📁 Data Upload", "⚙️ Model Configuration", "🚀 Run Analysis", 
             "📈 Interactive Results", "🔮 Single Prediction", "📊 Advanced Analysis"]
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        # Global settings
        physics_weight = st.slider("Physics Weight", 0.0, 1.0, 0.3, 0.1,
                                 help="Weight for physics prediction (0=pure ML, 1=pure physics)")
        
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5,
                            help="Percentage of data for testing")
        
        model_type = st.selectbox(
            "ML Algorithm",
            ["Gradient Boosting", "Random Forest"],
            help="Select machine learning algorithm"
        )
        
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5, 1,
                           help="Number of folds for cross-validation")
        
        st.markdown("---")
        st.markdown("## 📚 About")
        st.info("""
        **Hybrid Physics-ML Approach:**
        - Physics: Effective Field Method (EFM)
        - ML: Ensemble methods (GB/RF)
        - Target: Correlation > 0.75
        
        **Features:**
        - Interactive Plotly visualizations
        - R² correlation metrics
        - Feature importance analysis
        - Cross-validation
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
    
    # ==========================================================================
    # HOME PAGE
    # ==========================================================================
    if app_mode == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to Carbonate Velocity Predictor</h2>', 
                   unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Overview
            
            This application predicts acoustic velocities (Vp, Vs) in carbonate rocks using a 
            **hybrid physics-machine learning approach**:
            
            1. **Physics Model**: Effective Field Method (EFM) based on micromechanical theory
            2. **Machine Learning**: Gradient Boosting or Random Forest regression
            3. **Hybrid Integration**: Weighted combination for robust predictions
            
            ### 📈 Key Features
            
            - **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
            - **Comprehensive Metrics**: R², correlation, MAE, RMSE
            - **Feature Importance**: Understand what drives predictions
            - **Cross-Validation**: Robust model evaluation
            - **Single Prediction**: Interactive prediction for individual samples
            """)
        
        with col2:
            st.markdown("### 🎯 Performance Targets")
            
            metrics = [
                ("Vp Correlation", "> 0.75", "🎯"),
                ("Vs Correlation", "> 0.75", "🎯"),
                ("Mean Error", "< 5%", "📊"),
                ("R² Score", "> 0.70", "📈")
            ]
            
            for metric, target, icon in metrics:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #1E3A8A;">
                        {icon} {metric}
                    </div>
                    <div style="font-size: 1.5rem; color: #10B981; font-weight: bold;">
                        {target}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick start guide
        st.markdown('<h3 class="section-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
        
        steps = st.columns(4)
        with steps[0]:
            st.markdown("### 1. Upload Data")
            st.markdown("Go to **Data Upload** and upload your CSV file")
        
        with steps[1]:
            st.markdown("### 2. Configure Model")
            st.markdown("Set parameters in **Model Configuration**")
        
        with steps[2]:
            st.markdown("### 3. Run Analysis")
            st.markdown("Execute analysis in **Run Analysis**")
        
        with steps[3]:
            st.markdown("### 4. View Results")
            st.markdown("Explore results in **Interactive Results**")
        
        st.markdown("---")
        
        # Sample data
        st.markdown('<h3 class="section-header">📋 Sample Data Format</h3>', unsafe_allow_html=True)
        
        sample_data = pd.DataFrame({
            'DEPTH': [1000, 1005, 1010, 1015, 1020],
            'Vp': [4500, 4450, 4400, 4350, 4300],
            'Vs': [2500, 2480, 2450, 2420, 2400],
            'porosity': [0.05, 0.08, 0.10, 0.12, 0.15],
            'rho': [2.65, 2.63, 2.60, 2.58, 2.55],
            'sw': [1.0, 1.0, 0.9, 0.8, 0.7],
            'Vclay': [5, 8, 10, 12, 15],
            'RT': [100, 80, 60, 50, 40],
            'GR': [30, 35, 40, 45, 50]
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
        st.markdown('<h2 class="sub-header">📁 Data Upload & Exploration</h2>', 
                   unsafe_allow_html=True)
        
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
                    
                    # Display metrics
                    st.markdown("### 📊 Data Overview")
                    
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Samples", len(df))
                    with metrics_cols[1]:
                        st.metric("Features", len(df.columns))
                    with metrics_cols[2]:
                        missing = df.isnull().sum().sum()
                        st.metric("Missing Values", missing)
                    with metrics_cols[3]:
                        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                    
                    # Data preview
                    st.markdown("### 👁️ Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
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
                    
                    # Create correlation matrix visualization
                    st.markdown("### 📈 Quick Correlation")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        
                        # Simple correlation with Vp and Vs
                        if 'Vp' in numeric_cols:
                            vp_correlations = corr_matrix['Vp'].sort_values(ascending=False)
                            top_correlated = vp_correlations[1:6]  # Skip self-correlation
                            
                            st.markdown("**Top features correlated with Vp:**")
                            for feat, corr in top_correlated.items():
                                st.progress(float(abs(corr)), 
                                          text=f"{feat}: {corr:.3f}")
        
        # Interactive data visualization
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown('<h3 class="section-header">📈 Interactive Data Visualization</h3>', 
                       unsafe_allow_html=True)
            
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, 
                                         index=numeric_cols.index('porosity') if 'porosity' in numeric_cols else 0)
                with col2:
                    y_axis = st.selectbox("Y-axis", numeric_cols,
                                         index=numeric_cols.index('Vp') if 'Vp' in numeric_cols else 1)
                with col3:
                    color_by = st.selectbox("Color by", ['None'] + numeric_cols)
                
                # Create interactive scatter plot
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by if color_by != 'None' else None,
                               hover_data=df.columns.tolist(),
                               title=f"{y_axis} vs {x_axis}",
                               template='plotly_white')
                
                fig.update_traces(marker=dict(size=10, opacity=0.6, line=dict(width=1, color='white')))
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True, className="plotly-plot")
            
            # Histogram for selected column
            st.markdown("### 📊 Distribution Analysis")
            
            hist_col = st.selectbox("Select column for histogram", numeric_cols)
            
            fig = px.histogram(df, x=hist_col, nbins=30, 
                             title=f"Distribution of {hist_col}",
                             template='plotly_white')
            
            fig.update_layout(height=400, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True, className="plotly-plot")
    
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
            target_options = [col for col in df.columns if col in ['Vp', 'Vs', 'vp', 'vs']]
            if not target_options:
                target_options = df.select_dtypes(include=[np.number]).columns.tolist()[:2]
            
            vp_col = st.selectbox("Vp column", df.columns, 
                                 index=list(df.columns).index('Vp') if 'Vp' in df.columns else 0)
            vs_col = st.selectbox("Vs column", df.columns,
                                 index=list(df.columns).index('Vs') if 'Vs' in df.columns else 1)
            
            # Feature selection
            st.markdown("### 🔧 Feature Selection")
            
            exclude_default = [vp_col, vs_col, 'DEPTH', 'depth', 'VPVSMOD', 'PIMPMOD', 'SIMPMOD']
            available_cols = [col for col in df.columns if col not in exclude_default]
            
            selected_features = st.multiselect(
                "Select features for modeling",
                available_cols,
                default=[col for col in ['porosity', 'rho', 'sw', 'Vclay', 'RT', 'GR'] 
                        if col in available_cols]
            )
            
            st.session_state.selected_features = selected_features
        
        with col2:
            st.markdown("### ⚛️ Physics Model Configuration")
            
            # Matrix properties estimation
            st.markdown("#### Matrix Properties Estimation")
            
            estimation_method = st.radio(
                "Estimation Method:",
                ["Use low-porosity samples", "Manual input", "Use statistics"]
            )
            
            if estimation_method == "Use low-porosity samples":
                if 'porosity' in df.columns:
                    porosity_threshold = df['porosity'].quantile(0.25)
                    low_porosity_samples = df[df['porosity'] <= porosity_threshold]
                    
                    if len(low_porosity_samples) > 0:
                        matrix_vp = float(low_porosity_samples[vp_col].mean())
                        matrix_vs = float(low_porosity_samples[vs_col].mean())
                        matrix_rho = float(low_porosity_samples['rho'].mean()) if 'rho' in low_porosity_samples.columns else 2.71
                        
                        st.info(f"Estimated from {len(low_porosity_samples)} low-porosity samples")
                        st.write(f"Vp: {matrix_vp:.0f} m/s")
                        st.write(f"Vs: {matrix_vs:.0f} m/s")
                        st.write(f"ρ: {matrix_rho:.2f} g/cc")
                    else:
                        matrix_vp = 5500
                        matrix_vs = 3000
                        matrix_rho = 2.71
                else:
                    st.warning("Porosity column not found for estimation")
                    matrix_vp = 5500
                    matrix_vs = 3000
                    matrix_rho = 2.71
            
            elif estimation_method == "Manual input":
                col_a, col_b = st.columns(2)
                with col_a:
                    matrix_vp = st.number_input("Matrix Vp (m/s)", 3000, 7000, 5500, 100)
                    matrix_vs = st.number_input("Matrix Vs (m/s)", 1500, 4000, 3000, 100)
                with col_b:
                    matrix_rho = st.number_input("Matrix ρ (g/cc)", 2.0, 3.0, 2.71, 0.01)
            else:  # Use statistics
                matrix_vp = float(df[vp_col].mean())
                matrix_vs = float(df[vs_col].mean())
                matrix_rho = float(df['rho'].mean()) if 'rho' in df.columns else 2.71
            
            matrix_props = {
                'Vp': matrix_vp,
                'Vs': matrix_vs,
                'rho': matrix_rho
            }
            
            # Crack properties
            st.markdown("#### Crack Properties")
            
            aspect_ratio = st.slider("Aspect Ratio (α)", 0.001, 0.1, 0.01, 0.001,
                                   help="Crack thickness/length ratio")
            
            fluid_k = st.number_input("Fluid Bulk Modulus (GPa)", 0.1, 10.0, 2.25, 0.1) * 1e9
            
            crack_props = {
                'aspect_ratio': aspect_ratio,
                'fluid_K': fluid_k,
                'fluid_rho': 1000
            }
            
            st.session_state.matrix_props = matrix_props
            st.session_state.crack_props = crack_props
        
        # ML Parameters
        st.markdown("---")
        st.markdown('<h3 class="section-header">🤖 Machine Learning Parameters</h3>', 
                   unsafe_allow_html=True)
        
        if model_type == "Gradient Boosting":
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
            
            ml_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample
            }
        else:  # Random Forest
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
                max_depth = st.slider("Max Depth", 3, 20, 10, 1)
            
            with col2:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1)
            
            ml_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
        
        st.session_state.ml_params = ml_params
        
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
        
        # Initialize physics model
        status_text.text("Step 1/6: Initializing physics model...")
        progress_bar.progress(10)
        
        matrix_props = st.session_state.get('matrix_props', {
            'Vp': 5500, 'Vs': 3000, 'rho': 2.71
        })
        
        crack_props = st.session_state.get('crack_props', {
            'aspect_ratio': 0.01,
            'fluid_K': 2.25e9,
            'fluid_rho': 1000
        })
        
        efm_model = EffectiveFieldMethod(matrix_props, crack_props)
        
        # Initialize hybrid model
        status_text.text("Step 2/6: Initializing hybrid model...")
        progress_bar.progress(20)
        
        hybrid_model = HybridVelocityPredictor(matrix_props)
        hybrid_model.efm_model = efm_model
        
        # Create features
        status_text.text("Step 3/6: Creating features...")
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
        
        # Split data
        status_text.text("Step 4/6: Splitting data...")
        progress_bar.progress(60)
        
        X_train, X_test, y_vp_train, y_vp_test, y_vs_train, y_vs_test = train_test_split(
            X, y_vp, y_vs, test_size=test_size/100, random_state=42, shuffle=True
        )
        
        # Train model
        status_text.text("Step 5/6: Training model...")
        progress_bar.progress(80)
        
        hybrid_model.train(X_train, y_vp_train, y_vs_train, model_type, cv_folds)
        
        # Make predictions
        status_text.text("Step 6/6: Making predictions...")
        
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
            'df_features': df_features,
            'feature_importance': hybrid_model.get_feature_importance_df(top_n=15)
        }
        
        st.session_state.hybrid_model = hybrid_model
        st.session_state.analysis_complete = True
        
        # Display metrics
        st.markdown("---")
        st.markdown('<h3 class="section-header">📈 Performance Metrics</h3>', unsafe_allow_html=True)
        
        # Calculate metrics
        vp_test_r2 = r2_score(y_vp_test, vp_test_pred)
        vs_test_r2 = r2_score(y_vs_test, vs_test_pred)
        
        vp_test_corr = np.corrcoef(y_vp_test, vp_test_pred)[0, 1]
        vs_test_corr = np.corrcoef(y_vs_test, vs_test_pred)[0, 1]
        
        vp_test_mae = mean_absolute_error(y_vp_test, vp_test_pred)
        vs_test_mae = mean_absolute_error(y_vs_test, vs_test_pred)
        
        vp_test_rmse = np.sqrt(mean_squared_error(y_vp_test, vp_test_pred))
        vs_test_rmse = np.sqrt(mean_squared_error(y_vs_test, vs_test_pred))
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Vp R² Score", f"{vp_test_r2:.4f}")
            st.metric("Vp Correlation", f"{vp_test_corr:.4f}")
        
        with col2:
            st.metric("Vs R² Score", f"{vs_test_r2:.4f}")
            st.metric("Vs Correlation", f"{vs_test_corr:.4f}")
        
        with col3:
            st.metric("Vp MAE", f"{vp_test_mae:.0f} m/s")
            st.metric("Vp RMSE", f"{vp_test_rmse:.0f} m/s")
        
        with col4:
            st.metric("Vs MAE", f"{vs_test_mae:.0f} m/s")
            st.metric("Vs RMSE", f"{vs_test_rmse:.0f} m/s")
        
        # Cross-validation results
        if hasattr(hybrid_model, 'training_history'):
            cv_info = hybrid_model.training_history
            
            if 'Vp' in cv_info and 'cv_mean' in cv_info['Vp']:
                st.markdown("### 📊 Cross-Validation Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vp CV Mean R²", f"{cv_info['Vp']['cv_mean']:.4f}")
                    st.metric("Vp CV Std", f"{cv_info['Vp']['cv_std']:.4f}")
                with col2:
                    st.metric("Vs CV Mean R²", f"{cv_info['Vs']['cv_mean']:.4f}")
                    st.metric("Vs CV Std", f"{cv_info['Vs']['cv_std']:.4f}")
        
        # Success assessment
        st.markdown("---")
        st.markdown('<h3 class="section-header">🎯 Target Achievement</h3>', unsafe_allow_html=True)
        
        target = 0.75
        vp_success = vp_test_corr >= target
        vs_success = vs_test_corr >= target
        
        if vp_success and vs_success:
            st.markdown('<div class="success-box">🎉 SUCCESS: Both Vp and Vs correlations exceed 0.75!</div>', 
                       unsafe_allow_html=True)
        elif vp_success:
            st.markdown('<div class="warning-box">⚠️ PARTIAL: Vp achieved target ({vp_test_corr:.4f}) but Vs needs improvement ({vs_test_corr:.4f})</div>', 
                       unsafe_allow_html=True)
        elif vs_success:
            st.markdown('<div class="warning-box">⚠️ PARTIAL: Vs achieved target ({vs_test_corr:.4f}) but Vp needs improvement ({vp_test_corr:.4f})</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ NEEDS IMPROVEMENT: Both Vp ({vp_test_corr:.4f}) and Vs ({vs_test_corr:.4f}) below target 0.75</div>', 
                       unsafe_allow_html=True)
        
        # Quick visualization
        st.markdown("---")
        st.markdown("### 👁️ Quick Preview")
        
        # Create comparison plot
        fig, r2_vp, r2_vs = create_vp_vs_comparison_plot(y_vp_test, vp_test_pred, 
                                                       y_vs_test, vs_test_pred)
        
        st.plotly_chart(fig, use_container_width=True, className="plotly-plot")
    
    # ==========================================================================
    # INTERACTIVE RESULTS PAGE
    # ==========================================================================
    elif app_mode == "📈 Interactive Results":
        st.markdown('<h2 class="sub-header">📈 Interactive Results Dashboard</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please run the analysis first.")
            return
        
        results = st.session_state.results
        hybrid_model = st.session_state.hybrid_model
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Prediction Plots", 
            "📈 Error Analysis", 
            "🔍 Feature Importance",
            "📋 Data Tables",
            "🎯 Performance Summary"
        ])
        
        with tab1:
            st.markdown("### 📊 Prediction vs Actual")
            
            # Vp prediction plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig_vp, r2_vp, corr_vp = create_scatter_plot_with_r2(
                    results['y_vp_test'], results['vp_test_pred'],
                    "P-wave Velocity (Vp) Prediction",
                    "Measured Vp (m/s)",
                    "Predicted Vp (m/s)",
                    color='blue'
                )
                st.plotly_chart(fig_vp, use_container_width=True, className="plotly-plot")
            
            with col2:
                fig_vs, r2_vs, corr_vs = create_scatter_plot_with_r2(
                    results['y_vs_test'], results['vs_test_pred'],
                    "S-wave Velocity (Vs) Prediction",
                    "Measured Vs (m/s)",
                    "Predicted Vs (m/s)",
                    color='green'
                )
                st.plotly_chart(fig_vs, use_container_width=True, className="plotly-plot")
            
            # Vp/Vs ratio comparison
            st.markdown("### 📐 Vp/Vs Ratio Comparison")
            
            vpvs_actual = results['y_vp_test'] / results['y_vs_test']
            vpvs_pred = results['vp_test_pred'] / results['vs_test_pred']
            
            fig_vpvs, r2_vpvs, corr_vpvs = create_scatter_plot_with_r2(
                vpvs_actual, vpvs_pred,
                "Vp/Vs Ratio Prediction",
                "Measured Vp/Vs",
                "Predicted Vp/Vs",
                color='purple'
            )
            st.plotly_chart(fig_vpvs, use_container_width=True, className="plotly-plot")
            
            # Porosity vs Velocity plots
            st.markdown("### 🧪 Porosity-Velocity Relationships")
            
            if 'porosity' in st.session_state.df.columns:
                porosity_test = st.session_state.df.loc[results['X_test'].index, 'porosity']
                
                fig_porosity = create_porosity_velocity_plot(
                    porosity_test,
                    results['y_vp_test'],
                    results['vp_test_pred'],
                    results['y_vs_test'],
                    results['vs_test_pred']
                )
                
                st.plotly_chart(fig_porosity, use_container_width=True, className="plotly-plot")
        
        with tab2:
            st.markdown("### 📈 Error Analysis")
            
            # Calculate errors
            vp_error = 100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test']
            vs_error = 100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_vp_error, vp_mean_error, vp_std_error = create_error_histogram(
                    vp_error, "Vp Prediction Error Distribution", color='blue'
                )
                st.plotly_chart(fig_vp_error, use_container_width=True, className="plotly-plot")
            
            with col2:
                fig_vs_error, vs_mean_error, vs_std_error = create_error_histogram(
                    vs_error, "Vs Prediction Error Distribution", color='green'
                )
                st.plotly_chart(fig_vs_error, use_container_width=True, className="plotly-plot")
            
            # Error statistics
            st.markdown("### 📊 Error Statistics")
            
            error_stats = pd.DataFrame({
                'Metric': ['Mean Error (%)', 'Std Error (%)', 'MAE (m/s)', 'RMSE (m/s)', 'Max Error (%)', 'Min Error (%)'],
                'Vp': [
                    f"{vp_error.mean():.2f}",
                    f"{vp_error.std():.2f}",
                    f"{mean_absolute_error(results['y_vp_test'], results['vp_test_pred']):.0f}",
                    f"{np.sqrt(mean_squared_error(results['y_vp_test'], results['vp_test_pred'])):.0f}",
                    f"{vp_error.max():.2f}",
                    f"{vp_error.min():.2f}"
                ],
                'Vs': [
                    f"{vs_error.mean():.2f}",
                    f"{vs_error.std():.2f}",
                    f"{mean_absolute_error(results['y_vs_test'], results['vs_test_pred']):.0f}",
                    f"{np.sqrt(mean_squared_error(results['y_vs_test'], results['vs_test_pred'])):.0f}",
                    f"{vs_error.max():.2f}",
                    f"{vs_error.min():.2f}"
                ]
            })
            
            st.dataframe(error_stats, use_container_width=True)
            
            # Error vs features
            st.markdown("### 🔍 Error vs Features Analysis")
            
            feature_for_error = st.selectbox(
                "Select feature to analyze error against",
                results['feature_cols']
            )
            
            if feature_for_error in results['X_test'].columns:
                feature_values = results['X_test'][feature_for_error].values
                
                fig_error_vs_feature = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'Vp Error vs {feature_for_error}', 
                                  f'Vs Error vs {feature_for_error}'),
                    horizontal_spacing=0.15
                )
                
                # Vp error
                fig_error_vs_feature.add_trace(
                    go.Scatter(
                        x=feature_values,
                        y=vp_error,
                        mode='markers',
                        marker=dict(color='blue', size=8, opacity=0.6),
                        name='Vp Error'
                    ),
                    row=1, col=1
                )
                
                # Vs error
                fig_error_vs_feature.add_trace(
                    go.Scatter(
                        x=feature_values,
                        y=vs_error,
                        mode='markers',
                        marker=dict(color='green', size=8, opacity=0.6),
                        name='Vs Error',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add zero lines
                for col in [1, 2]:
                    fig_error_vs_feature.add_hline(y=0, line_dash="dash", line_color="red",
                                                  row=1, col=col)
                
                fig_error_vs_feature.update_layout(
                    title=f'Prediction Errors vs {feature_for_error}',
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                fig_error_vs_feature.update_xaxes(title_text=feature_for_error, row=1, col=1)
                fig_error_vs_feature.update_xaxes(title_text=feature_for_error, row=1, col=2)
                fig_error_vs_feature.update_yaxes(title_text='Error (%)', row=1, col=1)
                fig_error_vs_feature.update_yaxes(title_text='Error (%)', row=1, col=2)
                
                st.plotly_chart(fig_error_vs_feature, use_container_width=True, className="plotly-plot")
        
        with tab3:
            st.markdown("### 🔍 Feature Importance Analysis")
            
            if results['feature_importance'] is not None and not results['feature_importance'].empty:
                # Top features for Vp
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_vp_importance = create_feature_importance_plot(
                        results['feature_importance']['Feature'].tolist(),
                        results['feature_importance']['Vp_Importance'].tolist(),
                        "Top Features for Vp Prediction",
                        color='blue'
                    )
                    st.plotly_chart(fig_vp_importance, use_container_width=True, className="plotly-plot")
                
                with col2:
                    fig_vs_importance = create_feature_importance_plot(
                        results['feature_importance']['Feature'].tolist(),
                        results['feature_importance']['Vs_Importance'].tolist(),
                        "Top Features for Vs Prediction",
                        color='green'
                    )
                    st.plotly_chart(fig_vs_importance, use_container_width=True, className="plotly-plot")
                
                # Feature importance comparison
                st.markdown("### 🔄 Vp vs Vs Feature Importance Comparison")
                
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    x=results['feature_importance']['Feature'],
                    y=results['feature_importance']['Vp_Importance'],
                    name='Vp Importance',
                    marker_color='blue'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    x=results['feature_importance']['Feature'],
                    y=results['feature_importance']['Vs_Importance'],
                    name='Vs Importance',
                    marker_color='green'
                ))
                
                fig_comparison.update_layout(
                    title='Feature Importance Comparison: Vp vs Vs',
                    xaxis_title='Features',
                    yaxis_title='Importance Score',
                    template='plotly_white',
                    height=500,
                    barmode='group',
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True, className="plotly-plot")
                
                # Display feature importance table
                st.markdown("### 📋 Feature Importance Table")
                st.dataframe(results['feature_importance'].round(4), use_container_width=True)
            else:
                st.warning("Feature importance data not available.")
            
            # Cross-validation plots if available
            if hasattr(hybrid_model, 'training_history'):
                cv_info = hybrid_model.training_history
                
                if 'Vp' in cv_info and 'cv_scores' in cv_info['Vp']:
                    st.markdown("### 📊 Cross-Validation Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cv_vp = create_cross_validation_plot(
                            cv_info['Vp']['cv_scores'],
                            "Vp Cross-Validation Scores"
                        )
                        st.plotly_chart(fig_cv_vp, use_container_width=True, className="plotly-plot")
                    
                    with col2:
                        fig_cv_vs = create_cross_validation_plot(
                            cv_info['Vs']['cv_scores'],
                            "Vs Cross-Validation Scores"
                        )
                        st.plotly_chart(fig_cv_vs, use_container_width=True, className="plotly-plot")
        
        with tab4:
            st.markdown("### 📋 Prediction Results Table")
            
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
            original_df = st.session_state.df
            test_indices = results['X_test'].index
            
            for col in ['porosity', 'rho', 'sw', 'Vclay', 'RT', 'GR']:
                if col in original_df.columns:
                    result_table[col] = original_df.loc[test_indices, col].values
            
            # Sort by absolute Vp error
            result_table['Abs_Vp_Error'] = np.abs(result_table['Vp_Error_%'])
            result_table = result_table.sort_values('Abs_Vp_Error')
            
            # Display table
            st.dataframe(result_table.round(2), use_container_width=True)
            
            # Download button
            csv = result_table.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Best and worst predictions
            st.markdown("### 🏆 Best & Worst Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Best Predictions (Lowest Error)")
                best_predictions = result_table.nsmallest(5, 'Abs_Vp_Error')
                st.dataframe(best_predictions[['Actual_Vp', 'Predicted_Vp', 'Vp_Error_%', 
                                              'Actual_Vs', 'Predicted_Vs', 'Vs_Error_%']].round(2))
            
            with col2:
                st.markdown("#### ❌ Worst Predictions (Highest Error)")
                worst_predictions = result_table.nlargest(5, 'Abs_Vp_Error')
                st.dataframe(worst_predictions[['Actual_Vp', 'Predicted_Vp', 'Vp_Error_%',
                                               'Actual_Vs', 'Predicted_Vs', 'Vs_Error_%']].round(2))
        
        with tab5:
            st.markdown("### 🎯 Performance Summary")
            
            # Calculate all metrics
            metrics_data = []
            
            # R² scores
            metrics_data.append({
                'Metric': 'R² Score',
                'Vp': f"{r2_score(results['y_vp_test'], results['vp_test_pred']):.4f}",
                'Vs': f"{r2_score(results['y_vs_test'], results['vs_test_pred']):.4f}",
                'Target': '> 0.70',
                'Status': '✅' if r2_score(results['y_vp_test'], results['vp_test_pred']) > 0.70 and 
                               r2_score(results['y_vs_test'], results['vs_test_pred']) > 0.70 else '⚠️'
            })
            
            # Correlation
            vp_corr = np.corrcoef(results['y_vp_test'], results['vp_test_pred'])[0, 1]
            vs_corr = np.corrcoef(results['y_vs_test'], results['vs_test_pred'])[0, 1]
            
            metrics_data.append({
                'Metric': 'Correlation',
                'Vp': f"{vp_corr:.4f}",
                'Vs': f"{vs_corr:.4f}",
                'Target': '> 0.75',
                'Status': '✅' if vp_corr > 0.75 and vs_corr > 0.75 else '⚠️'
            })
            
            # MAE
            vp_mae = mean_absolute_error(results['y_vp_test'], results['vp_test_pred'])
            vs_mae = mean_absolute_error(results['y_vs_test'], results['vs_test_pred'])
            
            metrics_data.append({
                'Metric': 'MAE (m/s)',
                'Vp': f"{vp_mae:.0f}",
                'Vs': f"{vs_mae:.0f}",
                'Target': '< 100',
                'Status': '✅' if vp_mae < 100 and vs_mae < 100 else '⚠️'
            })
            
            # RMSE
            vp_rmse = np.sqrt(mean_squared_error(results['y_vp_test'], results['vp_test_pred']))
            vs_rmse = np.sqrt(mean_squared_error(results['y_vs_test'], results['vs_test_pred']))
            
            metrics_data.append({
                'Metric': 'RMSE (m/s)',
                'Vp': f"{vp_rmse:.0f}",
                'Vs': f"{vs_rmse:.0f}",
                'Target': '< 150',
                'Status': '✅' if vp_rmse < 150 and vs_rmse < 150 else '⚠️'
            })
            
            # Mean Error %
            vp_mean_error = np.mean(100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test'])
            vs_mean_error = np.mean(100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test'])
            
            metrics_data.append({
                'Metric': 'Mean Error (%)',
                'Vp': f"{vp_mean_error:.2f}",
                'Vs': f"{vs_mean_error:.2f}",
                'Target': '< 5%',
                'Status': '✅' if abs(vp_mean_error) < 5 and abs(vs_mean_error) < 5 else '⚠️'
            })
            
            # Std Error %
            vp_std_error = np.std(100 * (results['vp_test_pred'] - results['y_vp_test']) / results['y_vp_test'])
            vs_std_error = np.std(100 * (results['vs_test_pred'] - results['y_vs_test']) / results['y_vs_test'])
            
            metrics_data.append({
                'Metric': 'Std Error (%)',
                'Vp': f"{vp_std_error:.2f}",
                'Vs': f"{vs_std_error:.2f}",
                'Target': '< 10%',
                'Status': '✅' if vp_std_error < 10 and vs_std_error < 10 else '⚠️'
            })
            
            # Create metrics table
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display with styling
            st.dataframe(metrics_df, use_container_width=True)
            
            # Overall assessment
            st.markdown("---")
            st.markdown("### 🎯 Overall Assessment")
            
            success_count = sum(1 for m in metrics_data if m['Status'] == '✅')
            total_metrics = len(metrics_data)
            
            if success_count == total_metrics:
                st.markdown('<div class="success-box">🎉 EXCELLENT: All performance targets achieved!</div>', 
                           unsafe_allow_html=True)
            elif success_count >= total_metrics * 0.7:
                st.markdown('<div class="info-box">📊 GOOD: Most performance targets achieved ({}/{})</div>'.format(success_count, total_metrics), 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ NEEDS IMPROVEMENT: Only {}/{} targets achieved</div>'.format(success_count, total_metrics), 
                           unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if vp_corr < 0.75:
                st.info("**For Vp improvement:** Consider adding more physics-based features or increasing model complexity")
            
            if vs_corr < 0.75:
                st.info("**For Vs improvement:** Vs is more sensitive to crack parameters. Refine crack density estimation")
            
            if vp_mean_error > 5 or vs_mean_error > 5:
                st.info("**For error reduction:** Check for outliers in the data and consider feature scaling")
            
            # Export full results
            st.markdown("---")
            st.markdown("### 💾 Export Complete Results")
            
            # Prepare complete results dataframe
            complete_results = st.session_state.df.copy()
            complete_results['Vp_predicted'] = results['vp_full_pred']
            complete_results['Vs_predicted'] = results['vs_full_pred']
            complete_results['Vp_error_%'] = 100 * (results['vp_full_pred'] - results['y_vp']) / results['y_vp']
            complete_results['Vs_error_%'] = 100 * (results['vs_full_pred'] - results['y_vs']) / results['y_vs']
            complete_results['VpVs_predicted'] = results['vp_full_pred'] / results['vs_full_pred']
            complete_results['VpVs_actual'] = results['y_vp'] / results['y_vs']
            
            # Add physics features if available
            if 'df_features' in results:
                for col in ['crack_density_efm', 'orientation_beta', 'Vp_efm', 'Vs_efm']:
                    if col in results['df_features'].columns:
                        complete_results[col] = results['df_features'][col].values
            
            csv_complete = complete_results.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Complete Results",
                    data=csv_complete,
                    file_name="complete_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export model summary
                model_summary = f"""
                Hybrid Physics-ML Model Summary
                ===============================
                
                Model Configuration:
                - ML Algorithm: {model_type}
                - Physics Weight: {physics_weight}
                - Test Size: {test_size}%
                - CV Folds: {cv_folds}
                
                Performance Metrics:
                - Vp R²: {r2_score(results['y_vp_test'], results['vp_test_pred']):.4f}
                - Vs R²: {r2_score(results['y_vs_test'], results['vs_test_pred']):.4f}
                - Vp Correlation: {vp_corr:.4f}
                - Vs Correlation: {vs_corr:.4f}
                - Vp MAE: {vp_mae:.0f} m/s
                - Vs MAE: {vs_mae:.0f} m/s
                
                Feature Count: {len(results['feature_cols'])}
                Samples: {len(results['y_vp'])} total, {len(results['y_vp_test'])} test
                
                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="📄 Download Model Summary",
                    data=model_summary,
                    file_name="model_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # ==========================================================================
    # SINGLE PREDICTION PAGE
    # ==========================================================================
    elif app_mode == "🔮 Single Prediction":
        st.markdown('<h2 class="sub-header">🔮 Single Sample Prediction</h2>', unsafe_allow_html=True)
        
        if st.session_state.hybrid_model is None:
            st.warning("⚠️ Please train the model first in the Run Analysis section.")
            return
        
        hybrid_model = st.session_state.hybrid_model
        matrix_props = st.session_state.get('matrix_props', {'Vp': 5500, 'Vs': 3000, 'rho': 2.71})
        crack_props = st.session_state.get('crack_props', {'aspect_ratio': 0.01, 'fluid_K': 2.25e9})
        
        st.markdown("### 📝 Enter Rock Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            porosity = st.number_input("Porosity (fraction)", 0.0, 0.5, 0.15, 0.01)
            rho = st.number_input("Density (g/cc)", 2.0, 3.0, 2.65, 0.01)
            sw = st.number_input("Water Saturation", 0.0, 1.0, 1.0, 0.1)
        
        with col2:
            vclay = st.number_input("Clay Volume (%)", 0.0, 100.0, 10.0, 1.0)
            rt = st.number_input("Resistivity (ohm-m)", 0.1, 1000.0, 100.0, 10.0)
            gr = st.number_input("Gamma Ray (API)", 0.0, 200.0, 50.0, 5.0)
        
        with col3:
            # Display current physics model parameters
            st.markdown("#### ⚛️ Physics Parameters")
            st.info(f"Matrix Vp: {matrix_props['Vp']} m/s")
            st.info(f"Matrix Vs: {matrix_props['Vs']} m/s")
            st.info(f"Aspect Ratio: {crack_props['aspect_ratio']}")
            
            # Additional parameters
            depth = st.number_input("Depth (m)", 0.0, 5000.0, 1500.0, 100.0)
            pressure = st.number_input("Pressure (MPa)", 0.0, 100.0, 30.0, 5.0)
        
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
                    st.markdown('<h3 class="section-header">🎯 Prediction Results</h3>', 
                               unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### ⚛️ Physics Model")
                        st.metric("Vp", f"{physics_vp:.0f} m/s")
                        st.metric("Vs", f"{physics_vs:.0f} m/s")
                        st.metric("Vp/Vs", f"{physics_vp/physics_vs:.2f}")
                    
                    with col2:
                        st.markdown("#### 🤖 Hybrid Model")
                        hybrid_vp_val = hybrid_vp[0]
                        hybrid_vs_val = hybrid_vs[0]
                        st.metric("Vp", f"{hybrid_vp_val:.0f} m/s", 
                                 delta=f"{(hybrid_vp_val-physics_vp):.0f} m/s")
                        st.metric("Vs", f"{hybrid_vs_val:.0f} m/s",
                                 delta=f"{(hybrid_vs_val-physics_vs):.0f} m/s")
                        st.metric("Vp/Vs", f"{hybrid_vp_val/hybrid_vs_val:.2f}")
                    
                    with col3:
                        st.markdown("#### 📊 Crack Parameters")
                        st.metric("Crack Density", f"{crack_density:.3f}")
                        st.metric("Orientation (β)", f"{beta:.3f} rad")
                        st.metric("Orientation Factor", f"{eff_props['F_normal']:.3f}")
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("### 📈 Velocity Comparison")
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    models = ['Physics', 'Hybrid']
                    vp_values = [physics_vp, hybrid_vp_val]
                    vs_values = [physics_vs, hybrid_vs_val]
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=vp_values,
                        name='Vp',
                        marker_color='blue',
                        text=[f'{v:.0f} m/s' for v in vp_values],
                        textposition='auto'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=vs_values,
                        name='Vs',
                        marker_color='green',
                        text=[f'{v:.0f} m/s' for v in vs_values],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title='Physics vs Hybrid Predictions',
                        xaxis_title='Model',
                        yaxis_title='Velocity (m/s)',
                        template='plotly_white',
                        height=500,
                        barmode='group',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, className="plotly-plot")
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Geological Interpretation")
                    
                    interpretation_cols = st.columns(2)
                    
                    with interpretation_cols[0]:
                        st.markdown("#### 🪨 Rock Properties")
                        
                        porosity_class = "Low" if porosity < 0.1 else "Moderate" if porosity < 0.2 else "High"
                        density_class = "Low" if rho < 2.6 else "High" if rho > 2.8 else "Typical"
                        clay_class = "Clean" if vclay < 15 else "Slightly shaly" if vclay < 35 else "Shaly"
                        
                        st.info(f"**Porosity**: {porosity:.3f} ({porosity_class})")
                        st.info(f"**Density**: {rho:.2f} g/cc ({density_class})")
                        st.info(f"**Clay Content**: {vclay:.1f}% ({clay_class})")
                        st.info(f"**Saturation**: {'Full' if sw > 0.9 else 'Partial' if sw > 0.5 else 'Low'}")
                    
                    with interpretation_cols[1]:
                        st.markdown("#### 🔍 Velocity Analysis")
                        
                        vp_class = "High" if hybrid_vp_val > 5000 else "Moderate" if hybrid_vp_val > 4000 else "Low"
                        vs_class = "High" if hybrid_vs_val > 2800 else "Moderate" if hybrid_vs_val > 2000 else "Low"
                        vpvs_ratio = hybrid_vp_val / hybrid_vs_val
                        vpvs_class = "High (gas?)" if vpvs_ratio > 2.0 else "Typical" if vpvs_ratio > 1.7 else "Low"
                        
                        st.info(f"**Vp**: {hybrid_vp_val:.0f} m/s ({vp_class})")
                        st.info(f"**Vs**: {hybrid_vs_val:.0f} m/s ({vs_class})")
                        st.info(f"**Vp/Vs**: {vpvs_ratio:.2f} ({vpvs_class})")
                        st.info(f"**Crack Density**: {crack_density:.3f} ({'Low' if crack_density < 0.1 else 'Moderate' if crack_density < 0.3 else 'High'})")
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    # ==========================================================================
    # ADVANCED ANALYSIS PAGE
    # ==========================================================================
    elif app_mode == "📊 Advanced Analysis":
        st.markdown('<h2 class="sub-header">📊 Advanced Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please run the analysis first.")
            return
        
        results = st.session_state.results
        
        st.markdown("### 🔬 Sensitivity Analysis")
        
        # Feature sensitivity analysis
        st.markdown("#### Feature Sensitivity")
        
        selected_feature = st.selectbox(
            "Select feature for sensitivity analysis",
            results['feature_cols']
        )
        
        if selected_feature in results['X'].columns:
            feature_values = results['X'][selected_feature].values
            feature_range = np.linspace(feature_values.min(), feature_values.max(), 50)
            
            # Create synthetic data for sensitivity analysis
            sensitivity_data = []
            
            for val in feature_range:
                # Create a baseline row
                baseline = results['X'].iloc[0:1].copy()
                baseline[selected_feature] = val
                
                # Predict using hybrid model
                if st.session_state.hybrid_model:
                    try:
                        vp_pred, vs_pred = st.session_state.hybrid_model.hybrid_predict(
                            baseline, physics_weight
                        )
                        sensitivity_data.append({
                            selected_feature: val,
                            'Vp': vp_pred[0],
                            'Vs': vs_pred[0]
                        })
                    except:
                        pass
            
            if sensitivity_data:
                sensitivity_df = pd.DataFrame(sensitivity_data)
                
                # Plot sensitivity
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=sensitivity_df[selected_feature],
                    y=sensitivity_df['Vp'],
                    mode='lines',
                    name='Vp',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=sensitivity_df[selected_feature],
                    y=sensitivity_df['Vs'],
                    mode='lines',
                    name='Vs',
                    line=dict(color='green', width=3)
                ))
                
                fig.update_layout(
                    title=f'Velocity Sensitivity to {selected_feature}',
                    xaxis_title=selected_feature,
                    yaxis_title='Velocity (m/s)',
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True, className="plotly-plot")
        
        # Residual analysis
        st.markdown("---")
        st.markdown("### 📉 Residual Analysis")
        
        vp_residuals = results['vp_test_pred'] - results['y_vp_test']
        vs_residuals = results['vs_test_pred'] - results['y_vs_test']
        
        fig_residuals = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Vp Residuals', 'Vs Residuals'),
            horizontal_spacing=0.15
        )
        
        # Vp residuals
        fig_residuals.add_trace(
            go.Scatter(
                x=results['y_vp_test'],
                y=vp_residuals,
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Vp Residuals'
            ),
            row=1, col=1
        )
        
        # Vs residuals
        fig_residuals.add_trace(
            go.Scatter(
                x=results['y_vs_test'],
                y=vs_residuals,
                mode='markers',
                marker=dict(color='green', size=8, opacity=0.6),
                name='Vs Residuals',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add zero lines
        for col in [1, 2]:
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red",
                                  row=1, col=col)
        
        fig_residuals.update_layout(
            title='Prediction Residuals',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        fig_residuals.update_xaxes(title_text='Measured Velocity (m/s)', row=1, col=1)
        fig_residuals.update_xaxes(title_text='Measured Velocity (m/s)', row=1, col=2)
        fig_residuals.update_yaxes(title_text='Residual (m/s)', row=1, col=1)
        fig_residuals.update_yaxes(title_text='Residual (m/s)', row=1, col=2)
        
        st.plotly_chart(fig_residuals, use_container_width=True, className="plotly-plot")
        
        # QQ plots for normality check
        st.markdown("#### Normality Check (Q-Q Plots)")
        
        from scipy import stats
        
        fig_qq = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Vp Residuals Q-Q Plot', 'Vs Residuals Q-Q Plot'),
            horizontal_spacing=0.15
        )
        
        # Vp Q-Q plot
        vp_sorted = np.sort(vp_residuals)
        vp_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(vp_sorted)))
        
        fig_qq.add_trace(
            go.Scatter(
                x=vp_theoretical,
                y=vp_sorted,
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Vp Residuals'
            ),
            row=1, col=1
        )
        
        # Add diagonal line
        min_val = min(vp_theoretical.min(), vp_sorted.min())
        max_val = max(vp_theoretical.max(), vp_sorted.max())
        
        fig_qq.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Normal Line',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Vs Q-Q plot
        vs_sorted = np.sort(vs_residuals)
        vs_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(vs_sorted)))
        
        fig_qq.add_trace(
            go.Scatter(
                x=vs_theoretical,
                y=vs_sorted,
                mode='markers',
                marker=dict(color='green', size=8, opacity=0.6),
                name='Vs Residuals',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig_qq.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Normal Line',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig_qq.update_layout(
            title='Q-Q Plots for Residual Normality Check',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        fig_qq.update_xaxes(title_text='Theoretical Quantiles', row=1, col=1)
        fig_qq.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
        fig_qq.update_yaxes(title_text='Sample Quantiles', row=1, col=1)
        fig_qq.update_yaxes(title_text='Sample Quantiles', row=1, col=2)
        
        st.plotly_chart(fig_qq, use_container_width=True, className="plotly-plot")

# ==============================================================================
# RUN THE APP
# ==============================================================================

if __name__ == "__main__":
    main()
