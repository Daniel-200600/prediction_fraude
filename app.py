"""
=================================================================
APPLICATION DE DÉTECTION DE FRAUDE BANCAIRE
=================================================================
Interface Streamlit美丽 pour le déploiement
Version améliorée avec design moderne et interactif

Utilisation:
    streamlit run app.py

Déploiement:
    - Local: streamlit run app.py
    - Cloud: Deploy sur Streamlit Cloud, Render, Heroku, etc.
=================================================================
"""

# ============================================================================
# IMPORTATION DES LIBRAIRIES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="🛡️ Détection de Fraude Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## 🛡️ Détection de Fraude Bancaire
        
        Application de Machine Learning pour la détection de fraudes 
        aux cartes de crédit.
        
        Développé avec Streamlit
        """
    }
)

# ============================================================================
# STYLE CSS PERSONNALISÉ
# ============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Style général */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers principaux */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Couleurs personnalisées */
    .title-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .metric-card-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fff6b7 0%, #f6416c 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Animation fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Divider styling */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_data(file_path):
    """Charge le dataset depuis un fichier CSV"""
    df = pd.read_csv(file_path)
    return df

def get_dataset_info(df):
    """Retourne les informations sur le dataset"""
    info = {
        'n_lignes': df.shape[0],
        'n_colonnes': df.shape[1],
        'colonnes': df.columns.tolist(),
        'types': df.dtypes,
        'valeurs_manquantes': df.isnull().sum(),
        'pourcentage_manquant': (df.isnull().sum() / len(df)) * 100
    }
    return info

def check_missing_values(df):
    """Vérifie les valeurs manquantes"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valeurs Manquantes': missing,
        'Pourcentage': missing_pct
    })
    return missing_df[missing_df['Valeurs Manquantes'] > 0]

def handle_missing_values(df):
    """Gère les valeurs manquantes"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def standardize_amount(df):
    """Standardise la colonne Amount"""
    scaler = RobustScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    return df, scaler

def analyze_class_distribution(df):
    """Analyse la distribution des classes"""
    class_counts = df['Class'].value_counts()
    class_percentages = (class_counts / len(df)) * 100
    distribution = pd.DataFrame({
        'Nombre': class_counts,
        'Pourcentage': class_percentages
    })
    return distribution

def handle_imbalance(X, y, method='SMOTE'):
    """Gère le déséquilibre des classes"""
    samplers = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'RandomOverSampler': RandomOverSampler(random_state=42),
        'RandomUnderSampler': RandomUnderSampler(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42)
    }
    sampler = samplers.get(method, SMOTE(random_state=42))
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def train_test_split_data(X, y, test_size=0.3, random_state=42):
    """Divise les données en ensembles d'entraînement et de test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def compare_models(X_train, y_train):
    """Compare différents modèles ML"""
    models = {
        'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Arbre de Décision': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        results[name] = {
            'mean_f1': scores.mean(),
            'std_f1': scores.std()
        }
    
    return results

def train_model(X_train, y_train, model_type='Random Forest'):
    """Entraîne un modèle"""
    models = {
        'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
        'Arbre de Décision': DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    model = models.get(model_type, RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec plusieurs métriques"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Average Precision': average_precision_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def interpret_results(metrics):
    """Interprète les résultats"""
    interpretation = {}
    
    recall = metrics['Recall']
    interpretation['recall'] = "Excellent" if recall >= 0.9 else "Très bon" if recall >= 0.8 else "Bon" if recall >= 0.7 else "À améliorer"
    
    precision = metrics['Precision']
    interpretation['precision'] = "Excellent" if precision >= 0.9 else "Très bonne" if precision >= 0.8 else "Bonne" if precision >= 0.7 else "À améliorer"
    
    f1 = metrics['F1-Score']
    interpretation['f1'] = "Excellent" if f1 >= 0.9 else "Très bon" if f1 >= 0.8 else "Bon" if f1 >= 0.7 else "À améliorer"
    
    auc = metrics['ROC-AUC']
    interpretation['auc'] = "Excellent" if auc >= 0.95 else "Très bon" if auc >= 0.9 else "Bon" if auc >= 0.8 else "À améliorer"
    
    return interpretation

# ============================================================================
# FONCTION PRINCIPALE DE TRAITEMENT
# ============================================================================

@st.cache_data(show_spinner=False)
def process_data(file_path, imbalance_method, model_choice):
    """Traite les données et entraîne le modèle"""
    
    # 1. Chargement des données
    df = load_data(file_path)
    
    # 2. Informations sur le dataset
    info = get_dataset_info(df)
    
    # 3. Valeurs manquantes
    missing_values = check_missing_values(df)
    
    # 4. Nettoyage
    df = handle_missing_values(df)
    
    # 5. Standardisation
    df, scaler = standardize_amount(df)
    
    # 6. Distribution des classes
    class_dist = analyze_class_distribution(df)
    
    # 7. Préparation des données
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # 8. Gestion du déséquilibre
    X_resampled, y_resampled = handle_imbalance(X, y, method=imbalance_method)
    
    # 9. Division train/test
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )
    
    # 10. Entraînement du modèle
    model = train_model(X_train, y_train, model_type=model_choice)
    
    # 11. Évaluation
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 12. Matrice de confusion
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # 13. Interpretation
    interpretation = interpret_results(metrics)
    
    return {
        'df': df,
        'info': info,
        'missing_values': missing_values,
        'class_dist': class_dist,
        'metrics': metrics,
        'confusion_matrix': confusion_mat,
        'interpretation': interpretation,
        'model': model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'X_train': X_train,
        'y_train': y_train
    }

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

def create_sidebar():
    """Crée la barre latérale de configuration"""
    
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: white; margin: 0;'>🛡️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Logo/title in sidebar
    st.sidebar.markdown("---")
    
    # Upload du fichier
    st.sidebar.markdown("### 📁 Fichier de données")
    uploaded_file = st.sidebar.file_uploader(
        "Télécharger votre fichier CSV",
        type=['csv'],
        help="Format attendu: Time, V1-V28, Amount, Class"
    )
    
    st.sidebar.markdown("---")
    
    # Configuration du modèle
    st.sidebar.markdown("### ⚙️ Paramètres du Modèle")
    
    imbalance_method = st.sidebar.selectbox(
        "Méthode d'équilibrage",
        ['SMOTE', 'ADASYN', 'RandomOverSampler', 'RandomUnderSampler', 'SMOTETomek', 'SMOTEENN'],
        index=0,
        help="SMOTE est recommandé pour ce type de problème"
    )
    
    model_choice = st.sidebar.selectbox(
        "Modèle ML",
        ['Random Forest', 'Régression Logistique', 'Gradient Boosting', 'Arbre de Décision', 'K-Nearest Neighbors'],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Informations sur le projet
    st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px;'>
        <h4 style='color: white; margin: 0 0 10px 0;'>📋 Informations</h4>
        <p style='color: rgba(255,255,255,0.8); font-size: 12px; margin: 0;'>
            Cette application utilise des techniques de Machine Learning 
            pour détecter les fraudes aux cartes de crédit.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return uploaded_file, imbalance_method, model_choice

# ============================================================================
# PAGE D'ACCUEIL
# ============================================================================

def show_homepage():
    """Affiche la page d'accueil"""
    
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;' class='fade-in'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>
            <span class='title-gradient'>🛡️ Détection de Fraude Bancaire</span>
        </h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Intelligence Artificielle pour protéger vos transactions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='custom-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>📊</h3>
            <h4>Analyse Interactive</h4>
            <p style='color: #666;'>Visualisez vos données en temps réel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>🤖</h3>
            <h4>ML Avancé</h4>
            <p style='color: #666;'>Plusieurs algorithmes de Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='custom-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>📈</h3>
            <h4>Métriques Détaillées</h4>
            <p style='color: #666;'>Évaluation complète des performances</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='custom-card' style='text-align: center;'>
            <h3 style='color: #667eea;'>🔮</h3>
            <h4>Prédiction</h4>
            <p style='color: #666;'>Testez avec vos propres données</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    <div class='custom-card fade-in'>
        <h2>🚀 Comment utiliser</h2>
        <ol style='font-size: 1.1rem; line-height: 2;'>
            <li><strong>Téléchargez un fichier CSV</strong> contenant vos données de transactions</li>
            <li><strong>Configurez les paramètres</strong> dans la barre latérale (méthode d'équilibrage, modèle ML)</li>
            <li><strong>Analysez les résultats</strong> automatiquement générés</li>
            <li><strong>Testez le modèle</strong> avec de nouvelles transactions</li>
        </ol>
        
        <h3 style='margin-top: 30px;'>📊 Format des données attendu</h3>
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px;'>
            <code style='font-size: 0.9rem;'>
            Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data info
    st.markdown("""
    <div class='custom-card fade-in' style='margin-top: 20px;'>
        <h3>💡 Format du fichier CSV</h3>
        <ul style='font-size: 1rem; line-height: 1.8;'>
            <li><strong>Time:</strong> Temps entre les transactions (secondes)</li>
            <li><strong>V1 à V28:</strong> Features anonymisées (résultats PCA)</li>
            <li><strong>Amount:</strong> Montant de la transaction</li>
            <li><strong>Class:</strong> Variable cible (0 = Légitime, 1 = Fraude)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RÉSULTATS DE L'ANALYSE
# ============================================================================

def show_results(results, imbalance_method, model_choice):
    """Affiche les résultats de l'analyse"""
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;' class='fade-in'>
        <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>
            <span class='title-gradient'>📊 Résultats de l'Analyse</span>
        </h1>
        <p style='font-size: 1.1rem; color: #666;'>
            Configuration: <strong>{}</strong> | Modèle: <strong>{}</strong>
        </p>
    </div>
    """.format(imbalance_method, model_choice), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Métriques principales
    st.markdown("### 🎯 Métriques de Performance")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics = results['metrics']
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}", delta=None)
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.2%}", delta=None)
    with col3:
        st.metric("Recall", f"{metrics['Recall']:.2%}", delta=None)
    with col4:
        st.metric("F1-Score", f"{metrics['F1-Score']:.2%}", delta=None)
    with col5:
        st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}", delta=None)
    with col6:
        st.metric("Avg Precision", f"{metrics['Average Precision']:.4f}", delta=None)
    
    st.markdown("---")
    
    # Deux colonnes: Distribution et Matrice de confusion
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚖️ Distribution des Classes")
        
        # Graphique interactif avec Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=['Légitime (0)', 'Fraude (1)'],
                y=results['class_dist']['Nombre'].values,
                marker_color=['#11998e', '#eb3349'],
                text=results['class_dist']['Pourcentage'].values,
                textposition='auto',
                texttemplate='%{text:.2f}%'
            )
        ])
        
        fig.update_layout(
            title="Distribution des transactions",
            xaxis_title="Classe",
            yaxis_title="Nombre de transactions",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔢 Matrice de Confusion")
        
        # Matrice de confusion interactive
        cm = results['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Prédit: Légitime', 'Prédit: Fraude'],
            y=['Réel: Légitime', 'Réel: Fraude'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=False
        ))
        
        fig.update_layout(
            title="Matrice de Confusion",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Courbes ROC et Precision-Recall
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Courbe ROC")
        
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {metrics["ROC-AUC"]:.4f})',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aléatoire',
            line=dict(color='#eb3349', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Courbe ROC",
            xaxis_title="Taux de Faux Positifs",
            yaxis_title="Taux de Vrais Positifs",
            template="plotly_white",
            height=400,
            legend=dict(x=0.7, y=0.1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Courbe Precision-Rappel")
        
        precision_curve, recall_curve, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            mode='lines',
            name=f'PR (AP = {metrics["Average Precision"]:.4f})',
            line=dict(color='#38ef7d', width=3)
        ))
        
        fig.update_layout(
            title="Courbe Precision-Rappel",
            xaxis_title="Recall (Rappel)",
            yaxis_title="Precision",
            template="plotly_white",
            height=400,
            legend=dict(x=0.7, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interprétation
    st.markdown("### 💡 Interprétation des Résultats")
    
    interpretation = results['interpretation']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Recall:** {interpretation['recall']}")
    with col2:
        st.info(f"**Precision:** {interpretation['precision']}")
    with col3:
        st.info(f"**F1-Score:** {interpretation['f1']}")
    with col4:
        st.info(f"**ROC-AUC:** {interpretation['auc']}")
    
    st.markdown("---")
    
    # Features importantes (pour Random Forest)
    if model_choice == 'Random Forest':
        st.markdown("### 🌟 Importance des Features")
        
        feature_importance = pd.DataFrame({
            'Feature': results['X_train'].columns,
            'Importance': results['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 15 Features les plus importantes",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION PRÉDICTION
# ============================================================================

def show_prediction_section(results):
    """Section pour tester le modèle avec de nouvelles données"""
    
    st.markdown("""
    <div class='custom-card fade-in'>
        <h2>🔮 Test de Prédiction</h2>
        <p>Entrez les caractéristiques d'une transaction pour vérifier si elle est frauduleuse</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire de saisie
    st.markdown("### 📝 Caractéristiques de la Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Montant (Amount)", min_value=0.0, value=100.0, step=1.0)
        
        # Features V1-V14
        v_values = []
        for i in range(1, 15):
            v = st.slider(f"V{i}", -10.0, 10.0, 0.0, 0.1)
            v_values.append(v)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Features V15-V28
        for i in range(15, 29):
            v = st.slider(f"V{i}", -10.0, 10.0, 0.0, 0.1)
            v_values.append(v)
    
    # Bouton de prédiction
    if st.button("🔍 Vérifier la Transaction", type="primary"):
        try:
            # Récupérer le scaler et les noms de features du modèle entraîné
            scaler = results.get('scaler')
            
            # Obtenir les noms de colonnes utilisées lors de l'entraînement
            feature_columns = results['X_train'].columns.tolist()
            
            # Préparer les données dans le bon ordre
            # Créer un DataFrame avec les bonnes colonnes
            input_data = {}
            
            # Ajouter V1 à V28
            for i, v in enumerate(v_values):
                input_data[f'V{i+1}'] = [v]
            
            # Ajouter Amount (non scalé)
            input_data['Amount'] = [amount]
            
            # Créer le DataFrame
            input_df = pd.DataFrame(input_data)
            
            # Ajouter Amount_Scaled (en utilisant le même scaler que pendant l'entraînement)
            if scaler is not None:
                amount_scaled = scaler.transform([[amount]])[0][0]
            else:
                # Fallback: si pas de scaler, utiliser RobustScaler par défaut
                from sklearn.preprocessing import RobustScaler
                default_scaler = RobustScaler()
                default_scaler.fit(results['X_train']['Amount'].values.reshape(-1, 1))
                amount_scaled = default_scaler.transform([[amount]])[0][0]
            
            input_df['Amount_Scaled'] = [amount_scaled]
            
            # S'assurer que les colonnes sont dans le bon ordre
            input_df = input_df[feature_columns]
            
            # Convertir en numpy array
            features_array = input_df.values
            
            # Faire la prédiction
            prediction = results['model'].predict(features_array)[0]
            proba = results['model'].predict_proba(features_array)[0]
            
            # Afficher le résultat
            st.markdown("---")
            
            if prediction == 1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                            border-radius: 15px; padding: 30px; text-align: center; color: white;'>
                    <h1 style='margin: 0;'>⚠️ ALERTE FRAUDE!</h1>
                    <p style='font-size: 1.2rem;'>Cette transaction semble être une FRAUDE</p>
                    <p style='font-size: 2rem; font-weight: bold;'>Probabilité: {:.1f}%</p>
                </div>
                """.format(proba[1] * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            border-radius: 15px; padding: 30px; text-align: center; color: white;'>
                    <h1 style='margin: 0;'>✅ Transaction Légitime</h1>
                    <p style='font-size: 1.2rem;'>Cette transaction semble être AUTHENTIFIÉE</p>
                    <p style='font-size: 2rem; font-weight: bold;'>Probabilité: {:.1f}%</p>
                </div>
                """.format(proba[0] * 100), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            st.info("Veuillez vous assurer que les données sont correctement formatées.")

# ============================================================================
# COMPARAISON DES MODÈLES
# ============================================================================

def show_model_comparison(results):
    """Affiche la comparaison des modèles"""
    
    st.markdown("""
    <div class='custom-card fade-in'>
        <h2>🏆 Comparaison des Modèles</h2>
        <p>Comparez les performances de différents algorithmes de Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner('Comparaison des modèles en cours...'):
        comparison_results = compare_models(results['X_train'], results['y_train'])
    
    # Convertir en DataFrame
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df = comparison_df.reset_index()
    comparison_df.columns = ['Modèle', 'F1-Score (Moyenne)', 'F1-Score (Écart-type)']
    
    # Graphique de comparaison
    fig = px.bar(
        comparison_df,
        x='Modèle',
        y='F1-Score (Moyenne)',
        error_y='F1-Score (Écart-type)',
        title="Comparaison des modèles (F1-Score)",
        color='F1-Score (Moyenne)',
        color_continuous_scale='Viridis',
        text='F1-Score (Moyenne)'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(template="plotly_white", height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("### 📋 Détails des performances")
    st.dataframe(comparison_df.style.format({'F1-Score (Moyenne)': '{:.4f}', 'F1-Score (Écart-type)': '{:.4f}'}))

# ============================================================================
# PIED DE PAGE
# ============================================================================

def show_footer():
    """Affiche le pied de page"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>🛡️ <strong>Détection de Fraude Bancaire</strong> - Application de Machine Learning</p>
        <p style='font-size: 0.9rem;'>Déployé avec Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application"""
    
    # Créer la sidebar
    uploaded_file, imbalance_method, model_choice = create_sidebar()
    
    # Navigation principale
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    page = st.sidebar.radio(
        "Aller vers:",
        ["Accueil", "Analyse", "Prédiction", "Comparaison"]
    )
    
    # Gestion du fichier et des résultats
    results = None
    
    if uploaded_file is not None:
        with st.spinner('🔄 Traitement des données et entraînement du modèle...'):
            results = process_data(uploaded_file, imbalance_method, model_choice)
        st.success('✅ Modèle entraîné avec succès!')
    
    # Affichage selon la page sélectionnée
    if page == "Accueil":
        if results is None:
            show_homepage()
        else:
            st.markdown("""
            <div style='text-align: center; padding: 20px;' class='fade-in'>
                <h1 style='font-size: 2.5rem;'>
                    <span class='title-gradient'>🛡️ Détection de Fraude Bancaire</span>
                </h1>
                <p style='font-size: 1.2rem; color: #666;'>
                    Votre modèle est prêt! Cliquez sur "Analyse" pour voir les résultats complets.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Aperçu des Performances")
            col1, col2, col3, col4 = st.columns(4)
            metrics = results['metrics']
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
            with col2:
                st.metric("Recall", f"{metrics['Recall']:.2%}")
            with col3:
                st.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
            with col4:
                st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
            
            st.markdown("---")
            st.info("📊 Utilisez la navigation latérale pour voir l'analyse complète, tester des prédictions ou comparer les modèles.")
    
    elif page == "Analyse":
        if results is not None:
            show_results(results, imbalance_method, model_choice)
        else:
            st.warning("⚠️ Veuillez d'abord télécharger un fichier de données dans la barre latérale.")
            show_homepage()
    
    elif page == "Prédiction":
        if results is not None:
            show_prediction_section(results)
        else:
            st.warning("⚠️ Veuillez d'abord télécharger un fichier de données dans la barre latérale.")
            show_homepage()
    
    elif page == "Comparaison":
        if results is not None:
            show_model_comparison(results)
        else:
            st.warning("⚠️ Veuillez d'abord télécharger un fichier de données dans la barre latérale.")
            show_homepage()
    
    show_footer()


if __name__ == "__main__":
    main()
