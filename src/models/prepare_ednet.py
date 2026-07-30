# prepare_ednet.py
import pandas as pd
import numpy as np

def prepare_ednet(data_path):
    """
    Prepare EdNet dataset (131M interactions, 784K students).
    No demographic attributes available - use GMM clustering.
    """
    # Load interaction logs
    df = pd.read_csv(f"{data_path}/ednet.csv")
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Create sequences of length 50
    sequences = []
    for user in df['user_id'].unique():
        user_data = df[df['user_id'] == user]
        if len(user_data) >= 50:
            seq = user_data.iloc[-50:][['content_id', 'correct']].values
            sequences.append(seq)
    
    # Apply GMM clustering for demographic proxies
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    
    # Extract features for clustering (aggregate per user)
    user_features = df.groupby('user_id').agg({
        'correct': ['mean', 'std', 'count'],
        'timestamp': ['min', 'max']
    }).fillna(0).values
    
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    gmm = GaussianMixture(n_components=8, random_state=42)
    clusters = gmm.fit_predict(user_features_scaled)
    
    # Map clusters to users
    user_cluster_map = dict(zip(df['user_id'].unique(), clusters))
    
    return sequences, user_cluster_map

# prepare_assistments.py
def prepare_assistments(data_path):
    """
    Prepare ASSISTments dataset (merged multi-version).
    No demographic attributes - use GMM clustering.
    """
    # Similar to EdNet preparation
    # Load multiple versions if using merged corpus
    pass

# prepare_oulad.py
def prepare_oulad(data_path):
    """
    Prepare OULAD dataset with explicit demographic attributes.
    Attributes: gender, region, highest_education, imd_band, age_band, disability
    """
    # Load student info and interaction logs
    student_info = pd.read_csv(f"{data_path}/studentInfo.csv")
    interactions = pd.read_csv(f"{data_path}/studentAssessment.csv")
    
    # Merge demographic attributes
    df = interactions.merge(student_info[['id_student', 'gender', 'region', 
                                           'highest_education', 'imd_band', 
                                           'age_band', 'disability']], 
                            on='id_student')
    
    # Handle missing values (mode imputation)
    for col in ['imd_band', 'highest_education']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create demographic encoding
    from sklearn.preprocessing import LabelEncoder
    demographic_columns = ['gender', 'region', 'highest_education', 
                          'imd_band', 'age_band', 'disability']
    
    encoders = {}
    for col in demographic_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Create demographic ID (combined encoding)
    df['demographic_id'] = df[[f'{col}_encoded' for col in demographic_columns]].sum(axis=1)
    
    return df, encoders