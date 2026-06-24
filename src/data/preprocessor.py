"""
Data preprocessing and normalization utilities.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Handles feature scaling, categorical encoding, and missing value imputation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Manages feature transformation and normalization for educational data.
    
    Applies consistent preprocessing across train/val/test splits:
    - Numerical features: Z-score normalization
    - Categorical demographics: Label encoding
    - Missing values: Forward-fill then median imputation
    """
    
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> 'DataPreprocessor':
        """
        Fit preprocessing transformations on training data.
        
        Args:
            df: Training dataframe
            numerical_cols: Columns to apply standard scaling
            categorical_cols: Columns to apply label encoding
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting data preprocessing transformations")
        
        if numerical_cols:
            self.numerical_scaler.fit(df[numerical_cols].fillna(df[numerical_cols].median()))
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(df[col].fillna('unknown').astype(str))
            self.categorical_encoders[col] = encoder
        
        self.fitted = True
        logger.info(f"Fitted {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Apply fitted transformations to dataframe.
        
        Args:
            df: Input dataframe
            numerical_cols: Numerical feature columns
            categorical_cols: Categorical feature columns
            
        Returns:
            Transformed dataframe with normalized features
            
        Raises:
            RuntimeError: If transform called before fit
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        if numerical_cols:
            df[numerical_cols] = self.numerical_scaler.transform(
                df[numerical_cols].fillna(df[numerical_cols].median())
            )
        
        for col in categorical_cols:
            df[col] = self.categorical_encoders[col].transform(
                df[col].fillna('unknown').astype(str)
            )
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Fit and transform in single operation."""
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df, numerical_cols, categorical_cols)