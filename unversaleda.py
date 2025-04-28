import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

class LightningEDA:
    """
    Ultra-fast EDA tool optimized for datathons.
    Prioritizes speed, memory efficiency, and actionable insights.
    Designed to run in seconds even on larger datasets.
    """
    
    def __init__(self, df, target_col=None, sample_size=10000, cat_threshold=20, verbose=True):
        """
        Initialize the LightningEDA tool.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        target_col : str, optional
            Target column name
        sample_size : int
            Sample size for analysis if dataset is large (speeds up computation)
        cat_threshold : int
            Maximum unique values to be considered categorical
        verbose : bool
            Whether to print detailed information
        """
        self.start_time = time()
        self.original_shape = df.shape
        self.sample_size = sample_size
        self.cat_threshold = cat_threshold
        self.target_col = target_col
        self.verbose = verbose
        
        # Record memory usage before sampling
        self.original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # Sample large datasets for faster computation
        if len(df) > sample_size:
            if verbose:
                print(f"Dataset has {len(df):,} rows. Sampling {sample_size:,} rows for analysis.")
            self.df = df.sample(sample_size, random_state=42)
        else:
            self.df = df.copy()
        
        # Results dictionary
        self.results = {
            'basic_info': {},
            'missing_data': {},
            'outliers': {},
            'correlations': {},
            'target_analysis': {},
            'feature_importance': {},
            'low_variance_features': [],
            'potential_issues': [],
            'quick_recommendations': [],
            'duplicate_groups': {},
            'column_profiles': {},
            'computation_time': {}
        }
        
        # Store figures for reporting
        self.figures = {}
        
        # Identify column types
        self._start_timer('column_type_detection')
        self._identify_column_types()
        self._end_timer('column_type_detection')
    
    def _start_timer(self, operation):
        """Start timer for an operation."""
        if operation not in self.results['computation_time']:
            self.results['computation_time'][operation] = {}
        self.results['computation_time'][operation]['start'] = time()
    
    def _end_timer(self, operation):
        """End timer for an operation and record duration."""
        if operation in self.results['computation_time'] and 'start' in self.results['computation_time'][operation]:
            self.results['computation_time'][operation]['duration'] = time() - self.results['computation_time'][operation]['start']
            return self.results['computation_time'][operation]['duration']
        return None
    
    def _identify_column_types(self):
        """Quickly identify column types and potential special columns."""
        # Get column types
        self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Check for datetime columns
        self.date_cols = []
        for col in self.cat_cols.copy():
            try:
                pd.to_datetime(self.df[col])
                self.date_cols.append(col)
                self.cat_cols.remove(col)
            except:
                pass
        
        # Check for categorical-like numerical columns (low cardinality)
        for col in self.num_cols.copy():
            if self.df[col].nunique() <= self.cat_threshold:
                self.cat_cols.append(col)
                self.num_cols.remove(col)
        
        # Identify potential ID or near-constant columns
        self.id_cols = []
        self.const_cols = []
        self.binary_cols = []
        
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            if n_unique == 1:
                self.const_cols.append(col)
            elif n_unique == 2:
                self.binary_cols.append(col)
            elif n_unique > 0.9 * len(self.df) and n_unique > 100:
                self.id_cols.append(col)
                if col in self.num_cols:
                    self.num_cols.remove(col)
                elif col in self.cat_cols:
                    self.cat_cols.remove(col)
        
        # Remove target from feature lists
        if self.target_col:
            if self.target_col in self.num_cols:
                self.num_cols.remove(self.target_col)
            elif self.target_col in self.cat_cols:
                self.cat_cols.remove(self.target_col)
            elif self.target_col in self.binary_cols:
                self.binary_cols.remove(self.target_col)
        
        # Store results
        self.results['basic_info']['column_types'] = {
            'numerical': len(self.num_cols),
            'categorical': len(self.cat_cols),
            'datetime': len(self.date_cols),
            'binary': len(self.binary_cols),
            'id_like': len(self.id_cols),
            'constant': len(self.const_cols),
            'all_columns': self.df.shape[1]
        }
        
        self.results['basic_info']['columns_by_type'] = {
            'numerical': self.num_cols,
            'categorical': self.cat_cols,
            'datetime': self.date_cols,
            'binary': self.binary_cols,
            'id_like': self.id_cols,
            'constant': self.const_cols
        }
        
        if self.verbose:
            print(f"Column types: {len(self.num_cols)} numerical, {len(self.cat_cols)} categorical, "
                  f"{len(self.date_cols)} datetime, {len(self.binary_cols)} binary, "
                  f"{len(self.id_cols)} ID-like, {len(self.const_cols)} constant")
    
    def basic_analysis(self):
        """Run fast basic analysis."""
        self._start_timer('basic_analysis')
        
        # Data shape and size
        self.results['basic_info']['shape'] = self.original_shape
        self.results['basic_info']['memory_usage_mb'] = self.original_memory
        
        # Missing values analysis
        missing = self.df.isnull().sum()
        missing_pct = missing / len(self.df) * 100
        missing_cols = missing[missing > 0]
        
        self.results['missing_data'] = {
            'num_missing_cols': len(missing_cols),
            'total_missing_pct': missing.sum() / (self.df.shape[0] * self.df.shape[1]) * 100,
            'cols_with_missing': {col: {'count': count, 'percent': pct} 
                                  for col, count, pct in zip(missing_cols.index, 
                                                         missing_cols.values, 
                                                         missing_pct[missing_cols.index])}
        }
        
        # Check for duplicate rows
        dupe_count = self.df.duplicated().sum()
        self.results['basic_info']['duplicates'] = {
            'count': dupe_count,
            'percentage': dupe_count / len(self.df) * 100
        }
        
        # Quick check for duplicate columns (exact matches)
        if self.df.shape[1] > 1:  # Only if we have more than one column
            # Transpose to compare columns
            T = self.df.T
            duplicate_groups = {}
            processed_cols = set()
            
            for col in self.df.columns:
                if col in processed_cols:
                    continue
                    
                # Find duplicate columns
                duplicates = []
                for other_col in self.df.columns:
                    if other_col != col and other_col not in processed_cols:
                        # Check if columns are identical
                        if self.df[col].equals(self.df[other_col]):
                            duplicates.append(other_col)
                            processed_cols.add(other_col)
                
                if duplicates:
                    duplicates = [col] + duplicates
                    duplicate_groups[col] = duplicates
                    processed_cols.add(col)
            
            if duplicate_groups:
                self.results['duplicate_groups'] = duplicate_groups
                self.results['potential_issues'].append({
                    'issue': 'duplicate_columns',
                    'description': f"Found {sum(len(group) for group in duplicate_groups.values()) - len(duplicate_groups)} duplicate columns",
                    'severity': 'medium'
                })
                
                self.results['quick_recommendations'].append({
                    'recommendation': 'remove_duplicates',
                    'description': 'Remove duplicate columns to reduce dimensionality'
                })
        
        # Flag potential issues
        if self.results['missing_data']['num_missing_cols'] > 0:
            self.results['potential_issues'].append({
                'issue': 'missing_values',
                'description': f"{self.results['missing_data']['num_missing_cols']} columns have missing values",
                'severity': 'high' if self.results['missing_data']['total_missing_pct'] > 5 else 'medium'
            })
        
        if dupe_count > 0:
            self.results['potential_issues'].append({
                'issue': 'duplicate_rows',
                'description': f"{dupe_count} duplicate rows detected ({dupe_count/len(self.df)*100:.1f}%)",
                'severity': 'high' if dupe_count > 0.05 * len(self.df) else 'low'
            })
        
        if self.const_cols:
            self.results['potential_issues'].append({
                'issue': 'constant_columns',
                'description': f"Found {len(self.const_cols)} constant columns with only one value",
                'severity': 'medium'
            })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'remove_constants',
                'description': 'Remove constant columns as they provide no information'
            })
        
        if self.id_cols:
            self.results['potential_issues'].append({
                'issue': 'id_columns',
                'description': f"Found {len(self.id_cols)} potential ID columns with high cardinality",
                'severity': 'low'
            })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'handle_ids',
                'description': 'Be cautious with ID-like columns in modeling (high cardinality)'
            })
        
        # Print summary
        if self.verbose:
            print(f"\nBasic Info:")
            print(f"- Dataset shape: {self.original_shape[0]:,} rows × {self.original_shape[1]:,} columns")
            print(f"- Memory usage: {self.results['basic_info']['memory_usage_mb']:.2f} MB")
            print(f"- Missing data: {self.results['missing_data']['num_missing_cols']} columns with missing values")
            print(f"- Duplicates: {dupe_count} rows ({dupe_count/len(self.df)*100:.1f}%)")
            
            if self.const_cols:
                print(f"- Constant columns: {len(self.const_cols)}")
            
            if self.id_cols:
                print(f"- ID-like columns: {len(self.id_cols)}")
            
            if 'duplicate_groups' in self.results:
                total_dupes = sum(len(group) for group in self.results['duplicate_groups'].values()) - len(self.results['duplicate_groups'])
                print(f"- Duplicate columns: {total_dupes}")
        
        self._end_timer('basic_analysis')
        return self
    
    def analyze_target(self):
        """Analyze target variable quickly."""
        if not self.target_col or self.target_col not in self.df.columns:
            if self.verbose:
                print("No target column specified or found.")
            return self
        
        self._start_timer('target_analysis')
        
        target = self.df[self.target_col]
        
        # Detect target type
        if pd.api.types.is_numeric_dtype(target) and target.nunique() > self.cat_threshold:
            target_type = 'numerical'
            
            # Quick numerical target analysis
            target_stats = {
                'type': 'numerical',
                'min': target.min(),
                'max': target.max(),
                'mean': target.mean(),
                'median': target.median(),
                'std': target.std(),
                'skew': target.skew(),
                'kurtosis': target.kurtosis(),
                'unique_values': target.nunique(),
                'zeros_count': (target == 0).sum(),
                'zeros_percentage': (target == 0).sum() / len(target) * 100,
                'negatives_count': (target < 0).sum(),
                'negatives_percentage': (target < 0).sum() / len(target) * 100
            }
            
            # Check for skewed target
            if abs(target.skew()) > 1:
                self.results['potential_issues'].append({
                    'issue': 'skewed_target',
                    'description': f"Target variable is skewed (skewness = {target.skew():.2f})",
                    'severity': 'medium'
                })
                
                self.results['quick_recommendations'].append({
                    'recommendation': 'target_transformation',
                    'description': 'Consider log or Box-Cox transformation for the skewed target'
                })
            
            # Check for heavy-tailed distribution
            if target.kurtosis() > 3:
                self.results['potential_issues'].append({
                    'issue': 'heavy_tailed_target',
                    'description': f"Target has heavy tails (kurtosis = {target.kurtosis():.2f})",
                    'severity': 'low'
                })
            
            # Check for outliers in target
            Q1 = target.quantile(0.25)
            Q3 = target.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((target < lower) | (target > upper)).sum()
            target_stats['outliers_count'] = outliers
            target_stats['outliers_percentage'] = outliers / len(target) * 100
            
            if target_stats['outliers_percentage'] > 5:
                self.results['potential_issues'].append({
                    'issue': 'target_outliers',
                    'description': f"Target has {target_stats['outliers_percentage']:.1f}% outliers",
                    'severity': 'medium'
                })
                
                self.results['quick_recommendations'].append({
                    'recommendation': 'handle_target_outliers',
                    'description': 'Consider robust regression or target transformation'
                })
        else:
            target_type = 'categorical'
            
            # Quick categorical target analysis
            value_counts = target.value_counts()
            class_counts = value_counts.to_dict()
            
            target_stats = {
                'type': 'categorical',
                'unique_classes': target.nunique(),
                'class_counts': class_counts,
                'most_common': value_counts.index[0],
                'most_common_count': int(value_counts.iloc[0]),
                'most_common_percentage': value_counts.iloc[0] / len(target) * 100
            }
            
            # Calculate entropy to measure class distribution uniformity
            p = value_counts / len(target)
            entropy = -(p * np.log2(p)).sum()
            max_entropy = np.log2(target.nunique())
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            target_stats['entropy'] = entropy
            target_stats['normalized_entropy'] = normalized_entropy
            
            # Check for class imbalance
            if target.nunique() > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                target_stats['imbalance_ratio'] = imbalance_ratio
                
                if imbalance_ratio > 10:
                    self.results['potential_issues'].append({
                        'issue': 'severe_class_imbalance',
                        'description': f"Severe class imbalance detected (ratio = {imbalance_ratio:.1f})",
                        'severity': 'high'
                    })
                    
                    self.results['quick_recommendations'].append({
                        'recommendation': 'handle_imbalance',
                        'description': 'Use class weights, SMOTE, or stratified sampling to handle class imbalance'
                    })
                elif imbalance_ratio > 3:
                    self.results['potential_issues'].append({
                        'issue': 'class_imbalance',
                        'description': f"Class imbalance detected (ratio = {imbalance_ratio:.1f})",
                        'severity': 'medium'
                    })
            
            # Check if binary classification
            if target.nunique() == 2:
                target_stats['is_binary'] = True
                target_stats['positive_class'] = value_counts.index[0]
                target_stats['negative_class'] = value_counts.index[1]
                target_stats['positive_rate'] = value_counts.iloc[0] / len(target) * 100
                
                self.results['quick_recommendations'].append({
                    'recommendation': 'binary_classification',
                    'description': 'Use binary classification metrics (AUC, F1, etc.)'
                })
            else:
                target_stats['is_binary'] = False
                
                if target.nunique() > 10:
                    self.results['potential_issues'].append({
                        'issue': 'high_cardinality_target',
                        'description': f"Target has high cardinality ({target.nunique()} classes)",
                        'severity': 'medium'
                    })
        
        self.results['target_analysis'] = target_stats
        
        # Print target info
        if self.verbose:
            print(f"\nTarget Variable ({self.target_col}):")
            print(f"- Type: {target_type}")
            
            if target_type == 'numerical':
                print(f"- Range: {target_stats['min']} to {target_stats['max']}")
                print(f"- Mean: {target_stats['mean']:.2f}, Median: {target_stats['median']:.2f}")
                print(f"- Skewness: {target_stats['skew']:.2f}, Kurtosis: {target_stats['kurtosis']:.2f}")
                print(f"- Outliers: {target_stats['outliers_percentage']:.1f}%")
            else:
                print(f"- Classes: {target_stats['unique_classes']}")
                print(f"- Most common: {target_stats['most_common']} ({target_stats['most_common_percentage']:.1f}%)")
                if 'imbalance_ratio' in target_stats:
                    print(f"- Class imbalance ratio: {target_stats['imbalance_ratio']:.1f}")
                if target_stats['is_binary']:
                    print(f"- Binary classification with {target_stats['positive_rate']:.1f}% positive rate")
        
        self._end_timer('target_analysis')
        return self
    
    def analyze_numerical(self):
        """Quick analysis of numerical columns."""
        if not self.num_cols:
            if self.verbose:
                print("No numerical columns to analyze.")
            return self
        
        self._start_timer('numerical_analysis')
        
        # Calculate basic stats for all numerical columns
        num_profiles = {}
        skewed_cols = []
        low_variance_cols = []
        potential_log_transform = []
        
        for col in self.num_cols:
            # Skip if all values are missing
            if self.df[col].isna().all():
                continue
                
            # Calculate quick stats
            stats = {
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'variance': self.df[col].var(),
                'skew': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'zeros_count': (self.df[col] == 0).sum(),
                'zeros_percentage': (self.df[col] == 0).sum() / len(self.df) * 100,
                'missing_count': self.df[col].isna().sum(),
                'missing_percentage': self.df[col].isna().sum() / len(self.df) * 100
            }
            
            # Calculate coefficient of variation (normalized measure of dispersion)
            if stats['mean'] != 0:
                stats['cv'] = stats['std'] / abs(stats['mean'])
            else:
                stats['cv'] = np.nan
            
            # Check if the column has low variance (using coefficient of variation)
            if not np.isnan(stats['cv']) and stats['cv'] < 0.1:
                low_variance_cols.append(col)
            
            # Check for skewness
            if abs(stats['skew']) > 1.5:
                skewed_cols.append(col)
                
                # Check if suitable for log transform (all positive with right skew)
                if stats['min'] > 0 and stats['skew'] > 1:
                    potential_log_transform.append(col)
            
            # Quick check for outliers using IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            
            stats['outliers_count'] = outliers
            stats['outliers_percentage'] = outliers / len(self.df) * 100
            
            # Store column profile
            num_profiles[col] = stats
        
        # Store results
        self.results['column_profiles']['numerical'] = num_profiles
        self.results['low_variance_features'] = low_variance_cols
        
        # Add data quality issues
        if skewed_cols:
            self.results['potential_issues'].append({
                'issue': 'skewed_features',
                'description': f"{len(skewed_cols)} numerical features are highly skewed",
                'columns': skewed_cols,
                'severity': 'medium'
            })
            
            if potential_log_transform:
                self.results['quick_recommendations'].append({
                    'recommendation': 'log_transform',
                    'description': 'Apply log transformation to right-skewed positive features',
                    'columns': potential_log_transform
                })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'transform_skewed',
                'description': 'Apply appropriate transformations to skewed numerical features'
            })
        
        if low_variance_cols:
            self.results['potential_issues'].append({
                'issue': 'low_variance',
                'description': f"{len(low_variance_cols)} numerical features have very low variance",
                'columns': low_variance_cols,
                'severity': 'medium'
            })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'check_low_variance',
                'description': 'Consider removing or carefully examining low variance features'
            })
        
        # Check for outliers in multiple columns
        cols_with_outliers = {col: stats['outliers_percentage'] 
                             for col, stats in num_profiles.items() 
                             if stats['outliers_percentage'] > 5}
        
        if cols_with_outliers:
            self.results['outliers'] = cols_with_outliers
            self.results['potential_issues'].append({
                'issue': 'outliers',
                'description': f"{len(cols_with_outliers)} numerical features have significant outliers",
                'columns': list(cols_with_outliers.keys()),
                'severity': 'medium'
            })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'handle_outliers',
                'description': 'Consider capping outliers or using robust models'
            })
        
        if self.verbose:
            print(f"\nNumerical Features:")
            print(f"- {len(skewed_cols)} highly skewed numerical features")
            print(f"- {len(cols_with_outliers) if cols_with_outliers else 0} features with significant outliers")
            print(f"- {len(low_variance_cols)} features with low variance")
        
        self._end_timer('numerical_analysis')
        return self
    
    def analyze_categorical(self):
        """Quick analysis of categorical columns."""
        if not self.cat_cols:
            if self.verbose:
                print("No categorical columns to analyze.")
            return self
        
        self._start_timer('categorical_analysis')
        
        # Check cardinality and distribution
        cat_profiles = {}
        high_card_cols = {}
        imbalanced_cats = []
        
        for col in self.cat_cols:
            # Skip if all values are missing
            if self.df[col].isna().all():
                continue
                
            # Get value counts
            value_counts = self.df[col].value_counts()
            
            # Basic stats
            stats = {
                'unique_values': self.df[col].nunique(),
                'missing_count': self.df[col].isna().sum(),
                'missing_percentage': self.df[col].isna().sum() / len(self.df) * 100,
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'most_common_percentage': value_counts.iloc[0] / len(self.df) * 100 if not value_counts.empty else 0
            }
            
            # Get top categories for reporting
            top_n = min(5, len(value_counts))
            stats['top_categories'] = {
                str(val): {
                    'count': int(count),
                    'percentage': count / len(self.df) * 100
                }
                for val, count in zip(value_counts.index[:top_n], value_counts.values[:top_n])
            }
            
            # Calculate entropy to measure distribution uniformity
            if len(value_counts) > 1:
                p = value_counts / len(self.df)
                entropy = -(p * np.log2(p)).sum()
                max_entropy = np.log2(self.df[col].nunique())
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                stats['entropy'] = entropy
                stats['normalized_entropy'] = normalized_entropy
                
                # Calculate imbalance ratio
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                stats['imbalance_ratio'] = imbalance_ratio
                
                if imbalance_ratio > 10 and len(value_counts) > 1:
                    imbalanced_cats.append(col)
            
            # Store column profile
            cat_profiles[col] = stats
            
            # Check for high cardinality
            if stats['unique_values'] > 20:
                high_card_cols[col] = stats['unique_values']
        
        # Store results
        self.results['column_profiles']['categorical'] = cat_profiles
        
        if high_card_cols:
            self.results['potential_issues'].append({
                'issue': 'high_cardinality',
                'description': f"{len(high_card_cols)} categorical features have high cardinality",
                'columns': list(high_card_cols.keys()),
                'severity': 'medium'
            })
            
            self.results['quick_recommendations'].append({
                'recommendation': 'encode_high_cardinality',
                'description': 'Use target encoding, hash encoding, or embedding for high-cardinality categoricals'
            })
        
        if imbalanced_cats:
            self.results['potential_issues'].append({
                'issue': 'imbalanced_categories',
                'description': f"{len(imbalanced_cats)} categorical features have severe class imbalance",
                'columns': imbalanced_cats,
                'severity': 'low'
            })
        
        if self.verbose:
            print(f"\nCategorical Features:")
            print(f"- {len(high_card_cols)} high-cardinality categorical features")
            print(f"- {len(imbalanced_cats)} features with imbalanced categories")
        
        self._end_timer('categorical_analysis')
        return self
    
    def analyze_correlations(self):
        """Fast correlation analysis focusing only on the most important relationships."""
        if len(self.num_cols) < 2:
            return self
        
        self._start_timer('correlation_analysis')
        
        # Select a subset of numerical columns for speed if there are too many
        if len(self.num_cols) > 30:
            # Prioritize columns with fewer missing values
            missing_counts = {col: self.df[col].isnull().sum() for col in self.num_cols}
            analysis_cols = sorted(missing_counts.keys(), key=lambda x: missing_counts[x])[:30]
            if self.verbose:
                print(f"Analyzing correlations for 30 out of {len(self.num_cols)} numerical features")
        else:
            analysis_cols = self.num_cols
        
        # Calculate correlation matrix efficiently
        corr_matrix = self.df[analysis_cols].corr(method='pearson')
        
        # Find highly correlated pairs (using a higher threshold for speed)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:  # Higher threshold for datathon relevance
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        self.results['correlations']['high_corr_pairs'] = high_corr_pairs
        
        if len(high_corr_pairs) > 0:
            self.results['potential_issues'].append({
                'issue': 'multicollinearity',
                'description': f"{len(high_corr_pairs)} pairs of highly correlated features found",
                'severity': 'medium' if len(high_corr_pairs) > 5 else 'low'
            })
            
            if len(high_corr_pairs) > 5:
                self.results['quick_recommendations'].append({
                    'recommendation': 'handle_multicollinearity',
                    'description': 'Remove redundant features or use regularization/dimensionality reduction'
                })
        
        # If target is numerical, find top correlated features
        if self.target_col and self.target_col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[self.target_col]):
            target_corrs = self.df[analysis_cols].corrwith(self.df[self.target_col])
            sorted_corrs = target_corrs.abs().sort_values(ascending=False)
            top_corrs = sorted_corrs.head(10).to_dict()
            
            self.results['correlations']['target_correlations'] = {
                'top_positives': target_corrs.sort_values(ascending=False).head(5).to_dict(),
                'top_negatives': target_corrs.sort_values().head(5).to_dict()
            }
            
            if self.verbose:
                print(f"\nCorrelation Analysis:")
                print(f"- {len(high_corr_pairs)} pairs of highly correlated features")
                
                if 'target_correlations' in self.results['correlations']:
                    print("\nTop correlated features with target:")
                    for feat, corr in list(self.results['correlations']['target_correlations']['top_positives'].items())[:3]:
                        print(f"- {feat}: {corr:.3f}")
        else:
            if self.verbose:
                print(f"\nCorrelation Analysis:")
                print(f"- {len(high_corr_pairs)} pairs of highly correlated features")
        
        self._end_timer('correlation_analysis')
        return self
    
    def quick_feature_importance(self):
        """Estimate feature importance using a lightweight model."""
        if not self.target_col or self.target_col not in self.df.columns:
            return self
            
        try:
            self._start_timer('feature_importance')
            
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare the data - limit to 100 features max for speed
            if len(self.num_cols) + len(self.cat_cols) > 100:
                # Prioritize numerical features for speed
                if len(self.num_cols) > 50:
                    selected_num = self.num_cols[:50]
                else:
                    selected_num = self.num_cols
                
                remaining = 100 - len(selected_num)
                selected_cat = self.cat_cols[:remaining]
                
                if self.verbose:
                    print(f"Using {len(selected_num)} numerical and {len(selected_cat)} categorical features for importance calculation")
                
                X = self.df[selected_num + selected_cat].copy()
            else:
                X = self.df[self.num_cols + self.cat_cols].copy()
            
            y = self.df[self.target_col]
            
            # Handle missing values with simple imputation
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "MISSING")
            
            # Handle categorical features with simple label encoding
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Use a very lightweight Random Forest for speed
            if pd.api.types.is_numeric_dtype(y):
                model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
            
            # Fit model
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Sort and store
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.results['feature_importance'] = importance_df.set_index('feature')['importance'].to_dict()
            
            # Identify top features
            top_features = list(importance_df['feature'].head(10))
            
            # Recommend focusing on important features
            self.results['quick_recommendations'].append({
                'recommendation': 'focus_on_important',
                'description': 'Prioritize top features in modeling and feature engineering',
                'features': top_features
            })
            
            if self.verbose:
                print("\nTop 5 Important Features:")
                for feat, imp in list(self.results['feature_importance'].items())[:5]:
                    print(f"- {feat}: {imp:.4f}")
            
            self._end_timer('feature_importance')
            
        except Exception as e:
            if self.verbose:
                print(f"Could not compute feature importance: {str(e)}")
        
        return self
    
    def plot_key_insights(self, n_plots=6):
        """Create minimal yet informative visualizations."""
        self._start_timer('plotting')
        
        plt.style.use('ggplot')
        
        # Create a figure with subplots
        n_plots = min(n_plots, 6)  # Limit to 6 plots for speed
        fig = plt.figure(figsize=(15, 12))
        
        plot_idx = 1
        
        # 1. Plot missing data if significant
        if self.results['missing_data']['num_missing_cols'] > 0:
            ax = fig.add_subplot(3, 2, plot_idx)
            
            missing_cols = list(self.results['missing_data']['cols_with_missing'].keys())
            missing_pcts = [info['percent'] for info in self.results['missing_data']['cols_with_missing'].values()]
            
            # Limit to top 10 columns with missing values
            if len(missing_cols) > 10:
                missing_df = pd.DataFrame({
                    'column': missing_cols,
                    'percent': missing_pcts
                }).sort_values('percent', ascending=False).head(10)
                missing_cols = missing_df['column'].tolist()
                missing_pcts = missing_df['percent'].tolist()
            
            ax.barh(missing_cols, missing_pcts)
            ax.set_title('Columns with Missing Values (%)')
            ax.set_xlabel('Percentage Missing')
            plot_idx += 1
        
        # 2. Plot target distribution
        if self.target_col and 'target_analysis' in self.results:
            ax = fig.add_subplot(3, 2, plot_idx)
            
            if self.results['target_analysis']['type'] == 'categorical':
                value_counts = self.df[self.target_col].value_counts()
                # Limit to top 15 categories for readability
                if len(value_counts) > 15:
                    other_count = value_counts.iloc[15:].sum()
                    value_counts = value_counts.iloc[:15]
                    value_counts['Other'] = other_count
                
                colors = plt.cm.tab20(np.linspace(0, 1, len(value_counts)))
                ax.bar(value_counts.index.astype(str), value_counts.values, color=colors)
                ax.set_title(f'Target Distribution: {self.target_col}')
                if len(value_counts) > 5:
                    ax.tick_params(axis='x', rotation=45)
            else:
                # For numerical targets, use a histogram with KDE
                try:
                    sns.histplot(self.df[self.target_col].dropna(), kde=True, ax=ax)
                except:
                    ax.hist(self.df[self.target_col].dropna(), bins=30)
                ax.set_title(f'Target Distribution: {self.target_col}')
            
            plot_idx += 1
        
        # 3. Plot feature importance if available
        if 'feature_importance' in self.results and self.results['feature_importance']:
            ax = fig.add_subplot(3, 2, plot_idx)
            
            importance_df = pd.DataFrame({
                'feature': list(self.results['feature_importance'].keys()),
                'importance': list(self.results['feature_importance'].values())
            }).sort_values('importance', ascending=False).head(10)
            
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(importance_df)))
            ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
            ax.set_title('Top 10 Feature Importance')
            plot_idx += 1
        
        # 4. Plot correlation heatmap if available
        if 'high_corr_pairs' in self.results['correlations'] and self.results['correlations']['high_corr_pairs']:
            ax = fig.add_subplot(3, 2, plot_idx)
            
            # Get features involved in high correlations
            corr_features = set()
            for pair in self.results['correlations']['high_corr_pairs']:
                corr_features.add(pair['feature_1'])
                corr_features.add(pair['feature_2'])
            
            corr_features = list(corr_features)
            if len(corr_features) > 10:
                corr_features = corr_features[:10]  # Limit to 10 features
            
            # Create correlation matrix for those features
            corr_matrix = self.df[corr_features].corr()
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=0.5)
            ax.set_title('Correlation Heatmap (Most Correlated Features)')
            plot_idx += 1
        
        # 5. Plot outlier distributions if any
        if 'outliers' in self.results and self.results['outliers']:
            ax = fig.add_subplot(3, 2, plot_idx)
            
            outlier_cols = list(self.results['outliers'].keys())
            outlier_pcts = list(self.results['outliers'].values())
            
            # Limit to top 10 for readability
            if len(outlier_cols) > 10:
                outlier_df = pd.DataFrame({
                    'column': outlier_cols,
                    'percent': outlier_pcts
                }).sort_values('percent', ascending=False).head(10)
                outlier_cols = outlier_df['column'].tolist()
                outlier_pcts = outlier_df['percent'].tolist()
            
            colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(outlier_cols)))
            ax.barh(outlier_cols, outlier_pcts, color=colors)
            ax.set_title('Features with Outliers (%)')
            ax.set_xlabel('Percentage of Outliers')
            plot_idx += 1
        
        # 6. Plot target correlation with features
        if 'target_correlations' in self.results.get('correlations', {}):
            ax = fig.add_subplot(3, 2, plot_idx)
            
            # Combine positive and negative correlations
            all_corrs = {**self.results['correlations']['target_correlations']['top_positives'],
                        **self.results['correlations']['target_correlations']['top_negatives']}
            
            corr_df = pd.DataFrame({
                'feature': list(all_corrs.keys()),
                'correlation': list(all_corrs.values())
            }).sort_values('correlation')
            
            colors = ['r' if c < 0 else 'g' for c in corr_df['correlation']]
            ax.barh(corr_df['feature'], corr_df['correlation'], color=colors)
            ax.set_title(f'Feature Correlations with {self.target_col}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plot_idx += 1
        
        # Adjust layout and show
        plt.tight_layout()
        
        # Save the figure
        self.figures['key_insights'] = fig
        
        self._end_timer('plotting')
        return self
    
    def get_model_recommendations(self):
        """Generate quick model recommendations based on data characteristics."""
        self._start_timer('model_recommendations')
        
        recommendations = []
        
        # Check if we have a target and features
        if not self.target_col or self.target_col not in self.df.columns:
            return ["No target column specified for model recommendations."]
        
        # Determine problem type
        is_classification = False
        if 'target_analysis' in self.results:
            is_classification = self.results['target_analysis']['type'] == 'categorical'
        else:
            is_classification = not pd.api.types.is_numeric_dtype(self.df[self.target_col])
            
        problem_type = "classification" if is_classification else "regression"
        
        # Base recommendation based on problem type
        if problem_type == "classification":
            recommendations.append("Problem type: Classification")
            
            # Check if binary or multiclass
            is_binary = False
            if 'target_analysis' in self.results:
                is_binary = self.results['target_analysis'].get('is_binary', False)
            else:
                is_binary = self.df[self.target_col].nunique() == 2
            
            if is_binary:
                recommendations.append("• Binary classification problem")
            else:
                n_classes = self.df[self.target_col].nunique()
                recommendations.append(f"• Multiclass classification problem ({n_classes} classes)")
            
            # Check class imbalance
            if 'target_analysis' in self.results and 'imbalance_ratio' in self.results['target_analysis']:
                imbalance_ratio = self.results['target_analysis']['imbalance_ratio']
                
                if imbalance_ratio > 10:
                    recommendations.append("• Severe class imbalance detected - consider:")
                    recommendations.append("  - Tree-based models with class weights (XGBoost, LightGBM)")
                    recommendations.append("  - SMOTE or other resampling techniques")
                    recommendations.append("  - Focal loss or other imbalance-handling approaches")
                elif imbalance_ratio > 3:
                    recommendations.append("• Moderate class imbalance detected - consider:")
                    recommendations.append("  - Class weights or balanced sampling")
                    recommendations.append("  - Evaluation metrics that handle imbalance")
                else:
                    recommendations.append("• Standard classification approaches should work well:")
                    recommendations.append("  - Gradient boosting (XGBoost, LightGBM, CatBoost)")
                    recommendations.append("  - Random Forest")
                    recommendations.append("  - Logistic Regression (with regularization)")
            
            # Check high cardinality in target
            if 'target_analysis' in self.results and self.results['target_analysis'].get('unique_classes', 0) > 10:
                recommendations.append("• High number of target classes - consider:")
                recommendations.append("  - Hierarchical classification")
                recommendations.append("  - Grouping rare classes")
                recommendations.append("  - Using specialized multiclass methods")
            
            # Evaluation metric suggestions
            if is_binary:
                recommendations.append("• Recommended evaluation metrics:")
                recommendations.append("  - AUC-ROC (standard for binary classification)")
                recommendations.append("  - Average Precision / PR-AUC (for imbalanced data)")
                recommendations.append("  - F1-score (balances precision and recall)")
            else:
                if 'target_analysis' in self.results and 'imbalance_ratio' in self.results['target_analysis'] and self.results['target_analysis']['imbalance_ratio'] > 3:
                    recommendations.append("• Recommended evaluation metrics:")
                    recommendations.append("  - Macro/Weighted F1-score (for imbalanced multiclass)")
                    recommendations.append("  - Macro AUC-ROC (one-vs-rest)")
                    recommendations.append("  - Balanced accuracy")
                else:
                    recommendations.append("• Recommended evaluation metrics:")
                    recommendations.append("  - Accuracy (if classes are balanced)")
                    recommendations.append("  - Macro F1-score (for general performance)")
                    recommendations.append("  - Confusion matrix (for detailed error analysis)")
        else:
            recommendations.append("Problem type: Regression")
            
            # Check target skewness
            if 'target_analysis' in self.results and 'skew' in self.results['target_analysis']:
                if abs(self.results['target_analysis']['skew']) > 1:
                    recommendations.append("• Skewed target detected - consider:")
                    if self.results['target_analysis'].get('min', 0) >= 0:
                        recommendations.append("  - Log transform the target (appropriate for positive data)")
                    recommendations.append("  - Box-Cox or Yeo-Johnson transformation")
                    recommendations.append("  - Tree-based models that handle non-normal distributions")
            
            # Check for outliers in target
            if 'target_analysis' in self.results and 'outliers_percentage' in self.results['target_analysis']:
                if self.results['target_analysis']['outliers_percentage'] > 5:
                    recommendations.append("• Target has significant outliers - consider:")
                    recommendations.append("  - Robust regression methods")
                    recommendations.append("  - Quantile regression")
                    recommendations.append("  - Winsorizing/capping extreme values")
            
            recommendations.append("• Recommended models:")
            recommendations.append("  - Gradient boosting (XGBoost, LightGBM, CatBoost)")
            recommendations.append("  - Random Forest")
            recommendations.append("  - Linear models with regularization (Ridge/Lasso)")
            
            recommendations.append("• Recommended evaluation metrics:")
            
            # Suggest appropriate metrics based on target distribution
            if 'target_analysis' in self.results and 'skew' in self.results['target_analysis']:
                if abs(self.results['target_analysis']['skew']) > 1:
                    recommendations.append("  - MAPE or RMSLE (for skewed targets)")
                    recommendations.append("  - Quantile-based metrics (for robustness)")
                elif 'outliers_percentage' in self.results['target_analysis'] and self.results['target_analysis']['outliers_percentage'] > 5:
                    recommendations.append("  - MAE (more robust to outliers)")
                    recommendations.append("  - Huber loss (balanced between MSE and MAE)")
                else:
                    recommendations.append("  - RMSE (root mean squared error)")
                    recommendations.append("  - R² (coefficient of determination)")
            else:
                recommendations.append("  - RMSE (root mean squared error)")
                recommendations.append("  - MAE (mean absolute error)")
                recommendations.append("  - R² (coefficient of determination)")
        
        # Feature engineering recommendations
        recommendations.append("\nFeature engineering recommendations:")
        
        # Categorical features
        if self.cat_cols:
            high_card_cols = {col: self.df[col].nunique() for col in self.cat_cols if self.df[col].nunique() > 10}
            if high_card_cols:
                recommendations.append("• High-cardinality categorical features:")
                if problem_type == "classification":
                    recommendations.append("  - Target encoding (supervised, watch for leakage)")
                recommendations.append("  - Count/frequency encoding (simple but effective)")
                recommendations.append("  - One-hot encoding (only for low-cardinality features)")
                recommendations.append("  - Hash encoding (for extremely high cardinality)")
            else:
                recommendations.append("• Categorical features:")
                recommendations.append("  - One-hot encoding (recommended for most models)")
                recommendations.append("  - Label encoding (only for tree-based models)")
        
        # Numerical features
        num_issues = []
        if 'outliers' in self.results and self.results['outliers']:
            num_issues.append("outliers")
        if hasattr(self, 'results') and 'potential_issues' in self.results:
            for issue in self.results['potential_issues']:
                if issue['issue'] == 'skewed_features':
                    num_issues.append("skewness")
        
        if num_issues:
            recommendations.append("• Numerical features preprocessing:")
            if "outliers" in num_issues:
                recommendations.append("  - Outlier capping/Winsorization (limit extreme values)")
            if "skewness" in num_issues:
                recommendations.append("  - Log transformation for positively skewed features")
                recommendations.append("  - Box-Cox transformation for non-normal distributions")
            recommendations.append("  - Standardization/normalization (for non-tree models)")
        
        # Missing values
        if self.results['missing_data']['num_missing_cols'] > 0:
            recommendations.append("• Missing value handling:")
            if self.results['missing_data']['total_missing_pct'] < 5:
                recommendations.append("  - Simple imputation (median for numerical, mode for categorical)")
            else:
                recommendations.append("  - Models that handle missing values natively (XGBoost, LightGBM)")
                recommendations.append("  - Advanced imputation (KNN, MICE, or iterative imputation)")
                recommendations.append("  - Missing value indicators for features with >10% missing")
            
            high_missing = [col for col, info in self.results['missing_data']['cols_with_missing'].items() 
                           if info['percent'] > 30]
            if high_missing:
                recommendations.append(f"  - Consider dropping features with >30% missing ({len(high_missing)} features)")
        
        # Feature selection/reduction
        if len(self.num_cols) + len(self.cat_cols) > 20:
            recommendations.append("• Feature selection:")
            if 'feature_importance' in self.results and self.results['feature_importance']:
                top_features = list(self.results['feature_importance'].keys())[:10]
                recommendations.append(f"  - Focus on top important features: {', '.join(top_features[:3])}...")
            
            recommendations.append("  - Remove low variance and redundant features")
            if 'high_corr_pairs' in self.results['correlations'] and self.results['correlations']['high_corr_pairs']:
                recommendations.append("  - Remove one feature from each highly correlated pair")
        
        # Add cross-validation strategy
        recommendations.append("\nValidation strategy:")
        
        # Different recommendations based on problem characteristics
        if self.date_cols:
            recommendations.append("• Time-based features detected - consider:")
            recommendations.append("  - Time-based split instead of random cross-validation")
            recommendations.append("  - Forward-chaining or rolling-window validation")
        elif is_classification and 'target_analysis' in self.results and 'imbalance_ratio' in self.results['target_analysis']:
            if self.results['target_analysis']['imbalance_ratio'] > 3:
                recommendations.append("• Class imbalance detected - use:")
                recommendations.append("  - Stratified K-fold cross-validation")
                recommendations.append("  - Stratified train/test split")
        else:
            recommendations.append("• Standard validation approaches:")
            recommendations.append("  - 5-fold cross-validation")
            recommendations.append("  - Consistent random seed for reproducibility")
        
        # Quick win suggestions
        recommendations.append("\nQuick wins to try first:")
        recommendations.append("• Gradient boosting model (XGBoost/LightGBM) with minimal tuning")
        if problem_type == "classification" and is_binary:
            recommendations.append("• LogisticRegression with basic preprocessing (baseline)")
        else:
            recommendations.append("• RandomForest with default parameters (robust baseline)")
        
        if 'feature_importance' in self.results and self.results['feature_importance']:
            recommendations.append("• Model using only top 5-10 most important features")
        
        recommendations.append("• Simple feature engineering: handle missing values + encode categoricals")
        
        # Processing time info
        self._end_timer('model_recommendations')
        
        return recommendations
    
    def generate_summary(self):
        """Generate a concise, actionable summary of the dataset."""
        self._start_timer('generate_summary')
        
        summary = []
        
        # Dataset overview
        summary.append("=" * 40)
        summary.append("DATASET OVERVIEW")
        summary.append("=" * 40)
        summary.append(f"• Dimensions: {self.original_shape[0]:,} rows × {self.original_shape[1]:,} columns")
        summary.append(f"• Memory usage: {self.results['basic_info']['memory_usage_mb']:.2f} MB")
        
        column_types = self.results['basic_info']['column_types']
        summary.append(f"• Column composition: {column_types['numerical']} numerical, {column_types['categorical']} categorical, "
                     f"{column_types['datetime']} datetime, {column_types['binary']} binary")
        
        # Data quality issues
        if self.results['potential_issues']:
            summary.append("\n" + "=" * 40)
            summary.append("DATA QUALITY ISSUES")
            summary.append("=" * 40)
            
            # Sort issues by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            sorted_issues = sorted(self.results['potential_issues'], 
                                  key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
            
            for issue in sorted_issues:
                severity = issue.get('severity', 'low').upper()
                summary.append(f"• [{severity}] {issue['description']}")
        
        # Target insights
        if 'target_analysis' in self.results:
            summary.append("\n" + "=" * 40)
            summary.append(f"TARGET VARIABLE: {self.target_col}")
            summary.append("=" * 40)
            
            target_type = self.results['target_analysis']['type']
            summary.append(f"• Type: {target_type}")
            
            if target_type == 'numerical':
                summary.append(f"• Range: {self.results['target_analysis']['min']} to {self.results['target_analysis']['max']}")
                summary.append(f"• Central tendency: mean={self.results['target_analysis']['mean']:.2f}, "
                            f"median={self.results['target_analysis']['median']:.2f}")
                summary.append(f"• Distribution: skewness={self.results['target_analysis']['skew']:.2f}, "
                            f"kurtosis={self.results['target_analysis']['kurtosis']:.2f}")
                
                if 'outliers_percentage' in self.results['target_analysis']:
                    summary.append(f"• Outliers: {self.results['target_analysis']['outliers_percentage']:.1f}% of values")
            else:
                summary.append(f"• Classes: {self.results['target_analysis']['unique_classes']}")
                
                # For multiclass, show class distribution
                if self.results['target_analysis']['unique_classes'] <= 5:
                    # Show percentages for all classes if few
                    class_info = []
                    for cls, count in self.results['target_analysis']['class_counts'].items():
                        pct = count / sum(self.results['target_analysis']['class_counts'].values()) * 100
                        class_info.append(f"{cls}: {pct:.1f}%")
                    summary.append(f"• Class distribution: {', '.join(class_info)}")
                else:
                    # Just show most common for many classes
                    summary.append(f"• Most common class: {self.results['target_analysis']['most_common']} "
                                 f"({self.results['target_analysis']['most_common_percentage']:.1f}%)")
                
                if 'imbalance_ratio' in self.results['target_analysis']:
                    summary.append(f"• Class imbalance ratio: {self.results['target_analysis']['imbalance_ratio']:.1f}")
        
        # Key predictors
        if 'feature_importance' in self.results and self.results['feature_importance']:
            summary.append("\n" + "=" * 40)
            summary.append("TOP PREDICTIVE FEATURES")
            summary.append("=" * 40)
            
            for i, (feature, importance) in enumerate(list(self.results['feature_importance'].items())[:10], 1):
                summary.append(f"{i}. {feature}: {importance:.4f}")
        
        # Correlations with target
        if 'correlations' in self.results and 'target_correlations' in self.results['correlations']:
            top_pos = self.results['correlations']['target_correlations']['top_positives']
            top_neg = self.results['correlations']['target_correlations']['top_negatives']
            
            summary.append("\n" + "=" * 40)
            summary.append(f"TARGET CORRELATIONS WITH {self.target_col}")
            summary.append("=" * 40)
            
            if top_pos:
                summary.append("Strongest positive correlations:")
                for i, (feature, corr) in enumerate(list(top_pos.items())[:5], 1):
                    summary.append(f"{i}. {feature}: {corr:.3f}")
            
            if top_neg:
                summary.append("\nStrongest negative correlations:")
                for i, (feature, corr) in enumerate(list(top_neg.items())[:5], 1):
                    summary.append(f"{i}. {feature}: {corr:.3f}")
        
        # Quick recommendations
        model_recs = self.get_model_recommendations()
        if model_recs:
            summary.append("\n" + "=" * 40)
            summary.append("MODEL RECOMMENDATIONS")
            summary.append("=" * 40)
            summary.extend(model_recs)
        
        # Computation time
        total_time = time() - self.start_time
        summary.append("\n" + "=" * 40)
        summary.append("PERFORMANCE METRICS")
        summary.append("=" * 40)
        summary.append(f"• Total analysis time: {total_time:.2f} seconds")
        
        if 'computation_time' in self.results:
            # Sort operations by duration
            sorted_ops = sorted(
                [(op, time_info.get('duration', 0)) 
                 for op, time_info in self.results['computation_time'].items()
                 if 'duration' in time_info],
                key=lambda x: x[1],
                reverse=True
            )
            
            if sorted_ops:
                summary.append("• Operation times:")
                for op, duration in sorted_ops[:5]:  # Show top 5 most time-consuming operations
                    summary.append(f"  - {op}: {duration:.2f}s")
        
        # End timer
        self._end_timer('generate_summary')
        
        return summary
    
    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for HTML."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def generate_html_report(self):
        """Generate a simple HTML report for sharing results."""
        self._start_timer('html_report')
        
        # Generate summary
        summary_text = self.generate_summary()
        
        # Create plots if not already done
        if not hasattr(self, 'figures') or not self.figures:
            self.plot_key_insights()
            
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fast EDA Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #f8f9fa;
                }}
                .header {{
                    background-color: #343a40;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h2 {{
                    color: #343a40;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                }}
                pre {{
                    background-color: #f1f3f5;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                .plots {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .plots img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #dee2e6;
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #e9ecef;
                }}
                .highlight {{
                    font-weight: bold;
                    color: #dc3545;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 0.9em;
                    color: #6c757d;
                }}
                .metrics {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metric-card {{
                    flex: 0 0 30%;
                    background-color: #e9ecef;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .issue-high {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .issue-medium {{
                    color: #fd7e14;
                    font-weight: bold;
                }}
                .issue-low {{
                    color: #20c997;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fast EDA Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div>Rows</div>
                        <div class="metric-value">{self.original_shape[0]:,}</div>
                    </div>
                    <div class="metric-card">
                        <div>Columns</div>
                        <div class="metric-value">{self.original_shape[1]:,}</div>
                    </div>
                    <div class="metric-card">
                        <div>Memory Usage</div>
                        <div class="metric-value">{self.results['basic_info']['memory_usage_mb']:.2f} MB</div>
                    </div>
                </div>
                
                <h3>Column Types</h3>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Examples</th>
                    </tr>
                    <tr>
                        <td>Numerical</td>
                        <td>{len(self.num_cols)}</td>
                        <td>{', '.join(self.num_cols[:5])}{'...' if len(self.num_cols) > 5 else ''}</td>
                    </tr>
                    <tr>
                        <td>Categorical</td>
                        <td>{len(self.cat_cols)}</td>
                        <td>{', '.join(self.cat_cols[:5])}{'...' if len(self.cat_cols) > 5 else ''}</td>
                    </tr>
                    <tr>
                        <td>Datetime</td>
                        <td>{len(self.date_cols)}</td>
                        <td>{', '.join(self.date_cols[:5])}{'...' if len(self.date_cols) > 5 else ''}</td>
                    </tr>
                    <tr>
                        <td>Binary</td>
                        <td>{len(self.binary_cols)}</td>
                        <td>{', '.join(self.binary_cols[:5])}{'...' if len(self.binary_cols) > 5 else ''}</td>
                    </tr>
                    <tr>
                        <td>ID-like</td>
                        <td>{len(self.id_cols)}</td>
                        <td>{', '.join(self.id_cols[:5])}{'...' if len(self.id_cols) > 5 else ''}</td>
                    </tr>
                    <tr>
                        <td>Constant</td>
                        <td>{len(self.const_cols)}</td>
                        <td>{', '.join(self.const_cols[:5])}{'...' if len(self.const_cols) > 5 else ''}</td>
                    </tr>
                </table>
        """
        
        # Add data quality issues
        if self.results['potential_issues']:
            html += """
                <h3>Data Quality Issues</h3>
                <table>
                    <tr>
                        <th>Issue Type</th>
                        <th>Severity</th>
                        <th>Description</th>
                    </tr>
            """
            
            for issue in self.results['potential_issues']:
                severity_class = f"issue-{issue.get('severity', 'low')}"
                html += f"""
                    <tr>
                        <td>{issue['issue'].replace('_', ' ').title()}</td>
                        <td class="{severity_class}">{issue.get('severity', 'low').upper()}</td>
                        <td>{issue['description']}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        # Close dataset overview section
        html += """
            </div>
        """
        
        # Add target analysis if available
        if 'target_analysis' in self.results:
            target_type = self.results['target_analysis']['type']
            
            html += f"""
            <div class="section">
                <h2>Target Variable: {self.target_col}</h2>
                <p><strong>Type:</strong> {target_type.title()}</p>
            """
            
            if target_type == 'numerical':
                html += f"""
                <div class="metrics">
                    <div class="metric-card">
                        <div>Mean</div>
                        <div class="metric-value">{self.results['target_analysis']['mean']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div>Median</div>
                        <div class="metric-value">{self.results['target_analysis']['median']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div>Standard Deviation</div>
                        <div class="metric-value">{self.results['target_analysis']['std']:.2f}</div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Skewness</th>
                        <th>Kurtosis</th>
                        <th>Outliers</th>
                    </tr>
                    <tr>
                        <td>{self.results['target_analysis']['min']}</td>
                        <td>{self.results['target_analysis']['max']}</td>
                        <td>{self.results['target_analysis']['skew']:.2f}</td>
                        <td>{self.results['target_analysis']['kurtosis']:.2f}</td>
                        <td>{self.results['target_analysis'].get('outliers_percentage', 0):.1f}%</td>
                    </tr>
                </table>
                """
            else:
                # For categorical targets
                html += """
                <h3>Class Distribution</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                """
                
                # Add class distribution
                class_counts = self.results['target_analysis']['class_counts']
                total = sum(class_counts.values())
                
                # Limit to top 10 classes if there are many
                classes = list(class_counts.keys())
                if len(classes) > 10:
                    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    for cls, count in top_classes:
                        pct = count / total * 100
                        html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{count:,}</td>
                            <td>{pct:.1f}%</td>
                        </tr>
                        """
                    
                    # Add row for others
                    other_count = sum(count for cls, count in class_counts.items() 
                                     if cls not in [c[0] for c in top_classes])
                    other_pct = other_count / total * 100
                    
                    html += f"""
                    <tr>
                        <td>Others ({len(classes) - 10} classes)</td>
                        <td>{other_count:,}</td>
                        <td>{other_pct:.1f}%</td>
                    </tr>
                    """
                else:
                    # Show all classes if there are few
                    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                        pct = count / total * 100
                        html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{count:,}</td>
                            <td>{pct:.1f}%</td>
                        </tr>
                        """
                
                html += """
                </table>
                """
                
                # Add class imbalance info if available
                if 'imbalance_ratio' in self.results['target_analysis']:
                    ratio = self.results['target_analysis']['imbalance_ratio']
                    severity = "high" if ratio > 10 else "medium" if ratio > 3 else "low"
                    
                    html += f"""
                    <h3>Class Imbalance</h3>
                    <p class="issue-{severity}">Imbalance Ratio: {ratio:.1f}</p>
                    <p>The ratio between the most frequent and least frequent class.</p>
                    """
            
            html += """
            </div>
            """
        
        # Add feature importance if available
        if 'feature_importance' in self.results and self.results['feature_importance']:
            html += """
            <div class="section">
                <h2>Feature Importance</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            for i, (feature, importance) in enumerate(list(self.results['feature_importance'].items())[:20], 1):
                html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature}</td>
                    <td>{importance:.4f}</td>
                </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Add correlation analysis if available
        if 'correlations' in self.results:
            if 'high_corr_pairs' in self.results['correlations'] and self.results['correlations']['high_corr_pairs']:
                html += """
                <div class="section">
                    <h2>Feature Correlations</h2>
                    <h3>Highly Correlated Feature Pairs (>0.8)</h3>
                    <table>
                        <tr>
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Correlation</th>
                        </tr>
                """
                
                for pair in self.results['correlations']['high_corr_pairs'][:15]:  # Limit to 15 pairs
                    html += f"""
                    <tr>
                        <td>{pair['feature_1']}</td>
                        <td>{pair['feature_2']}</td>
                        <td>{pair['correlation']:.3f}</td>
                    </tr>
                    """
                
                if len(self.results['correlations']['high_corr_pairs']) > 15:
                    html += f"""
                    <tr>
                        <td colspan="3">... and {len(self.results['correlations']['high_corr_pairs']) - 15} more pairs</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                html += """
                </div>
                """
            
            # Add target correlations if available
            if 'target_correlations' in self.results['correlations']:
                html += f"""
                <div class="section">
                    <h2>Correlations with {self.target_col}</h2>
                    <div class="row">
                        <div class="col">
                            <h3>Top Positive Correlations</h3>
                            <table>
                                <tr>
                                    <th>Feature</th>
                                    <th>Correlation</th>
                                </tr>
                """
                
                # Change to:
                for feature, corr in list(self.results['correlations']['target_correlations']['top_positives'].items())[:5]:
                    html += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{corr:.3f}</td>
                    </tr>
                    """
                
                html += """
                            </table>
                        </div>
                        <div class="col">
                            <h3>Top Negative Correlations</h3>
                            <table>
                                <tr>
                                    <th>Feature</th>
                                    <th>Correlation</th>
                                </tr>
                """
                # Change to:
                for feature, corr in list(self.results['correlations']['target_correlations']['top_negatives'].items())[:5]:
                    html += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{corr:.3f}</td>
                    </tr>
                    """
                
                html += """
                            </table>
                        </div>
                    </div>
                </div>
                """
        
        # Add visualizations
        if hasattr(self, 'figures') and self.figures:
            html += """
            <div class="section">
                <h2>Visualizations</h2>
                <div class="plots">
            """
            
            # Add key insights plot
            if 'key_insights' in self.figures:
                img_str = self.fig_to_base64(self.figures['key_insights'])
                html += f'<img src="data:image/png;base64,{img_str}" alt="Key Insights">'
            
            html += """
                </div>
            </div>
            """
        
        # Add model recommendations
        model_recs = self.get_model_recommendations()
        if model_recs:
            html += """
            <div class="section">
                <h2>Model Recommendations</h2>
                <pre>""" + "\n".join(model_recs) + """</pre>
            </div>
            """
        
        # Add summary
        html += """
            <div class="section">
                <h2>Executive Summary</h2>
                <pre>""" + "\n".join(summary_text) + """</pre>
            </div>
        """
        
        # Performance metrics
        total_time = time() - self.start_time
        html += f"""
            <div class="section">
                <h2>Performance Metrics</h2>
                <p>Total analysis time: {total_time:.2f} seconds</p>
                
                <table>
                    <tr>
                        <th>Operation</th>
                        <th>Time (seconds)</th>
                    </tr>
        """
        
        if 'computation_time' in self.results:
            # Sort operations by duration
            sorted_ops = sorted(
                [(op, time_info.get('duration', 0)) 
                 for op, time_info in self.results['computation_time'].items()
                 if 'duration' in time_info],
                key=lambda x: x[1],
                reverse=True
            )
            
            for op, duration in sorted_ops:
                html += f"""
                <tr>
                    <td>{op}</td>
                    <td>{duration:.2f}</td>
                </tr>
                """
        
        html += """
                </table>
            </div>
        """
        
        # Close HTML
        html += """
            <div class="footer">
                <p>Generated by LightningEDA</p>
            </div>
        </body>
        </html>
        """
        
        self._end_timer('html_report')
        
        return html
    
    def run_quick_analysis(self):
        """Run a complete quick analysis pipeline."""
        if self.verbose:
            print("Starting Lightning-Fast EDA analysis...")
        
        # Run all analyses
        self.basic_analysis()
        self.analyze_target()
        self.analyze_numerical()
        self.analyze_categorical()
        self.analyze_correlations()
        self.quick_feature_importance()
        
        # Plot key insights
        self.plot_key_insights()
        
        # Generate summary
        summary = self.generate_summary()
        
        if self.verbose:
            print("\n" + "\n".join(summary))
        
        # Generate HTML report
        html_report = self.generate_html_report()
        
        # Save report to file
        report_file = f"lightning_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_report)
            
        if self.verbose:
            print(f"\nHTML report saved: {report_file}")
        
        return self


def quick_eda(df, target_col=None, sample_size=10000, verbose=True):
    """
    Run a lightning-fast, efficient EDA on any dataframe.
    Optimized for speed and actionable insights during datathons.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    target_col : str, optional
        Name of the target column
    sample_size : int
        Maximum number of rows to use for analysis
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    LightningEDA object
    """
    eda = LightningEDA(df, target_col=target_col, sample_size=sample_size, verbose=verbose)
    return eda.run_quick_analysis()

df = pd.read_csv('train.csv')
target_col = 'target'

eda = quick_eda(df, target_col=target_col, sample_size=10000, verbose=True)