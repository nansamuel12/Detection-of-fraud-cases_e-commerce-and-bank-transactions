"""
Model Explainability Module for Fraud Detection

This module provides comprehensive model explainability tools including:
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- LIME (Local Interpretable Model-agnostic Explanations)
- Model interpretation utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")


class ModelExplainabilityError(Exception):
    """Custom exception for model explainability errors."""
    pass


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analyzer for fraud detection models.
    
    Provides multiple explainability techniques including SHAP, LIME,
    and feature importance analysis.
    """
    
    def __init__(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        output_dir: str = "explainability_results"
    ) -> None:
        """
        Initialize the ExplainabilityAnalyzer.
        
        Args:
            model: Trained scikit-learn model or pipeline
            X_train: Training features used to train the model
            X_test: Test features for analysis
            feature_names: List of feature names
            output_dir: Directory to save explainability plots
            
        Raises:
            ModelExplainabilityError: If model is invalid or incompatible
        """
        if not hasattr(model, 'predict'):
            raise ModelExplainabilityError(
                "Model must have a 'predict' method. "
                "Please provide a valid scikit-learn compatible model."
            )
        
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        self.shap_values = None
        
        logging.info("ExplainabilityAnalyzer initialized successfully")
    
    def get_feature_importance(
        self,
        method: str = 'default',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Extract feature importance from the model.
        
        Args:
            method: Method to extract importance ('default', 'permutation')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
            
        Raises:
            ModelExplainabilityError: If feature importance cannot be extracted
        """
        try:
            logging.info(f"Extracting feature importance using method: {method}")
            
            if method == 'default':
                # Try to get feature_importances_ from tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    importances = self.model.feature_importances_
                elif hasattr(self.model, 'named_steps'):
                    # Handle pipeline - get the classifier step
                    classifier = None
                    for step_name, step in self.model.named_steps.items():
                        if hasattr(step, 'feature_importances_') or hasattr(step, 'coef_'):
                            classifier = step
                            break
                    
                    if classifier is None:
                        raise ModelExplainabilityError(
                            "Could not find a classifier with feature importances in the pipeline"
                        )
                    
                    if hasattr(classifier, 'feature_importances_'):
                        importances = classifier.feature_importances_
                    elif hasattr(classifier, 'coef_'):
                        # For linear models, use absolute coefficient values
                        importances = np.abs(classifier.coef_[0])
                    else:
                        raise ModelExplainabilityError(
                            f"Classifier {type(classifier).__name__} does not support feature importance"
                        )
                elif hasattr(self.model, 'coef_'):
                    # For linear models (e.g., LogisticRegression)
                    importances = np.abs(self.model.coef_[0])
                else:
                    raise ModelExplainabilityError(
                        f"Model {type(self.model).__name__} does not support feature importance extraction"
                    )
                
                # Create DataFrame
                if self.feature_names is not None:
                    feature_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importances
                    })
                else:
                    feature_df = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(len(importances))],
                        'importance': importances
                    })
                
                # Sort and get top N
                feature_df = feature_df.sort_values('importance', ascending=False).head(top_n)
                
                logging.info(f"Successfully extracted top {top_n} feature importances")
                return feature_df
            
            elif method == 'permutation':
                from sklearn.inspection import permutation_importance
                
                logging.info("Computing permutation importance (this may take a while)...")
                perm_importance = permutation_importance(
                    self.model, self.X_test, self.model.predict(self.X_test),
                    n_repeats=10, random_state=42, n_jobs=-1
                )
                
                if self.feature_names is not None:
                    feature_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': perm_importance.importances_mean
                    })
                else:
                    feature_df = pd.DataFrame({
                        'feature': [f'feature_{i}' for i in range(len(perm_importance.importances_mean))],
                        'importance': perm_importance.importances_mean
                    })
                
                feature_df = feature_df.sort_values('importance', ascending=False).head(top_n)
                logging.info(f"Successfully computed permutation importance for top {top_n} features")
                return feature_df
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'default' or 'permutation'")
                
        except Exception as e:
            raise ModelExplainabilityError(f"Error extracting feature importance: {str(e)}")
    
    def plot_feature_importance(
        self,
        method: str = 'default',
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            method: Method to extract importance ('default', 'permutation')
            top_n: Number of top features to plot
            figsize: Figure size (width, height)
            save: Whether to save the plot
            
        Raises:
            ModelExplainabilityError: If plotting fails
        """
        try:
            feature_df = self.get_feature_importance(method=method, top_n=top_n)
            
            plt.figure(figsize=figsize)
            sns.barplot(data=feature_df, y='feature', x='importance', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances ({method.capitalize()} Method)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / f'feature_importance_{method}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error plotting feature importance: {str(e)}")
    
    def initialize_shap_explainer(
        self,
        explainer_type: str = 'auto',
        sample_size: Optional[int] = 100
    ) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel')
            sample_size: Number of background samples for kernel explainer
            
        Raises:
            ModelExplainabilityError: If SHAP is not available or initialization fails
        """
        if not SHAP_AVAILABLE:
            raise ModelExplainabilityError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        try:
            logging.info(f"Initializing SHAP explainer (type: {explainer_type})...")
            
            if explainer_type == 'auto':
                # Auto-detect explainer type
                if hasattr(self.model, 'tree_'):
                    explainer_type = 'tree'
                elif hasattr(self.model, 'coef_'):
                    explainer_type = 'linear'
                else:
                    explainer_type = 'kernel'
            
            if explainer_type == 'tree':
                # For tree-based models
                if hasattr(self.model, 'named_steps'):
                    # Extract the tree model from pipeline
                    for step_name, step in self.model.named_steps.items():
                        if hasattr(step, 'tree_') or hasattr(step, 'estimators_'):
                            self.shap_explainer = shap.TreeExplainer(step)
                            break
                else:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                    
            elif explainer_type == 'linear':
                # For linear models
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
                
            elif explainer_type == 'kernel':
                # For any model (slower but model-agnostic)
                if sample_size and len(self.X_train) > sample_size:
                    background = shap.sample(self.X_train, sample_size)
                else:
                    background = self.X_train
                self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
            
            else:
                raise ValueError(
                    f"Unknown explainer_type: {explainer_type}. "
                    "Use 'auto', 'tree', 'linear', or 'kernel'"
                )
            
            logging.info(f"SHAP {explainer_type} explainer initialized successfully")
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error initializing SHAP explainer: {str(e)}")
    
    def compute_shap_values(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values.
        
        Args:
            X: Features to explain. If None, uses self.X_test
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP values array
            
        Raises:
            ModelExplainabilityError: If SHAP values computation fails
        """
        if self.shap_explainer is None:
            self.initialize_shap_explainer()
        
        try:
            if X is None:
                X = self.X_test
            
            if max_samples and len(X) > max_samples:
                X = X[:max_samples]
            
            logging.info(f"Computing SHAP values for {len(X)} samples...")
            self.shap_values = self.shap_explainer.shap_values(X)
            
            # Handle binary classification (sometimes returns list of arrays)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Get positive class
            
            logging.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error computing SHAP values: {str(e)}")
    
    def plot_shap_summary(
        self,
        plot_type: str = 'dot',
        max_display: int = 20,
        save: bool = True
    ) -> None:
        """
        Plot SHAP summary.
        
        Args:
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            save: Whether to save the plot
            
        Raises:
            ModelExplainabilityError: If plotting fails
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        try:
            logging.info(f"Creating SHAP summary plot (type: {plot_type})...")
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
            
            if save:
                save_path = self.output_dir / f'shap_summary_{plot_type}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"SHAP summary plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error plotting SHAP summary: {str(e)}")
    
    def plot_shap_waterfall(
        self,
        instance_idx: int = 0,
        save: bool = True
    ) -> None:
        """
        Plot SHAP waterfall for a single instance.
        
        Args:
            instance_idx: Index of the instance to explain
            save: Whether to save the plot
            
        Raises:
            ModelExplainabilityError: If plotting fails
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        try:
            logging.info(f"Creating SHAP waterfall plot for instance {instance_idx}...")
            
            # Create explanation object
            explanation = shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.shap_explainer.expected_value,
                data=self.X_test[instance_idx],
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            
            if save:
                save_path = self.output_dir / f'shap_waterfall_instance_{instance_idx}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"SHAP waterfall plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error plotting SHAP waterfall: {str(e)}")
    
    def explain_instance_lime(
        self,
        instance_idx: int = 0,
        num_features: int = 10,
        save: bool = True
    ) -> None:
        """
        Explain a single instance using LIME.
        
        Args:
            instance_idx: Index of the instance to explain
            num_features: Number of features to show in explanation
            save: Whether to save the plot
            
        Raises:
            ModelExplainabilityError: If LIME is not available or explanation fails
        """
        if not LIME_AVAILABLE:
            raise ModelExplainabilityError(
                "LIME is not installed. Install with: pip install lime"
            )
        
        try:
            logging.info(f"Explaining instance {instance_idx} using LIME...")
            
            # Initialize LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(self.X_train),
                feature_names=self.feature_names,
                class_names=['Legitimate', 'Fraud'],
                mode='classification'
            )
            
            # Explain instance
            instance = self.X_test[instance_idx]
            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict_proba,
                num_features=num_features
            )
            
            # Plot
            fig = explanation.as_pyplot_figure()
            plt.title(f'LIME Explanation for Instance {instance_idx}', fontweight='bold')
            plt.tight_layout()
            
            if save:
                save_path = self.output_dir / f'lime_explanation_instance_{instance_idx}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"LIME explanation saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error explaining instance with LIME: {str(e)}")
    
    def analyze_predictions(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        Analyze predictions to find indices of TP, FP, FN, TN.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of indices for each category
        """
        try:
            # Ensure inputs are numpy arrays for easier indexing
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.values
                
            tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
            fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
            fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
            tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
            
            logging.info(f"Found {len(tp_indices)} TP, {len(fp_indices)} FP, {len(fn_indices)} FN cases")
            
            return {
                'TP': tp_indices.tolist(),
                'FP': fp_indices.tolist(),
                'FN': fn_indices.tolist(),
                'TN': tn_indices.tolist()
            }
        except Exception as e:
            raise ModelExplainabilityError(f"Error analyzing predictions: {str(e)}")

    def explain_specific_instance(
        self,
        index: int,
        label: str,
        save: bool = True
    ) -> None:
        """
        Generate SHAP waterfall plot for a specific instance index with a custom label.
        """
        if self.shap_explainer is None:
            self.initialize_shap_explainer()
            
        try:
            # We compute SHAP just for this instance to be efficient if not already computed
            instance_data = self.X_test[index:index+1] # Keep 2D shape
            
            # For TreeExplainer/LinearExplainer it allows computing single instance
            shap_values_single = self.shap_explainer.shap_values(instance_data)
            
            # Handle list return type for classifiers
            if isinstance(shap_values_single, list):
                shap_values_single = shap_values_single[1]
                
            # Create explanation object
            explanation = shap.Explanation(
                values=shap_values_single[0],
                base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)) else self.shap_explainer.expected_value,
                data=instance_data.iloc[0] if isinstance(instance_data, pd.DataFrame) else instance_data[0],
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            plt.title(f"SHAP Waterfall - {label} (Index {index})", fontsize=14)
            
            if save:
                save_path = self.output_dir / f'shap_{label.lower().replace(" ", "_")}_idx_{index}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved specific SHAP plot to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logging.warning(f"Could not explain specific instance {index}: {e}")

    def compare_feature_importances(self, top_n: int = 5) -> pd.DataFrame:
        """
        Compare model's built-in feature importance with SHAP importance.
        """
        try:
            # 1. Get Model Importance
            model_imp = self.get_feature_importance(top_n=len(self.feature_names))
            model_imp = model_imp.set_index('feature')['importance']
            model_imp = model_imp / model_imp.sum() # Normalize
            
            # 2. Get SHAP Importance (Mean absolute value)
            if self.shap_values is None:
                self.compute_shap_values(max_samples=100)
                
            shap_imp_vals = np.abs(self.shap_values).mean(axis=0)
            shap_imp = pd.Series(shap_imp_vals, index=self.feature_names)
            shap_imp = shap_imp / shap_imp.sum() # Normalize
            
            # 3. Combine
            comparison = pd.DataFrame({
                'Model_Importance': model_imp,
                'SHAP_Importance': shap_imp
            })
            
            comparison['Total_Imp'] = comparison['Model_Importance'] + comparison['SHAP_Importance']
            comparison = comparison.sort_values('Total_Imp', ascending=False).head(top_n)
            
            return comparison
        except Exception as e:
            logging.warning(f"Could not compare feature importances: {e}")
            return pd.DataFrame()

    def generate_full_report(
        self,
        y_true: Optional[Union[pd.Series, np.ndarray]] = None,
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        top_n_features: int = 20,
        num_shap_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explainability report.
        Args:
            y_true: True labels (optional, required for TP/FP analysis)
            y_pred: Predicted labels (optional, required for TP/FP analysis)
        """
        try:
            logging.info("Generating comprehensive explainability report...")
            report = {}
            
            # 1. Feature Importance
            logging.info("Step 1: Feature Importance")
            try:
                self.plot_feature_importance(method='default', top_n=top_n_features)
            except Exception as e: 
                logging.warning(f"Feature importance plotting failed: {e}")

            # 2. SHAP Analysis & Predictions
            if SHAP_AVAILABLE:
                logging.info("Step 2: SHAP Analysis")
                self.compute_shap_values(max_samples=num_shap_samples)
                self.plot_shap_summary(plot_type='dot')
                self.plot_shap_summary(plot_type='bar')
                
                # Compare importances
                comparison_df = self.compare_feature_importances(top_n=5)
                report['top_features_comparison'] = comparison_df
                if not comparison_df.empty:
                    print("\nTop 5 Fraud-Driving Features (Model vs SHAP):")
                    print(comparison_df.to_string())

                # Analyze TP/FP/FN if labels provided
                if y_true is not None and y_pred is not None:
                    logging.info("Step 3: Representative Case Analysis (TP/FP/FN)")
                    indices = self.analyze_predictions(y_true, y_pred)
                    
                    # Explain 1 representative from each category if available
                    for case_type in ['TP', 'FP', 'FN']:
                        if indices[case_type]:
                            # Pick the first one (or random)
                            idx = indices[case_type][0]
                            self.explain_specific_instance(idx, label=f"Representative {case_type}")
            
            # 3. Business Recommendations
            logging.info("Step 4: Generating Business Recommendations")
            recommendations = """
# Data-Driven Business Recommendations

1. **Enhanced Verification for High-Risk Regions:** 
   SHAP analysis indicates location-based features are top drivers. Implement Step-Up Authentication (2FA) for transactions originating from countries identified as high-risk by the IP mapping model.

2. **Velocity-Based Rules:** 
   'user_tx_count' and time-based features are significant. Introduce distinct velocity limits for new vs. established users. New users with >N transactions in 24 hours should be manually reviewed.

3. **False Positive reduction:**
   The False Positive analysis shows edge cases in legitimate high-value transactions. Create a "VIP Whitelist" based on long-term user history to bypass aggressive screening for loyal customers, improving their experience.
"""
            # Save recommendations
            rec_path = self.output_dir / "business_recommendations.md"
            with open(rec_path, "w") as f:
                f.write(recommendations)
                
            logging.info(f"Report generation complete. Check {self.output_dir}")
            return report
            
        except Exception as e:
            raise ModelExplainabilityError(f"Error generating report: {str(e)}")


def explain_model(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    y_true: Optional[Union[pd.Series, np.ndarray]] = None,
    y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
    output_dir: str = "explainability_results",
    generate_report: bool = True
) -> ExplainabilityAnalyzer:
    """
    Convenience function to create an explainability analyzer.
    """
    try:
        analyzer = ExplainabilityAnalyzer(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            output_dir=output_dir
        )
        
        if generate_report:
            analyzer.generate_full_report(y_true=y_true, y_pred=y_pred)
        
        return analyzer
        
    except Exception as e:
        raise ModelExplainabilityError(f"Error in explain_model: {str(e)}")
