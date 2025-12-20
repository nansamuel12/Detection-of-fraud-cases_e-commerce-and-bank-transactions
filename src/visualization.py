import matplotlib.pyplot as plt
import seaborn as sns
import logging

def set_style():
    sns.set_style("whitegrid")
    sns.set_palette("muted")

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Plots the count of each class.
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
    plt.ylabel("Count")
    
    # Add percentages
    total = len(y)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y_coord = p.get_height()
        ax.annotate(percentage, (x, y_coord), ha='center')
        
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_numerical_distributions(df, numerical_cols, hue=None, save_dir=None):
    """
    Plots distributions of numerical columns.
    """
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, hue=hue, kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        
        if save_dir:
            path = f"{save_dir}/distribution_{col}.png"
            plt.savefig(path)
            logging.info(f"Saved plot to {path}")
            plt.close()
        else:
            plt.show()

def plot_categorical_breakdown(df, cat_cols, target=None, save_dir=None):
    """
    Plots breakdown of categorical features vs target.
    """
    for col in cat_cols:
        if df[col].nunique() > 20:
             logging.info(f"Skipping plot for {col} - too many unique values.")
             continue
             
        plt.figure(figsize=(10, 5))
        if target:
            sns.barplot(x=col, y=target, data=df)
            plt.title(f"Fraud Probability by {col}")
        else:
            sns.countplot(x=col, data=df)
            plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        
        if save_dir:
             path = f"{save_dir}/categorical_{col}.png"
             plt.savefig(path)
             logging.info(f"Saved plot to {path}")
             plt.close()
        else:
             plt.show()
