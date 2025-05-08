"""
MAGIC Gamma Telescope Dataset - Exploratory Data Analysis
UCI Dataset: Detection of gamma particles vs. background hadron particles
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set random seed and styling
np.random.seed(42)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------------
# 1. DATA LOADING AND CLEANING
# ---------------------------------------------------------------------------------


def load_and_clean_data():
    """
    Load and preprocess the MAGIC04 dataset.

    Returns:
        pd.DataFrame: Cleaned dataframe with proper dtypes and encoded target
    """
    # Load the data with the correct column names
    cols = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class",
    ]

    df = pd.read_csv("magic04.data", names=cols)

    # Display basic info
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nDescriptive statistics:")
    print(df.describe())

    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)

    # Convert column names to snake_case
    df.columns = [col.lower() for col in df.columns]

    # Encode class: g (gamma) → 1, h (hadron) → 0
    df["class"] = df["class"].map({"g": 1, "h": 0})

    # Downcast numeric types for efficiency
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

    # Rename columns to be more descriptive
    rename_dict = {
        "flength": "length",
        "fwidth": "width",
        "fsize": "size",
        "fconc": "concentration",
        "fconc1": "concentration1",
        "fasym": "asymmetry",
        "fm3long": "m3_long",
        "fm3trans": "m3_trans",
        "falpha": "alpha",
        "fdist": "distance",
    }
    df = df.rename(columns=rename_dict)

    print("\nAfter cleaning - Memory usage:")
    print(df.memory_usage(deep=True))

    return df


# ---------------------------------------------------------------------------------
# 2. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------------------------


def plot_histograms(df, feature, bins=30):
    """
    Plot histogram with KDE for a feature split by class.

    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature column name
        bins (int): Number of histogram bins
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms by class with KDE
    sns.histplot(
        data=df,
        x=feature,
        hue="class",
        element="step",
        stat="density",
        common_norm=False,
        palette=["blue", "red"],
        bins=bins,
        alpha=0.6,
        kde=True,
    )

    # Add vertical lines for means
    gamma_mean = df[df["class"] == 1][feature].mean()
    hadron_mean = df[df["class"] == 0][feature].mean()

    plt.axvline(
        gamma_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Gamma mean: {gamma_mean:.2f}",
    )
    plt.axvline(
        hadron_mean,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Hadron mean: {hadron_mean:.2f}",
    )

    plt.title(f"Distribution of {feature} by Class")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend(
        title="Class",
        labels=[
            "Hadron (0)",
            "Gamma (1)",
            f"Gamma mean: {gamma_mean:.2f}",
            f"Hadron mean: {hadron_mean:.2f}",
        ],
    )
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, feature):
    """
    Plot boxplots for a feature split by class.

    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature column name
    """
    plt.figure(figsize=(10, 6))

    # Plot boxplots by class
    sns.boxplot(data=df, x="class", y=feature, palette=["blue", "red"])

    plt.title(f"Boxplot of {feature} by Class")
    plt.xlabel("Class (0=Hadron, 1=Gamma)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


def compute_statistics(df, feature_list):
    """
    Compute skewness and kurtosis for given features.

    Args:
        df (pd.DataFrame): Input dataframe
        feature_list (list): List of feature column names

    Returns:
        pd.DataFrame: DataFrame with skewness and kurtosis for each feature
    """
    stats_df = pd.DataFrame(columns=["Feature", "Skewness", "Kurtosis"])

    for i, feature in enumerate(feature_list):
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()

        stats_df.loc[i] = [feature, skew, kurt]

    return stats_df


def analyze_univariate(df):
    """
    Perform univariate analysis on all features.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Get feature columns (exclude class column)
    features = df.columns.tolist()
    features.remove("class")

    # Plot histograms with KDE for each feature
    for feature in features:
        plot_histograms(df, feature)

    # Plot boxplots for each feature
    for feature in features:
        plot_boxplots(df, feature)

    # Compute skewness and kurtosis
    stats_df = compute_statistics(df, features)
    print("\nSkewness and Kurtosis by Feature:")
    print(stats_df)

    # Highlight highly skewed features
    high_skew = stats_df[abs(stats_df["Skewness"]) > 1].sort_values(
        "Skewness", ascending=False
    )
    print("\nHighly skewed features (|skewness| > 1):")
    print(high_skew)

    # Highlight features with high kurtosis
    high_kurt = stats_df[abs(stats_df["Kurtosis"]) > 3].sort_values(
        "Kurtosis", ascending=False
    )
    print("\nFeatures with high kurtosis (|kurtosis| > 3):")
    print(high_kurt)


# ---------------------------------------------------------------------------------
# 3. BIVARIATE ANALYSIS AND CORRELATION
# ---------------------------------------------------------------------------------


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for all features.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Find highly correlated feature pairs
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    print("\nHighly correlated feature pairs (|r| > 0.8):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.4f}")


def find_discriminative_features(df):
    """
    Find the most discriminative features between classes.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        list: List of most discriminative feature names
    """
    # Get feature columns
    features = df.columns.tolist()
    features.remove("class")

    # Calculate class means
    gamma_means = df[df["class"] == 1][features].mean()
    hadron_means = df[df["class"] == 0][features].mean()

    # Calculate difference between means
    mean_diff = abs(gamma_means - hadron_means)

    # Normalize by feature standard deviations for fair comparison
    feature_stds = df[features].std()
    normalized_diff = mean_diff / feature_stds

    # Sort and get top discriminative features
    top_features = normalized_diff.sort_values(ascending=False).head(6).index.tolist()

    print("\nTop discriminative features (normalized mean difference):")
    for feature in top_features:
        print(f"{feature}: {normalized_diff[feature]:.4f}")

    return top_features


def plot_pairplot(df, features):
    """
    Create pairplot for selected features.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to plot
    """
    # Create pairplot
    plot_df = df[features + ["class"]].copy()

    # Map class to descriptive names for the plot
    plot_df["class"] = plot_df["class"].map({1: "Gamma", 0: "Hadron"})

    plt.figure(figsize=(12, 10))
    pair_plot = sns.pairplot(
        plot_df, hue="class", palette=["blue", "red"], corner=True, diag_kind="kde"
    )

    pair_plot.fig.suptitle("Pairplot of Top Discriminative Features", y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------
# 4. CLASS BALANCE AND STATISTICS
# ---------------------------------------------------------------------------------


def analyze_class_balance(df):
    """
    Analyze class distribution and statistics by class.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Count observations by class
    class_counts = df["class"].value_counts().sort_index()
    class_names = {0: "Hadron", 1: "Gamma"}

    # Plot class distribution
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x=class_counts.index.map(lambda x: class_names[x]),
        y=class_counts.values,
        palette=["blue", "red"],
    )

    # Add count labels on bars
    for i, count in enumerate(class_counts):
        ax.text(
            i,
            count / 2,
            f"{count} ({count/len(df):.1%})",
            ha="center",
            color="white",
            fontweight="bold",
        )

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Calculate statistics by class
    features = df.columns.tolist()
    features.remove("class")

    stats_by_class = df.groupby("class")[features].agg(["mean", "std"]).T
    stats_by_class.columns = pd.MultiIndex.from_product(
        [["Hadron", "Gamma"], ["mean", "std"]]
    )

    print("\nFeature statistics by class:")
    print(stats_by_class)


def plot_violin_plots(df, features):
    """
    Create violin plots for selected features.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to plot
    """
    # Plot violin plots for each selected feature
    for feature in features:
        plt.figure(figsize=(10, 6))

        sns.violinplot(
            data=df, x="class", y=feature, palette=["blue", "red"], inner="quartile"
        )

        plt.title(f"Violin Plot of {feature} by Class")
        plt.xlabel("Class (0=Hadron, 1=Gamma)")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------------
# 5. OUTLIER DETECTION AND HANDLING
# ---------------------------------------------------------------------------------


def detect_outliers(df):
    """
    Detect outliers using IQR method.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: DataFrame with outlier count for each feature
    """
    features = df.columns.tolist()
    features.remove("class")

    outlier_counts = pd.DataFrame(columns=["Feature", "Outliers", "Percentage"])

    for i, feature in enumerate(features):
        # Calculate IQR
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1

        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(df) * 100

        outlier_counts.loc[i] = [feature, outlier_count, outlier_pct]

    outlier_counts = outlier_counts.sort_values("Percentage", ascending=False)

    return outlier_counts


def plot_outliers(df, feature):
    """
    Plot boxplot with outliers highlighted for a feature.

    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature column name
    """
    # Calculate IQR
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x="class", y=feature, palette=["blue", "red"])
    plt.title(f"Boxplot of {feature} with Outlier Thresholds")
    plt.xlabel("Class (0=Hadron, 1=Gamma)")
    plt.ylabel(feature)
    plt.axhline(
        y=upper_bound,
        color="red",
        linestyle="--",
        label=f"Upper bound: {upper_bound:.2f}",
    )
    plt.axhline(
        y=lower_bound,
        color="green",
        linestyle="--",
        label=f"Lower bound: {lower_bound:.2f}",
    )
    plt.legend()

    # Plot histogram with outlier bounds
    plt.subplot(1, 2, 2)
    sns.histplot(
        data=df,
        x=feature,
        hue="class",
        element="step",
        palette=["blue", "red"],
        alpha=0.6,
    )
    plt.axvline(
        x=upper_bound,
        color="red",
        linestyle="--",
        label=f"Upper bound: {upper_bound:.2f}",
    )
    plt.axvline(
        x=lower_bound,
        color="green",
        linestyle="--",
        label=f"Lower bound: {lower_bound:.2f}",
    )
    plt.title(f"Distribution of {feature} with Outlier Thresholds")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()


def suggest_transformations(df, features):
    """
    Suggest transformations for skewed features.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of skewed features
    """
    print("\nSuggested transformations for skewed features:")

    for feature in features:
        skew = df[feature].skew()

        # Apply log transformation to positively skewed features
        if skew > 1:
            # Check if we have zero or negative values
            min_val = df[feature].min()

            if min_val <= 0:
                # Add a constant before log transform
                const = abs(min_val) + 1
                transformed = np.log(df[feature] + const)
                transform_type = f"log(x + {const})"
            else:
                transformed = np.log(df[feature])
                transform_type = "log(x)"

            new_skew = transformed.skew()
            print(
                f"{feature}: Original skew={skew:.4f}, "
                + f"After {transform_type} transform: skew={new_skew:.4f}"
            )

        # Apply square root for moderately positive skewed features
        elif skew > 0.5:
            # Check if we have negative values
            min_val = df[feature].min()

            if min_val < 0:
                # Add a constant before sqrt transform
                const = abs(min_val) + 1
                transformed = np.sqrt(df[feature] + const)
                transform_type = f"sqrt(x + {const})"
            else:
                transformed = np.sqrt(df[feature])
                transform_type = "sqrt(x)"

            new_skew = transformed.skew()
            print(
                f"{feature}: Original skew={skew:.4f}, "
                + f"After {transform_type} transform: skew={new_skew:.4f}"
            )

        # Apply square for negative skewed features
        elif skew < -0.5:
            transformed = df[feature] ** 2
            new_skew = transformed.skew()
            print(
                f"{feature}: Original skew={skew:.4f}, "
                + f"After square transform: skew={new_skew:.4f}"
            )

        else:
            print(f"{feature}: Skew={skew:.4f} is acceptable, no transformation needed")


# ---------------------------------------------------------------------------------
# 6. CREATIVE EXPLORATION (PCA, t-SNE)
# ---------------------------------------------------------------------------------


def plot_pca(df):
    """
    Perform PCA and plot the first two principal components.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    # Get feature columns
    features = df.columns.tolist()
    features.remove("class")

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Create dataframe for plotting
    pca_df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "class": df["class"].map({1: "Gamma", 0: "Hadron"}),
        }
    )

    # Plot PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="class", palette=["blue", "red"], alpha=0.6
    )

    plt.title("PCA: First Two Principal Components")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

    # Add a legend with better positioning
    plt.legend(title="Class", loc="best")

    plt.tight_layout()
    plt.show()

    # Print explained variance
    print("\nPCA explained variance ratio:")
    print(pca.explained_variance_ratio_)

    # Print feature importance
    print("\nPCA feature importance:")
    components_df = pd.DataFrame(
        pca.components_.T, columns=[f"PC{i+1}" for i in range(2)], index=features
    )
    print(components_df.abs().sort_values("PC1", ascending=False))


def plot_tsne(df, perplexity=30, random_state=42):
    """
    Perform t-SNE and plot the results.

    Args:
        df (pd.DataFrame): Input dataframe
        perplexity (int): t-SNE perplexity parameter
        random_state (int): Random seed
    """
    # Get feature columns
    features = df.columns.tolist()
    features.remove("class")

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Apply t-SNE
    # Using a sample for speed if dataset is large
    sample_size = min(5000, len(df))
    indices = np.random.choice(len(df), size=sample_size, replace=False)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(scaled_features[indices])

    # Create dataframe for plotting
    tsne_df = pd.DataFrame(
        {
            "tsne_1": tsne_result[:, 0],
            "tsne_2": tsne_result[:, 1],
            "class": df["class"].iloc[indices].map({1: "Gamma", 0: "Hadron"}),
        }
    )

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=tsne_df,
        x="tsne_1",
        y="tsne_2",
        hue="class",
        palette=["blue", "red"],
        alpha=0.6,
    )

    plt.title(f"t-SNE Visualization (perplexity={perplexity})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Add a legend with better positioning
    plt.legend(title="Class", loc="best")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------------------


def main():
    """
    Execute the full exploratory data analysis pipeline.
    """
    print("=" * 80)
    print("MAGIC GAMMA TELESCOPE DATASET ANALYSIS")
    print("=" * 80)

    # 1. Load and clean data
    print("\n\n" + "=" * 30 + " DATA LOADING AND CLEANING " + "=" * 30)
    df = load_and_clean_data()

    # 2. Univariate analysis
    print("\n\n" + "=" * 30 + " UNIVARIATE ANALYSIS " + "=" * 30)
    analyze_univariate(df)

    # 3. Bivariate analysis and correlation
    print("\n\n" + "=" * 30 + " BIVARIATE ANALYSIS AND CORRELATION " + "=" * 30)
    plot_correlation_heatmap(df)
    top_features = find_discriminative_features(df)
    plot_pairplot(df, top_features[:4])  # Use top 4 for clearer pairplot

    # 4. Class balance and statistics
    print("\n\n" + "=" * 30 + " CLASS BALANCE AND STATISTICS " + "=" * 30)
    analyze_class_balance(df)
    plot_violin_plots(df, top_features[:3])  # Use top 3 for violin plots

    # 5. Outlier detection and handling
    print("\n\n" + "=" * 30 + " OUTLIER DETECTION AND HANDLING " + "=" * 30)
    outlier_counts = detect_outliers(df)
    print("\nOutlier count by feature:")
    print(outlier_counts)

    # Plot top 3 features with most outliers
    for feature in outlier_counts.head(3)["Feature"]:
        plot_outliers(df, feature)

    # Get highly skewed features for transformation suggestions
    skewed_features = df.columns[abs(df.skew()) > 0.5].tolist()
    if "class" in skewed_features:
        skewed_features.remove("class")

    suggest_transformations(df, skewed_features)

    # 6. Creative exploration
    print("\n\n" + "=" * 30 + " CREATIVE EXPLORATION " + "=" * 30)
    plot_pca(df)
    plot_tsne(df)


if __name__ == "__main__":
    main()
