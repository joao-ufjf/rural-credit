# Basic imports
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Kmeans imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Save/load models
from joblib import dump, load

def load_and_prepare_data(scaled_data_path="3_data_with_predictions.csv"):
    """Load scaled data and prepare features"""
    df = pd.read_csv(scaled_data_path)
    df = df.dropna()
    
    # Features for clustering
    features = ["cost", "productivity", "value", "area"]
    X = df[features]
    
    # Store the original (unscaled) data
    original_data_path = "1_data_to_scale.csv"
    original_df = pd.read_csv(original_data_path)
    original_df = original_df.dropna()
    
    return X, df, original_df[features]

def create_kmeans_model(n_clusters, random_state=42):
    """Create KMeans with optimized parameters"""
    return KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=random_state,
        tol=1e-4
    )

def plot_cluster_distribution(labels, k, output_dir):
    """Plot bar chart showing the number of items in each cluster"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts)
    plt.title(f"Distribution of Items per Cluster (k={k})")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Items")
    
    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.xticks(unique_labels)
    plt.savefig(f"{output_dir}/cluster_distribution_k{k}.png")
    plt.close()

def plot_3d_clusters(X, labels, feature_names, k, output_dir):
    """Create 3D scatter plot of the clusters"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=labels,
        cmap='viridis',
        alpha=0.6
    )
    
    # Set labels
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    
    # Add title and colorbar
    plt.title(f'3D Cluster Visualization (k={k})', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_3d_k{k}.png")
    plt.close()

def clean_directory(directory):
    """Remove all files in the specified directory"""
    path = Path(directory)
    if path.exists():
        for file in path.glob('*'):
            file.unlink()
    else:
        path.mkdir(parents=True)

def train_and_evaluate_model(X, df, k, output_dir, models_dir, feature_names):
    """Train and evaluate a KMeans model with k clusters"""
    print(f"\nTraining model with k={k}")
    
    # Create and fit model
    kmeans = create_kmeans_model(k)
    kmeans.fit(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Plot visualizations
    plot_cluster_distribution(kmeans.labels_, k, output_dir)
    plot_3d_clusters(X[feature_names].values, kmeans.labels_, feature_names, k, output_dir)
    
    # Save cluster centers
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X.columns
    )
    cluster_centers.to_csv(f"{output_dir}/cluster_centers_k{k}.csv")
    
    # Save model in models directory
    dump(kmeans, f"{models_dir}/k{k}.joblib")
    
    # Create DataFrame with results
    df_result = df.copy()
    df_result['cluster'] = kmeans.labels_
    df_result.to_csv(f"{output_dir}/clustered_data_k{k}.csv", index=False)
    
    return silhouette_avg

def create_combined_distribution_plot(k_values, output_dir):
    """Create a combined plot showing cluster distributions for all k values"""
    # Calculate number of rows and columns for subplots
    n_plots = len(k_values)
    n_cols = min(3, n_plots)  # Maximum 3 plots per row
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Iterate through each k value and create subplot
    for idx, k in enumerate(k_values):
        row = idx // n_cols
        col = idx % n_cols
        
        # Load the labels from the saved clustered data
        df = pd.read_csv(f"{output_dir}/clustered_data_k{k}.csv")
        labels = df['cluster']
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        axes[row, col].bar(unique_labels, counts)
        axes[row, col].set_title(f"k={k}")
        axes[row, col].set_xlabel("Cluster")
        axes[row, col].set_ylabel("Number of Items")
        
        # Add value labels on top of each bar
        for i, count in enumerate(counts):
            axes[row, col].text(i, count, str(count), ha='center', va='bottom')
        
        axes[row, col].set_xticks(unique_labels)
    
    # Remove empty subplots if any
    for idx in range(len(k_values), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle("Cluster Size Distribution Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_cluster_distributions.png", bbox_inches='tight')
    plt.close()

def create_elbow_plot(X, k_values, output_dir):
    """Create and save elbow plot using KElbowVisualizer"""
    plt.figure(figsize=(10, 6))
    
    # Create visualizer
    visualizer = KElbowVisualizer(
        KMeans(init="k-means++", n_init=10, max_iter=300, random_state=42),
        k=k_values,
        metric='distortion',
        timings=False
    )
    
    # Fit the data
    visualizer.fit(X)
    
    # Save the elbow plot
    visualizer.show(outpath=f"{output_dir}/elbow_plot.png")
    plt.close()
    
    return visualizer.elbow_value_

def create_cluster_profiles(k, output_dir, original_data):
    """Create detailed profiles for each cluster based on their centers and characteristics"""
    # Read cluster centers and data
    centers = pd.read_csv(f"{output_dir}/cluster_centers_k{k}.csv", index_col=0)
    data = pd.read_csv(f"{output_dir}/clustered_data_k{k}.csv")
    
    profiles = []
    
    for cluster in range(k):
        # Get cluster center values (original scale)
        center = centers.iloc[cluster]
        cluster_data = data[data['cluster'] == cluster]
        
        # Get original data for this cluster
        cluster_mask = data['cluster'] == cluster
        original_cluster_data = original_data[cluster_mask]
        
        # Calculate statistics for this cluster using original scale
        stats = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(data)) * 100,
            'center': center.to_dict(),  # Assuming centers are already in original scale
            'min': original_cluster_data.min().to_dict(),
            'max': original_cluster_data.max().to_dict(),
            'std': original_cluster_data.std().to_dict()
        }
        
        # Create profile description
        profile = {
            'cluster': cluster + 1,
            'stats': stats,
            'description': _generate_cluster_description(stats)
        }
        
        profiles.append(profile)
    
    # Save profiles to JSON
    with open(f"{output_dir}/cluster_profiles_k{k}.json", 'w') as f:
        json.dump(profiles, f, indent=4)
    
    return profiles

def _generate_cluster_description(stats):
    """Generate a human-readable description of the cluster based on its statistics"""
    center = stats['center']
    
    # Format values based on their scale
    def format_value(key, value):
        if key == 'cost':
            return f"${value:,.2f}"
        elif key == 'area':
            return f"{value:,.0f} sq ft"
        else:
            return f"{value:.2f}"
    
    description = f"This cluster contains {stats['size']} items ({stats['percentage']:.1f}% of total) and is characterized by:\n"
    
    for metric in ['cost', 'productivity', 'value', 'area']:
        center_val = format_value(metric, center[metric])
        min_val = format_value(metric, stats['min'][metric])
        max_val = format_value(metric, stats['max'][metric])
        
        description += f"- {metric.capitalize()}: {center_val} (range: {min_val} to {max_val})\n"
    
    return description

def _create_markdown_report(profiles, k, output_dir):
    """Create a markdown report with cluster profiles"""
    report = f"# Cluster Analysis Report (k={k})\n\n"
    
    for profile in profiles:
        report += f"## Cluster {profile['cluster']}\n\n"
        report += f"{profile['description']}\n\n"
        report += "### Detailed Statistics\n\n"
        
        stats = profile['stats']
        report += "#### Center Values\n"
        for metric, value in stats['center'].items():
            report += f"- {metric}: {value:.3f}\n"
        
        report += "\n#### Ranges\n"
        for metric in stats['min'].keys():
            report += f"- {metric}: {stats['min'][metric]:.3f} to {stats['max'][metric]:.3f} (std: {stats['std'][metric]:.3f})\n"
        
        report += "\n---\n\n"
    
    with open(f"{output_dir}/cluster_profiles_k{k}.md", 'w') as f:
        f.write(report)

def main():
    # Setup directories
    output_dir = "kmeans_results"
    models_dir = "models/kmeans"
    
    # Clean and create directories
    clean_directory(output_dir)
    clean_directory(models_dir)
    
    # Load scaled data and original data
    X, df, original_data = load_and_prepare_data()
    feature_names = ["cost", "productivity", "value"]
    
    # Train models for different k values
    k_values = [i for i in range(2, 11)]
    results = []
    
    # Create elbow plot
    optimal_k = create_elbow_plot(X, k_values, output_dir)
    print(f"\nOptimal k according to elbow method: {optimal_k}")
    
    for k in k_values:
        silhouette = train_and_evaluate_model(X, df, k, output_dir, models_dir, feature_names)
        profiles = create_cluster_profiles(k, output_dir, original_data)
        results.append({
            'k': k,
            'silhouette_score': silhouette
        })
    
    # Create combined distribution plot
    create_combined_distribution_plot(k_values, output_dir)
    
    # Save summary of results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/kmeans_summary.csv", index=False)
    print("\nClustering Summary:")
    print(results_df)

if __name__ == "__main__":
    main()