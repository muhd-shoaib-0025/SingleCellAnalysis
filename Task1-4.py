import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from harmony import harmonize
import seaborn as sns
np.random.seed(42)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset_path', '--dataset_path', type=str, help='Specify the dataset path')
args = parser.parse_args()
dataset_path = args.dataset_path

#Task 1.1. Read the h5ad file in using Scanpy.
#What is the number of cells, number of genes, number of clusters, which column lists disease condition such as IZ, FZ, RZ etc, and number of samples in the dataset?

adata = sc.read(dataset_path)

num_genes = adata.n_vars
num_cells = adata.n_obs

print(f"Number of genes: {num_genes}")
print(f"Number of cells: {num_cells}")

unique_clusters = len(adata.obs['final_cluster'].unique())
print(f'The number of clusters is: {unique_clusters}')

disease_conditions = adata.obs['major_labl'].unique().tolist()
print(f'The disease conditions in major_labl column are: {disease_conditions}')

print('The sample column lists the number of samples.')

random_indices = np.random.randint(0, adata.shape[0], size=10000)
adata = adata[random_indices, :]

#Task 1.2. Is the data raw counts or log-normalized?
#If the data is not log-normalized, please find the layer that contains raw counts and set it as default for further analyses.

if adata.raw is not None:
    print("The data contains raw counts.")
else:
    print("The data is likely log-normalized.")

adata.X = adata.raw.X
print("Raw counts set as default for further analyses.")

#Task 1.3. Plots the UMAP of the data labeled by the cluster names.

cluster_labels = adata.obs['final_cluster']
numeric_labels = pd.Categorical(cluster_labels).codes
umap_result = adata.obsm['X_umap']
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=numeric_labels, cmap='viridis', s=20, alpha=0.5)
plt.title('UMAP of the Data Labeled by Cluster Names')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(numeric_labels)), title='Clusters')
plt.tight_layout()
plt.show()

#Task 1.4. Plot the gene-expression of TTN, RYR2, NEAT1, DCN, and KIT

genes_to_plot = ['TTN', 'RYR2', 'NEAT1', 'DCN', 'KIT']
gene_expression_data = adata[:, genes_to_plot].X.mean(axis=0).tolist()[0]
plt.bar(genes_to_plot, gene_expression_data, color='skyblue')
plt.title(f'Gene Expression of {genes_to_plot}')
plt.ylabel('Mean Expression Level')
plt.xlabel('Genes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Task 1.5. Marker gene discovery – Without changing the data, please find marker genes for each cluster.
# You can use functions in Scanpy to do this. Plot top 5 markers for each cluster as you like e.g. like figure 1.e of the paper

cluster_column = 'final_cluster'
sc.tl.rank_genes_groups(adata, groupby=cluster_column, use_raw=True, n_jobs=-1)
plt.figure(figsize=(15, 10))
for i, cluster in enumerate(adata.obs[cluster_column].unique()):
    top_markers = adata.uns['rank_genes_groups']['names'][cluster][:5]
    plt.subplot(5, 5, i + 1)
    plt.barh(top_markers, adata.uns['rank_genes_groups']['scores'][cluster][:5], color='skyblue')
    plt.title(f'Cluster {cluster}')
    plt.ylabel('Genes')
plt.tight_layout()
plt.show()

#Task 2.1. Sub-set the data on Fibroblasts. Report the results as a UMAP.

cell_type_column = 'cell_type_original'
fibroblast_subset = adata[adata.obs[cell_type_column] == 'Fibroblast']
sc.pp.neighbors(fibroblast_subset, n_neighbors=10, n_pcs=30)
sc.tl.leiden(fibroblast_subset)
sc.tl.umap(fibroblast_subset, random_state=42)
sc.pl.umap(fibroblast_subset, color='leiden', legend_loc='on data', title='Fibroblast Sub-Clustering')

#Task 2.2. Use Harmony to batch-effect correct just the fibroblasts sub-cluster.
#Report the clustering before and after harmony batch-effect correction as UMAP.

batch_column = 'sample'
raw_counts = fibroblast_subset.raw.X.toarray()
harmonized_data = harmonize(raw_counts, fibroblast_subset.obs, batch_column)
fibroblast_subset.obsm['X_harmony'] = harmonized_data
sc.pp.neighbors(fibroblast_subset, n_neighbors=10, n_pcs=30)
sc.tl.leiden(fibroblast_subset)
sc.tl.umap(fibroblast_subset, random_state=42)
sc.pl.umap(fibroblast_subset, color='leiden', title='Fibroblast Sub-Clustering After Harmony')

#Task 2.3. What are the usual steps that one needs to do for batch-effect correction?
#(Hint: look at harmony tutorials, Scanpy-harmony tutorials, MI Human paper, and https://www.nature.com/articles/s41592-021-01336-8)

'''
1. Logarithmize and normalize the raw count data.
2. Identify the column that contains information about experimental conditions.
3. Choose and implement Batch-Effect Correction Method.
4. Recompute dimensionality reduction techniques on batch-effect corrected data.
5. Perform analysis task on the batch-corrected data
6. Assess the effectiveness of batch-effect correction by comparing results before and after correction.
7. Examine the impact on the separation of biological groups and the reduction of technical variation.
'''

#Task 3.1. Perform DE analyses between iz vs control samples, and fz vs control samples using any test available in Scanpy.
#How many genes are differentially expressed after multiple hypothesis correction.

iz_samples = fibroblast_subset.obs['major_labl'].isin(['IZ', 'CTRL'])
fz_samples = fibroblast_subset.obs['major_labl'].isin(['FZ', 'CTRL'])
iz_data = fibroblast_subset[iz_samples].copy()
fz_data = fibroblast_subset[fz_samples].copy()
sc.tl.rank_genes_groups(iz_data, groupby='major_labl', groups=['IZ', 'CTRL'], method='wilcoxon')
sc.tl.rank_genes_groups(fz_data, groupby='major_labl', groups=['FZ', 'CTRL'], method='wilcoxon')
results_iz = iz_data.uns['rank_genes_groups']
genes_iz = list(zip(results_iz['names']['IZ'][results_iz['pvals_adj']['IZ'] < 0.05], results_iz['scores']['IZ'][results_iz['pvals_adj']['IZ'] < 0.05]))
results_fz = fz_data.uns['rank_genes_groups']
genes_fz = list(zip(results_fz['names']['FZ'][results_fz['pvals_adj']['FZ'] < 0.05], results_fz['scores']['FZ'][results_fz['pvals_adj']['FZ'] < 0.05]))
print(f"Differentially expressed genes in IZ vs Control: {len(genes_iz)}")
print(f"Differentially expressed genes in FZ vs Control: {len(genes_fz)}")

#Task 3.2. Plot the top 10 and bottom 10 genes as a single heat map.
#Here the gene names will be on the x-axis and the two tests you conducted on the y-axis.

top_bottom_genes_iz = genes_iz[0:10] + genes_iz[-10:]
top_bottom_genes_fz = genes_fz[0:10] + genes_fz[-10:]
df_iz = pd.DataFrame(top_bottom_genes_iz, columns=['Gene', 'IZ vs CTRL'])
df_fz = pd.DataFrame(top_bottom_genes_fz, columns=['Gene', 'FZ vs CTRL'])
df_combined = pd.merge(df_iz, df_fz, on='Gene', how='outer')
sns.heatmap(df_combined.set_index('Gene').transpose(), cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.xlabel('Genes')
plt.ylabel('Tests')
plt.title('Top and Bottom Genes in IZ and FZ Tests')
plt.tight_layout()
plt.show()

#Task 4.1. Plot the composition of each sample.
#To do this, make a stacked bar plot where each sample is shown with proportion of different cell-types.
#Hint: You can use functions provided in a tool called scCODA for it.

sample_composition = adata.obs.groupby(['sample', 'cell_type_original']).size().unstack(fill_value=0)
sample_composition_percentage = sample_composition.div(sample_composition.sum(axis=1), axis=0) * 100
sample_composition_percentage.plot(kind='bar', stacked=True)
plt.title('Composition of Each Sample')
plt.xlabel('Sample')
plt.ylabel('Percentage')
plt.legend(title='Cell Type')
plt.tight_layout()
plt.show()

#Task 4.2. Do the same as above but instead of per sample, plot the stacked bar plot for each disease condition mentioned in the downloaded data
#(Hint: use the disease column you found in task 1.1)

disease_composition = adata.obs.groupby(['major_labl', 'cell_type_original']).size().unstack(fill_value=0)
disease_composition_percentage = disease_composition.div(disease_composition.sum(axis=1), axis=0) * 100
disease_composition_percentage.plot(kind='bar', stacked=True)
plt.title('Composition for Each Disease Condition')
plt.xlabel('Disease Condition')
plt.ylabel('Percentage')
plt.legend(title='Cell Type')
plt.tight_layout()
plt.show()
