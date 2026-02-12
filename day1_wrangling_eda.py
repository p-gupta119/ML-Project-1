import pandas as pd
import matplotlib.pyplot as plt

mdf = pd.read_csv('data/lmf_parsed.csv') # Load mortality file
print( mdf.head() )
gdf = pd.read_sas("data/DEMO.xpt", format="xport") # Load demographics file
print( gdf.head() )
df = gdf.merge(mdf, on="SEQN", how="inner") # Merge mortality and demographics on SEQN variable

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].bar([1, 2, 3], df['ELIGSTAT'].value_counts().sort_index(), color='#3498db', edgecolor='white')
axes[0, 0].set_xticks([1, 2, 3])
axes[0, 0].set_title('ELIGSTAT')
axes[0, 0].set_xlabel('Eligibility Status')

axes[0, 1].bar([0, 1], df['MORTSTAT'].dropna().value_counts().sort_index(), color='#e74c3c', edgecolor='white')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_title('MORTSTAT')
axes[0, 1].set_xlabel('Mortality Status')

axes[1, 0].hist(df['PERMTH_INT'].dropna(), bins=50, color='#2ecc71', edgecolor='white')
axes[1, 0].set_title('PERMTH_INT')
axes[1, 0].set_xlabel('Person-Months of Follow-up')

axes[1, 1].hist(df['RIDAGEEX'].dropna(), bins=50, color='#9b59b6', edgecolor='white')
axes[1, 1].set_title('RIDAGEEX')
axes[1, 1].set_xlabel('Age in Months at Examination')

for ax in axes.flat:
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histograms.png', dpi=150)
plt.show()