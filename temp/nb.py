import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Project 1 -- DS3021
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Collecting Data
    """)
    return


@app.cell
def _():
    mdf = pd.read_csv('data/lmf_parsed.csv')
    mdf.head()
    return (mdf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sample Code Provided By Prof
    """)
    return


@app.cell
def _(mdf):
    gdf = pd.read_sas("data/DEMO.xpt", format="xport")
    df_merge = gdf.merge(mdf, on="SEQN", how="inner")
    gdf.head()
    return (df_merge,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Find Counts for Missing Values
    There's definitely a better, more efficient way to do this with a variables list but I'm lazy.
    """)
    return


@app.cell
def _(df_merge):
    eligstat_na = df_merge["ELIGSTAT"].isna().values.sum()
    mortstat_na = df_merge["MORTSTAT"].isna().values.sum()
    permth_int_na = df_merge["PERMTH_INT"].isna().values.sum()
    ridageex_na = df_merge["RIDAGEEX"].isna().values.sum()
    dmdeduc2_na = df_merge["DMDEDUC2"].isna().values.sum()
    dmdmartl_na = df_merge["DMDMARTL"].isna().values.sum()
    dmdhhsiz_na = df_merge["DMDHHSIZ"].isna().values.sum()
    indhhinc_na = df_merge["INDHHINC"].isna().values.sum()
    eligstat_na, mortstat_na, permth_int_na, ridageex_na, dmdeduc2_na, dmdmartl_na, dmdhhsiz_na, indhhinc_na
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Remove Rows Irrelevant for Our Purposes
    I'm not entirely sure if everything here *should* be removed, but it would've taken me so much longer if I didn't just exclude everything that was missing any one of the columns. Also `INDHHINC` values are really weird and I didn't want to deal with the weird values (see the codebook images at the end).
    """)
    return


@app.cell
def _(df_merge):
    df = df_merge[["ELIGSTAT", "MORTSTAT", "PERMTH_INT", "RIDAGEEX", "DMDEDUC2", "DMDMARTL", "DMDHHSIZ", "INDHHINC"]].dropna()
    df = df.query("DMDEDUC2 < 7 and DMDMARTL < 77 and INDHHINC < 13")
    df
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preliminary Data Summary
    """)
    return


@app.cell
def _(df):
    skew_data = df[['RIDAGEEX', 'PERMTH_INT', 'DMDHHSIZ']].skew()
    summary_stats = df.describe()
    skew_data, summary_stats, df.columns
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Skewness Check
    """)
    return


@app.cell
def _(df):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df['RIDAGEEX'], kde=True, ax=_axes[0], color='purple').set_title('Age Distribution (Skewness)')
    sns.histplot(df['PERMTH_INT'], kde=True, ax=_axes[1], color='green').set_title('Follow-up Months Distribution (Skewness)')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Outlier Check
    """)
    return


@app.cell
def _(df):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(x='MORTSTAT', y='RIDAGEEX', data=df, ax=_axes[0])
    _axes[0].set_title('Outlier Check: Age vs Mortality')

    sns.boxplot(x='MORTSTAT', y='PERMTH_INT', data=df, ax=_axes[1])
    _axes[1].set_title('Outlier Check: Follow-up Months vs Mortality')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Correlation Matrix
    For the "How correlated are pairs of variables?" question. Used jippity to figure out what kind of visualization to use but I built it myself.
    """)
    return


@app.cell
def _(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Features and Target')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Categorical Contingency Tables
    To fulfill the discovery of "interesting patterns in contingency tables" requirement
    """)
    return


@app.cell
def _(df):
    edu_mort = pd.crosstab(df['DMDEDUC2'], df['MORTSTAT'], normalize='index')
    inc_mort = pd.crosstab(df['INDHHINC'], df['MORTSTAT'], normalize='index')

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(edu_mort, annot=True, cmap='YlGnBu', ax=_axes[0])
    _axes[0].set_title('Mortality Rate by Education Level')
    _axes[0].set_ylabel('Education Code (1-5)')

    sns.heatmap(inc_mort, annot=True, cmap='YlGnBu', ax=_axes[1])
    _axes[1].set_title('Mortality Rate by Income Level')
    _axes[1].set_ylabel('Income Code (1-11)')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ridge Plot Recommended by Prof
    Pain. I know pain.
    """)
    return


@app.cell
def _(df):
    ridge_df = df.copy()
    ridge_df['Age_Years'] = (ridge_df['RIDAGEEX'] / 12).astype(int)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 85]
    labels = [f'{bins[i]}-{bins[i+1]} yrs' for i in range(len(bins)-1)]
    ridge_df['Age_Group'] = pd.cut(ridge_df['Age_Years'], bins=bins, labels=labels)
    ridge_df = ridge_df.dropna(subset=['Age_Group'])

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "font.family": "sans-serif"})
    pal = sns.cubehelix_palette(len(labels), rot=-.25, light=.7)

    g = sns.FacetGrid(ridge_df, row="Age_Group", hue="Age_Group", aspect=12, height=1.2, palette=pal)

    g.map(sns.kdeplot, "PERMTH_INT", bw_adjust=.6, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "PERMTH_INT", clip_on=False, color="w", lw=2, bw_adjust=.6)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "PERMTH_INT")

    g.figure.subplots_adjust(hspace=-.65)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    plt.xlabel("Months of Follow-up (PERMTH_INT)", fontsize=12, fontweight='bold')
    g.fig.suptitle("Follow-up Time Distribution by Age Cohort", fontsize=16, fontweight='bold', y=0.98)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Codebooks
    ![](./public/codebook1.png)
    ![alt](public/codebook2.png)
    ![alt](public/codebook3.png)
    ![alt](public/codebook4.png)
    ![alt](public/codebook5.png)
    ![alt](public/codebook6.png)
    ![alt](public/codebook7.png)
    """)
    return


if __name__ == "__main__":
    app.run()
