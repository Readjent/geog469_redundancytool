import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('FullData.csv')

df_clean = df.dropna()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

corr_matrix = df_clean.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
plt.title('ADI Variables Correlation Map')
plt.savefig('adi_correlation_heatmap.png')

vif_data = pd.DataFrame()
vif_data["Feature"] = df_clean.columns
vif_data["VIF"] = [variance_inflation_factor(df_clean.values, i) for i in range(len(df_clean.columns))]
vif_data.to_csv('vif_values.csv', index=False)
