import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

BOTH_SEX, MALE, FEMALE = 0, 1, 2

# Load region/départements nomenclature
depts = pd.read_csv('depts2018.txt', encoding='latin-1', sep='\t')
regions = pd.read_csv('reg2018.txt', encoding='latin-1', sep='\t')
depname = lambda depnum: depts.query(f"DEP=='{dep}'")['NCCENR'].iloc[0]
depnamenum = lambda depnum: f"""{depname(depnum)} ({depnum})"""

# Load hospitals data
df = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/'
                 '63352e38-d353-4b54-bfd1-f1b3ee1cabd7',
                 sep=';')
meta = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/'
                   '3f0f1885-25f4-4102-bbab-edec5a58e34a',
                   sep=';')

# Aggregate by "départements"
male = df['sexe'] == BOTH_SEX
dfb = df[male].drop(['sexe'], axis=1).groupby(['dep', 'jour']).sum()

# Cumulated hospitalisations and deaths
daily_rad = dfb['rad'].diff().dropna()
daily_hosp = dfb['hosp'].diff().dropna() + daily_rad
total_hosp = daily_hosp.cumsum()
dfbhd = pd.DataFrame({'hosp': total_hosp, 'dc': dfb['dc']})
dfbhd['dc/hosp'] = dfbhd['dc']/dfbhd['hosp']*100
dfbhd = dfbhd.dropna()

# Plot
dfbhd1 = dfbhd.groupby(level='dep').sum()
y, X = dmatrices('dc ~ hosp', data=dfbhd1, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
fig = plt.figure()
ax = fig.add_subplot(111)
sns.regplot(x='hosp', y='dc', data=dfbhd1, ax=ax,
            marker='.',
            label='Fit and région à 95% de confiance')
for index, row in dfbhd1.iterrows():
    x, y = row['hosp'], row['dc']
    if index != '13':
        note = index
    else:
        ratio = row['dc']/row['hosp']*100
        note = f"{index} ({ratio:.1f}% décès)"
    ax.annotate(note, (x, y), textcoords='offset pixels', xytext=(3,3))
ax.set_xlabel('Hospitalisations')
ax.set_ylabel('Décès')
slope = res.params["hosp"]*100
ax.set_title(f'Fit: les décès représentent {slope:.1f}% des hospitalisations')
plt.show()


# Show and that's all folks!
plt.show()
