import numpy as np
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
date = max(dfb.groupby('jour').indices.keys())

# Cumulated hospitalisations and deaths
daily_rad = dfb['rad'].diff().dropna()
daily_hosp = dfb['hosp'].diff().dropna() + daily_rad
total_hosp = daily_hosp.cumsum()
dfbhd = pd.DataFrame({'hosp': total_hosp, 'dc': dfb['dc']})
dfbhd['dc/hosp'] = dfbhd['dc']/dfbhd['hosp']*100
dfbhd = dfbhd.dropna()

# ICU vs hospitalisations
# We can't compute the cumulated number of patients in ICU because we don't have
# the number of people leaving ICU every day. So instead fit rea ~ hosp,
# i.e. daily numbers and store the slope.
def fit_rea_to_hosp(df):
    mod = sm.OLS.from_formula('rea ~ hosp', data=df)
    res = mod.fit()
    return pd.Series({'rea/hosp':res.params['hosp'],
                      'esu':res.bse['hosp'],
                      'R2':res.rsquared})
dfbhr = dfb.groupby(level='dep').apply(fit_rea_to_hosp)
dfbhr = dfbhr.sort_values('rea/hosp')

# Plot hospitalisations vs deaths
current = dfbhd.groupby('dep').tail(1)
mod = sm.OLS.from_formula('dc ~ hosp', data=current)
res = mod.fit()
print(res.summary())
alpha = 0.01
pdc13 = res.get_prediction(current.xs('13')).summary_frame(alpha)
print('\n*** Bouche-du-Rhône (13) ***')
print(f"Décès prédit entre {pdc13['mean_ci_lower'][0]:.0f} et "
      f"{pdc13['mean_ci_upper'][0]:.0f}, à {100*(1-alpha):.0f}% de confiance")
fig = plt.figure()
ax = fig.add_subplot(111)
confidence = 99.99
sns.regplot(ax=ax, data=current,
            x='hosp', y='dc',
            ci=confidence,
            scatter=False,
            label=f'Fit and région à {confidence}% de confiance')
current.plot.scatter(ax=ax,
                     x='hosp', y='dc',
                     marker='.')
for index, row in current.iterrows():
    x, y = row['hosp'], row['dc']
    idx = index[0]
    if idx != '13':
        note = idx
    else:
        note = f"{idx} ({row['dc/hosp']:.1f}% décès)"
    ax.annotate(note, (x, y), textcoords='offset pixels', xytext=(3,3))
ax.set_xlabel('Hospitalisations')
ax.set_ylabel('Décès')
slope = res.params["hosp"]*100
ax.set_title(f'Fit: les décès représentent {slope:.1f}% des hospitalisations'
             f'\n(cumulées au {date})')
ax.legend()

# Plot ICU vs hospitalisations
fig = plt.figure()
ax = fig.add_subplot(111)
dfbhr['rea/hosp'].plot.bar(
    ax=ax,
    yerr=dfbhr['esu'],
    use_index=True,
    color=np.where(np.isin(dfbhr.index.values, ('13', '57', '68')), 'r', 'b'))
for i, p in enumerate(ax.patches):
    p.set_alpha(dfbhr.iloc[i]['R2'])
ax.axhline(dfbhr['rea/hosp'].mean(), color='g')
ax.set_xlabel('Département')
ax.set_ylabel('Réa / Hospitalisation')
ax.set_title(f"Le ratio est la pente d'une régression linéaire entre les "
             f"chiffres journaliers des réa et hospitalisations "
             f"\n(données au {date})")
ax.grid(axis='x')

# Show and that's all folks!
plt.show()
