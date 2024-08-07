import os
import glob
import pandas as pd
from adjustText import adjust_text
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# https://github.com/urazakgul/super-lig-data-footystats
# https://github.com/urazakgul/super-lig-data-tff

def merge_json_files(directory):
    files = glob.glob(os.path.join(directory, '*.json'))
    return pd.concat([pd.read_json(file, orient='records', lines=True) for file in files], ignore_index=True)

directories = {
    'footystats_xg': './footystats_xg/',
    'tff_points_table': './tff_points_table/'
}

dataframes = {key: merge_json_files(directory) for key, directory in directories.items()}

footystats_xg_data = dataframes['footystats_xg']
footystats_xg_data = footystats_xg_data[['season','mp','team','xg','xga']]

last_season = '23/24'
footystats_xg_data = footystats_xg_data[footystats_xg_data['season'] == last_season]
last_week = footystats_xg_data[footystats_xg_data['season'] == last_season]['mp'].max()

footystats_xg_data = footystats_xg_data[(footystats_xg_data['season'] == last_season) & (footystats_xg_data['mp'] == last_week)]
footystats_xg_data['xg_xga'] = footystats_xg_data['xg'] - footystats_xg_data['xga']

tff_points_table_data = dataframes['tff_points_table']
tff_points_table_data = tff_points_table_data[['season','week','team_name','goal_difference']]
tff_points_table_data = tff_points_table_data[(tff_points_table_data['season'] == last_season) & (tff_points_table_data['week'] == last_week)]

merged_data = pd.merge(
    footystats_xg_data,
    tff_points_table_data,
    left_on=['season','mp','team'],
    right_on=['season','week','team_name']
)

footystats_xg_data = footystats_xg_data.copy()
footystats_xg_data.loc[:, 'xg_minus_xga'] = footystats_xg_data['xg'] - footystats_xg_data['xga']
cmap = plt.get_cmap('coolwarm')
norm = plt.Normalize(footystats_xg_data['xg_minus_xga'].min(), footystats_xg_data['xg_minus_xga'].max())

plt.figure(figsize=(12, 8))

plt.scatter(
    footystats_xg_data[footystats_xg_data['season'] == last_season]['xg'],
    footystats_xg_data[footystats_xg_data['season'] == last_season]['xga'],
    c=footystats_xg_data[footystats_xg_data['season'] == last_season]['xg_minus_xga'],
    cmap=cmap,
    norm=norm,
    s=200,
    alpha=.5
)

mean_xg = footystats_xg_data['xg'].mean()
mean_xga = footystats_xg_data['xga'].mean()

plt.axhline(y=mean_xga, color='darkred', linestyle='--', linewidth=1.5, label='Ortalama xGA')
plt.axvline(x=mean_xg, color='darkblue', linestyle='--', linewidth=1.5, label='Ortalama xG')

texts = []
for _, row in footystats_xg_data[footystats_xg_data['season'] == last_season].iterrows():
    texts.append(plt.text(row['xg'], row['xga'], row['team'], fontsize=10))
adjust_text(texts, only_move={'xga': 'y', 'texts': 'xy'}, arrowprops=dict(arrowstyle='->', color='black'))

plt.xlabel('xG')
plt.ylabel('xGA')
plt.title(f'{last_season} Sezonu {last_week}. Hafta xG ve xGA Durumu')
plt.grid(True)
plt.legend(
    fontsize='small',
    bbox_to_anchor=(0.5, -0.1),
    loc='upper center',
    ncol=4
)
plt.figtext(
    0.99, -0.15,
    'Veri: FootyStats\nGrafik: buanalitikfutbol.com\n',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=10,
    fontstyle='italic',
    color='gray'
)
plt.text(
    x=footystats_xg_data['xg'].max(),
    y=footystats_xg_data['xga'].min() - 0.1,
    s='İyi Defans-İyi Hücum',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=12,
    bbox=dict(facecolor='none', edgecolor='none')
)
plt.text(
    x=footystats_xg_data['xg'].min(),
    y=footystats_xg_data['xga'].min() - 0.1,
    s='İyi Defans-Kötü Hücum',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=12,
    bbox=dict(facecolor='none', edgecolor='none')
)
plt.text(
    x=footystats_xg_data['xg'].max(),
    y=footystats_xg_data['xga'].max() + 0.1,
    s='Kötü Defans-İyi Hücum',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=12,
    bbox=dict(facecolor='none', edgecolor='none')
)
plt.text(
    x=footystats_xg_data['xg'].min(),
    y=footystats_xg_data['xga'].max() + 0.1,
    s='Kötü Defans-Kötü Hücum',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=12,
    bbox=dict(facecolor='none', edgecolor='none')
)
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(
    merged_data['xg_xga'],
    merged_data['goal_difference'],
    s=200,
    alpha=.5,
    color='darkblue'
)

X = merged_data[['xg_xga']].values
y = merged_data['goal_difference'].values

regressor = LinearRegression()
regressor.fit(X, y)
predicted_y = regressor.predict(X)

plt.plot(
    merged_data['xg_xga'],
    predicted_y,
    color='red',
    linewidth=2
)

texts = []
for _, row in merged_data.iterrows():
    texts.append(plt.text(row['xg_xga'], row['goal_difference'], row['team'], fontsize=10))
adjust_text(texts, only_move={'xg_xga': 'y', 'texts': 'xy'}, arrowprops=dict(arrowstyle='->', color='black'))

plt.xlabel('xG-xGA')
plt.ylabel('Averaj')
plt.title(f'{last_season} Sezonu {last_week}. Hafta xG-xGA ve Averaj İlişkisi')

corr, _ = pearsonr(merged_data['xg_xga'], merged_data['goal_difference'])

plt.suptitle(
    f'ρ: {corr:.2f}',
    fontsize=14,
    y=0.85
)

plt.figtext(
    0.99, -0.1,
    'Veri: FootyStats ve TFF\nHesaplamalar ve Grafik: buanalitikfutbol.com\n',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=10,
    fontstyle='italic',
    color='gray'
)
plt.show()