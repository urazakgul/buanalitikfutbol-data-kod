import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from adjustText import adjust_text
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# https://github.com/urazakgul/super-lig-data-tff

def merge_json_files(directory):
    files = glob.glob(os.path.join(directory, '*.json'))
    return pd.concat([pd.read_json(file, orient='records', lines=True) for file in files], ignore_index=True)

directories = {
    'tff_points_table': './tff_points_table/',
    'tff_matches': './tff_matches/',
}

dataframes = {key: merge_json_files(directory) for key, directory in directories.items()}

tff_points_table_data = dataframes['tff_points_table']
tff_matches_data = dataframes['tff_matches']

season = '23/24'

tff_points_table_data = tff_points_table_data[tff_points_table_data['season'] == season]
tff_points_table_data = tff_points_table_data.reset_index(drop=True)

tff_matches_data = tff_matches_data[tff_matches_data['season'] == season]
tff_matches_data = tff_matches_data.reset_index(drop=True)

teams = pd.concat([tff_matches_data['home_team'], tff_matches_data['away_team']]).unique()

initial_elo = 1200

elo_scores = {team: initial_elo for team in teams}
elo_scores_df = pd.DataFrame(columns=['week'] + list(teams))

def elo_calculation(home_team, away_team, home_score, away_score, k=20):
    home_team_elo = elo_scores.get(home_team)
    away_team_elo = elo_scores.get(away_team)

    dr = home_team_elo - away_team_elo + 100
    win_expectancy_home = 1 / (1 + 10 ** (-dr / 400))
    win_expectancy_away = 1 / (1 + 10 ** (dr / 400))

    if home_score > away_score:
        result_home = 1
        result_away = 0
    elif home_score < away_score:
        result_home = 0
        result_away = 1
    else:
        result_home = 0.5
        result_away = 0.5

    goal_difference = abs(home_score - away_score)
    if goal_difference == 0 or goal_difference == 1:
        k = k
    elif goal_difference == 2:
        k = k * 1.5
    elif goal_difference == 3:
        k = k * 1.75
    elif goal_difference > 3:
        k = k * (1.75 + (goal_difference - 3) / 8)

    elo_scores[home_team] = home_team_elo + k * (result_home - win_expectancy_home)
    elo_scores[away_team] = away_team_elo + k * (result_away - win_expectancy_away)

    return win_expectancy_home, win_expectancy_away

tff_matches_data['home_team_elo_before'] = None
tff_matches_data['away_team_elo_before'] = None
tff_matches_data['home_team_win_expectancy'] = None
tff_matches_data['away_team_win_expectancy'] = None

for week in tff_matches_data['week'].unique():
    week_matches = tff_matches_data[tff_matches_data['week'] == week]
    for idx, match in week_matches.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']

        tff_matches_data.at[idx, 'home_team_elo_before'] = elo_scores[home_team]
        tff_matches_data.at[idx, 'away_team_elo_before'] = elo_scores[away_team]

        home_score = match['home_score']
        away_score = match['away_score']

        win_expectancy_home, win_expectancy_away = elo_calculation(home_team, away_team, home_score, away_score)

        tff_matches_data.at[idx, 'home_team_win_expectancy'] = win_expectancy_home
        tff_matches_data.at[idx, 'away_team_win_expectancy'] = win_expectancy_away

    new_row = pd.DataFrame({'week': [week], **{team: [elo_scores[team]] for team in teams}})
    elo_scores_df = pd.concat([elo_scores_df, new_row], ignore_index=True)

def match_result(row):
    if row['home_score'] > row['away_score']:
        return 'home'
    elif row['home_score'] < row['away_score']:
        return 'away'
    else:
        return 'draw'

def win_expectancy(row):
    if row['home_team_win_expectancy'] > row['away_team_win_expectancy']:
        return 'home'
    elif row['home_team_win_expectancy'] < row['away_team_win_expectancy']:
        return 'away'
    else:
        return 'draw'

tff_matches_data['match_result'] = tff_matches_data.apply(match_result, axis=1)
tff_matches_data['predicted_result'] = tff_matches_data.apply(win_expectancy, axis=1)

filtered_matches = tff_matches_data[tff_matches_data['week'] != 1]
accuracy = (filtered_matches['match_result'] == filtered_matches['predicted_result']).mean()
print(f"Accuracy of Elo win expectancies: {accuracy:.2f}")

filtered_matches = tff_matches_data[tff_matches_data['match_result'] != 'draw']
accuracy = (filtered_matches['match_result'] == filtered_matches['predicted_result']).mean()
print(f"Accuracy of Elo win expectancies (excluding draws): {accuracy:.2f}")

tff_matches_data['win_expectancy_difference'] = tff_matches_data['home_team_win_expectancy'] - tff_matches_data['away_team_win_expectancy']

plt.figure(figsize=(12, 7))
plt.hist(tff_matches_data['win_expectancy_difference'], bins='auto', edgecolor='black')
plt.axvline(x=0.0, color='darkred', linestyle='--', linewidth=1.5)
plt.title(f'Süper Lig {season} Sezonu Kazanma Beklentisi Farklarının Dağılımı')
plt.xlabel('Fark')
plt.ylabel('Frekans')
plt.grid(True)
plt.figtext(
    0.99, -0.07,
    'Hesaplamalar ve Grafik: buanalitikfutbol.com',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    fontstyle='italic',
    color='gray'
)
plt.show()

bins = np.arange(-1, 1.1, 0.1)
tff_matches_data['win_prob_difference_class'] = pd.cut(
    tff_matches_data['win_expectancy_difference'],
    bins=bins,
    include_lowest=True,
    labels=[f'{i:.1f},{i+0.1:.1f}' for i in bins[:-1]]
)

def calculate_accuracy_by_class(df):
    accuracy_by_class = df.groupby('win_prob_difference_class').apply(
        lambda x: (x['match_result'] == x['predicted_result']).mean()
    )
    return accuracy_by_class

accuracy_by_class = calculate_accuracy_by_class(tff_matches_data)

plt.figure(figsize=(12, 8))
accuracy_by_class.plot(kind='bar', edgecolor='black')
plt.axhline(y=0.8, color='darkred', linestyle='--', linewidth=1.5)
plt.title(f'Süper Lig {season} Sezonu Fark Aralıklarına Göre Başarı Oranları')
plt.xlabel('Fark Aralıkları')
plt.ylabel('Başarı')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.figtext(
    0.99, -0.13,
    'Hesaplamalar ve Grafik: buanalitikfutbol.com\nÇubuk olmayan aralıklarda olasılık bulunmamaktadır.',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    fontstyle='italic',
    color='gray'
)
plt.show()

elo_scores_df = elo_scores_df.set_index('week')
elo_scores_df = elo_scores_df - initial_elo
last_row_scores = elo_scores_df.iloc[-1]
sorted_teams = last_row_scores.sort_values(ascending=False).index
sorted_elo_scores_df = elo_scores_df[sorted_teams]

last_week = sorted_elo_scores_df.index.max()
last_week_elo = sorted_elo_scores_df[sorted_elo_scores_df.index == last_week].iloc[0]
elo_difference = last_week_elo[teams]
elo_difference_sorted = elo_difference.sort_values()
teams_sorted = elo_difference_sorted.index
values_sorted = elo_difference_sorted.values
colors_sorted = ['darkblue' if val > 0 else 'darkred' for val in values_sorted]

num_teams = len(sorted_elo_scores_df.columns)
num_cols = 4
num_rows = int(np.ceil(num_teams / num_cols))

right_value = 38

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), sharex=True, sharey=True)
axes = axes.flatten()

for team, ax in zip(sorted_elo_scores_df.columns, axes):
    ax.plot(sorted_elo_scores_df.index, sorted_elo_scores_df[team], label=team, color='black')
    ax.axhline(y=0, color='gray', linewidth=1.5)
    ax.fill_between(
        sorted_elo_scores_df.index,
        sorted_elo_scores_df[team],
        where=(sorted_elo_scores_df[team] > 0),
        color='darkblue',
        alpha=0.2,
        interpolate=True
    )
    ax.fill_between(
        sorted_elo_scores_df.index,
        sorted_elo_scores_df[team],
        where=(sorted_elo_scores_df[team] < 0),
        color='darkred',
        alpha=0.2,
        interpolate=True
    )
    ax.set_title(team)
    ax.set_xlim(left=1, right=right_value)
    ax.grid(True)

for ax in axes[len(sorted_elo_scores_df.columns):]:
    ax.set_visible(False)

fig.text(0.5, -0.01, 'Hafta', ha='center', va='center', fontsize=24)
fig.text(-0.01, 0.5, 'ELO Reyting', ha='center', va='center', rotation='vertical', fontsize=24)
fig.suptitle(f'Süper Lig {season} Sezonu {last_week}. Hafta Takımların ELO Reytingi', fontsize=32)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.figtext(
    0.99, -0.07,
    'Veri: TFF\nHesaplamalar ve Grafik: buanalitikfutbol.com\nReytingler başlangıç değerinden çıkarılmıştır.',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=20,
    fontstyle='italic',
    color='gray'
)
plt.show()

plt.figure(figsize=(10, 15))
bars = plt.barh(teams_sorted, values_sorted, color=colors_sorted, alpha=0.7)
plt.title(f'Süper Lig {season} Sezonu {last_week}. Hafta Takımların ELO Reytingi')
plt.axvline(x=0, color='black', linewidth=1.5)
plt.figtext(
    0.99, -0.03,
    'Veri: TFF\nHesaplamalar ve Grafik: buanalitikfutbol.com\nReytingler başlangıç değerinden çıkarılmıştır.',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    fontstyle='italic',
    color='gray'
)
plt.show()

final_week_elo_scores = sorted_elo_scores_df.iloc[-1].reset_index()
final_week_elo_scores.columns = ['team_name', 'final_week_elo']
final_week_points_table = tff_points_table_data[tff_points_table_data['week'] == 38][['team_name', 'point']]
final_week_data = pd.merge(final_week_elo_scores, final_week_points_table, how='left', left_on='team_name', right_on='team_name')
correlation, _ = pearsonr(final_week_data['final_week_elo'], final_week_data['point'])

plt.figure(figsize=(12, 8))
plt.scatter(
    final_week_data['final_week_elo'],
    final_week_data['point'],
    color='blue',
    alpha=0.5,
    s=200
)
plt.title(f'Süper Lig {season} Sezonu {last_week}. Hafta ELO Reyting ve Puan')
plt.suptitle(f'Pearson Korelasyon: {correlation:.2f}', y=0.85, fontsize=12)
plt.xlabel('ELO Reyting')
plt.ylabel('Puan')
plt.grid(True)

slope, intercept = np.polyfit(final_week_data['final_week_elo'], final_week_data['point'], 1)
x = np.linspace(final_week_data['final_week_elo'].min(), final_week_data['final_week_elo'].max(), 100)
y = slope * x + intercept
plt.plot(x, y, color='darkred', linestyle='--', linewidth=2)

texts = []
for i, row in final_week_data.iterrows():
    texts.append(plt.text(row['final_week_elo'], row['point'], row['team_name'], fontsize=9, ha='right'))
adjust_text(
    texts,
    expand_text=(1.5, 1.5),
    force_text=(0.7, 0.7),
    force_points=(0.3, 0.3),
    arrowprops=dict(arrowstyle='-', color='gray')
)
plt.figtext(
    0.99, -0.09,
    'Veri: TFF\nHesaplamalar ve Grafik: buanalitikfutbol.com\nReytingler başlangıç değerinden çıkarılmıştır.',
    horizontalalignment='right',
    verticalalignment='bottom',
    fontsize=12,
    fontstyle='italic',
    color='gray'
)
plt.show()