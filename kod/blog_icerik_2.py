import pandas as pd
from mplsoccer import Sbopen
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

match_ids = {
    'Georgia': 3938639,
    'Portugal': 3930174,
    'Czech Republic': 3930184,
    'Austria': 3941022,
    'Netherlands': 3942382
}
match_ids_tr = {
    'Gürcistan': 3938639,
    'Portekiz': 3930174,
    'Çekya': 3930184,
    'Avusturya': 3941022,
    'Hollanda': 3942382
}
translation_dict = {
    'Turkey': 'Türkiye',
    'Georgia': 'Gürcistan',
    'Portugal': 'Portekiz',
    'Czech Republic': 'Çekya',
    'Austria': 'Avusturya',
    'Netherlands': 'Hollanda'
}

parser = Sbopen()

events_list = []

for opponent, match_id in match_ids.items():
    event_data, _, _, _ = parser.event(match_id)
    events_list.append(event_data)

events_df = pd.concat(events_list, ignore_index=True)
events_df['team_name'] = events_df['team_name'].replace(translation_dict)

columns_to_select = ['match_id', 'minute', 'team_name', 'type_name', 'outcome_name', 'shot_statsbomb_xg']
filtered_df = events_df[columns_to_select]

result_df = pd.DataFrame(columns=['match_id', 'minute', 'team_name', 'shot_statsbomb_xg'])

for opponent, match_id in match_ids_tr.items():
    match_df = filtered_df[filtered_df['match_id'] == match_id]

    min_minute = match_df['minute'].min()
    max_minute = match_df['minute'].max()
    minute_df = pd.DataFrame({'minute': range(min_minute, max_minute + 1)})

    match_df = match_df[match_df['type_name'] == 'Shot']

    grouped_df = match_df.groupby(['minute', 'team_name'])['shot_statsbomb_xg'].sum().reset_index()
    grouped_df['match_id'] = match_id

    for team_name in grouped_df['team_name'].unique():
        team_df = grouped_df[grouped_df['team_name'] == team_name]
        merged_df = minute_df.merge(team_df, on='minute', how='left')
        merged_df['team_name'] = team_name
        merged_df['match_id'] = match_id
        merged_df['shot_statsbomb_xg'] = merged_df['shot_statsbomb_xg'].fillna(0)

        result_df = pd.concat([result_df, merged_df], ignore_index=True)

total_cum_xg_df = pd.DataFrame(columns=['match_id', 'team_name', 'cumulative_xg'])

for opponent, match_id in match_ids_tr.items():
    xg_df = result_df[result_df['match_id'] == match_id].copy()
    xg_df['minute'] = xg_df['minute'].astype(int)
    xg_df = xg_df.pivot(index='minute', columns='team_name', values='shot_statsbomb_xg')
    xg_df.columns.name = None

    for team in xg_df.columns:
        xg_df[f'{team}_cumulative'] = xg_df[team].cumsum()

    goals_df = filtered_df[(filtered_df['outcome_name'] == 'Goal') & (filtered_df['match_id'] == match_id)]
    own_goals_df = filtered_df[(filtered_df['type_name'] == 'Own Goal For') & (filtered_df['match_id'] == match_id)]

    team = 'Türkiye'

    plt.figure(figsize=(12, 8))
    plt.plot(xg_df.index, xg_df[f'{team}_cumulative'], color='darkred', label=f'{team} xG')
    plt.plot(xg_df.index, xg_df[f'{opponent}_cumulative'], color='darkblue', label=f'{opponent} xG')
    plt.fill_between(
        xg_df.index, xg_df[f'{team}_cumulative'],
        xg_df[f'{opponent}_cumulative'],
        where=(xg_df[f'{team}_cumulative'] > xg_df[f'{opponent}_cumulative']),
        color='orange',
        alpha=0.3
    )
    plt.fill_between(
        xg_df.index, xg_df[f'{team}_cumulative'],
        xg_df[f'{opponent}_cumulative'],
        where=(xg_df[f'{team}_cumulative'] <= xg_df[f'{opponent}_cumulative']),
        color='gray',
        alpha=0.3
    )

    for _, row in goals_df.iterrows():
        goal_team = row['team_name']
        color = 'darkred' if goal_team == team else 'darkblue'
        plt.scatter(
            row['minute'],
            xg_df.loc[row['minute'], f'{goal_team}_cumulative'] if row['minute'] in xg_df.index else 0,
            color=color,
            marker='o',
            s=300,
            edgecolor='black',
            label=f'{goal_team} Gol' if f'{goal_team} Gol' not in plt.gca().get_legend_handles_labels()[1] else None
        )

    for _, row in own_goals_df.iterrows():
        own_goal_team = opponent if team == row['team_name'] else team
        flag = row['team_name']
        color = 'darkblue' if own_goal_team == team else 'darkred'
        plt.scatter(
            row['minute'],
            xg_df.loc[row['minute'], f'{flag}_cumulative'] if row['minute'] in xg_df.index else 0,
            color=color,
            marker='x',
            s=300,
            edgecolor='black',
            label=f'{own_goal_team} Kendi Kalesine Gol' if f'{own_goal_team} Kendi Kalesine Gol' not in plt.gca().get_legend_handles_labels()[1] else None
        )

    plt.xlabel('Dakika')
    plt.ylabel('Kümülatif xG')
    plt.title(f"Euro 2024 {team} - {opponent} Maçında Üretilen Kümülatif xG")
    plt.legend()
    plt.grid(True)
    plt.figtext(
        0.95, -0.02,
        'Veri: StatsBomb\nGrafik: buanalitikfutbol.com',
        horizontalalignment='right',
        fontsize=10,
        fontstyle='italic',
        color='gray'
    )
    plt.show()

    for team in [team, opponent]:
        final_cumulative_xg = xg_df[f'{team}_cumulative'].iloc[-1]
        total_cum_xg_df = pd.concat([total_cum_xg_df, pd.DataFrame({
            'match_id': [match_id],
            'team_name': [team],
            'cumulative_xg': [final_cumulative_xg]
        })], ignore_index=True)

turkey_xg = total_cum_xg_df[total_cum_xg_df['team_name'] == 'Türkiye'].reset_index(drop=True)
opponents_xg = total_cum_xg_df[total_cum_xg_df['team_name'] != 'Türkiye'].reset_index(drop=True)

merged_xg_df = pd.merge(turkey_xg, opponents_xg, on='match_id', suffixes=('_turkey', '_opponent'))

plt.figure(figsize=(12, 8))
plt.scatter(merged_xg_df['cumulative_xg_turkey'], merged_xg_df['cumulative_xg_opponent'], color='darkblue', s=100)

for i in range(len(merged_xg_df)):
    plt.annotate(
        f"{merged_xg_df['team_name_opponent'][i]}",
        (merged_xg_df['cumulative_xg_turkey'][i], merged_xg_df['cumulative_xg_opponent'][i]),
        textcoords="offset points", xytext=(0,10), ha='center'
    )

plt.xlabel('Türkiye xG')
plt.ylabel('Rakip xG')
plt.title("Euro 2024'te Türkiye ve Rakiplerinin Karşılıklı Ürettikleri Toplam xG")
plt.grid(True)
plt.figtext(
    0.95, -0.02,
    'Veri: StatsBomb\nGrafik: buanalitikfutbol.com',
    horizontalalignment='right',
    fontsize=10,
    fontstyle='italic',
    color='gray'
)

plt.show()

avg_xg_per_match = filtered_df[filtered_df['type_name'] == 'Shot'].groupby(['match_id', 'team_name'])['shot_statsbomb_xg'].mean().reset_index()

avg_xgs = []
for match_id in avg_xg_per_match['match_id'].unique():
    match_data = avg_xg_per_match[avg_xg_per_match['match_id'] == match_id]
    team = 'Türkiye'
    team_xg = match_data[match_data['team_name'] == team]['shot_statsbomb_xg'].values[0]
    opponent = match_data[match_data['team_name'] != team]['team_name'].values[0]
    opponent_xg = match_data[match_data['team_name'] != team]['shot_statsbomb_xg'].values[0]
    avg_xgs.append({
        'match_id': match_id,
        'team': team,
        'team_xg': team_xg,
        'opponent': opponent,
        'opponent_xg': opponent_xg
    })

avg_xgs_df = pd.DataFrame(avg_xgs)
avg_xgs_df['global_average'] = 0.11

ordered_countries = ['Hollanda', 'Avusturya', 'Çekya', 'Portekiz', 'Gürcistan']
my_range = range(1, len(avg_xgs_df.index) + 1)

avg_xgs_df_sorted = avg_xgs_df.set_index('opponent').reindex(ordered_countries).reset_index()

plt.figure(figsize=(8, 8))
plt.hlines(y=my_range, xmin=avg_xgs_df_sorted['team_xg'], xmax=avg_xgs_df_sorted['opponent_xg'], color='grey', alpha=0.4, linestyle='dashed', linewidth=2)
plt.hlines(y=my_range, xmin=avg_xgs_df_sorted['team_xg'], xmax=avg_xgs_df_sorted['global_average'], color='grey', alpha=0.4, linestyle='dashed', linewidth=2)
plt.scatter(avg_xgs_df_sorted['team_xg'], my_range, color='darkred', s=200, label='Türkiye xG')
plt.scatter(avg_xgs_df_sorted['opponent_xg'], my_range, color='darkblue', s=200, label='Rakip xG')
plt.scatter(avg_xgs_df_sorted['global_average'], my_range, color='orange', s=200, label='Global Ortalama xG (0.11)')
plt.yticks(my_range, avg_xgs_df_sorted['opponent'])
plt.title("Euro 2024'te Türkiye ve Rakiplerinin Karşılıklı Ürettikleri Ortalama xG", fontsize=14)
plt.xlabel('Ortalama xG')
plt.ylabel('Rakip')
plt.legend(fontsize='x-small', loc='upper center', frameon=False, bbox_to_anchor=(0.5, -0.09), ncol=3)
plt.figtext(
    0.95, -0.08,
    'Veri: StatsBomb\nGrafik: buanalitikfutbol.com',
    horizontalalignment='right',
    fontsize=10,
    fontstyle='italic',
    color='gray'
)
plt.show()