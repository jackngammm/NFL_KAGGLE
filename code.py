import pandas as pd
import matplotlib.cm 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

players = pd.read_csv("C://Users//Jack//Downloads//players.csv")
plays = pd.read_csv("C://Users//Jack//Downloads//plays.csv")
games = pd.read_csv("C://Users//Jack//Downloads//games.csv")
player_play = pd.read_csv("C://Users//Jack//Downloads//player_play.csv")

# Display basic information about each dataset
games_info = games.info()
player_play_info = player_play.info()
players_info = players.info()
plays_info = plays.info()

(games_info, player_play_info, players_info, plays_info)

formation_yardage = plays.groupby('offenseFormation')['yardsGained'].mean().sort_values(ascending=False)
print("Average Yardage by Offensive Formation:", formation_yardage)

# Plotting the Average Yardage by Offensive Formation
plt.figure(figsize=(10, 6))
plt.bar(formation_yardage.index, formation_yardage.values)
plt.title("Average Yardage by Offensive Formation")
plt.xlabel("Offensive Formation")
plt.ylabel("Average Yards Gained")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


play_action_plays = plays[plays['playAction'] == True]
team_play_action_effectiveness = play_action_plays.groupby('possessionTeam')['yardsGained'].mean().sort_values(ascending=False)
viridis = matplotlib.cm.get_cmap('viridis', len(team_play_action_effectiveness))
# Plot 1: Average Yards Gained per Play-Action Play by Team
plt.figure(figsize=(12, 6))
colors_viridis = [viridis(i) for i in range(len(team_play_action_effectiveness))]
plt.bar(team_play_action_effectiveness.index, team_play_action_effectiveness.values, color=colors_viridis, edgecolor="black")
plt.title("Average Yards Gained per Play-Action Play by Team", fontsize=18, fontweight="bold")
plt.xlabel("Team", fontsize=14, fontweight="bold")
plt.ylabel("Average Yards Gained", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

alignment_play_action_effectiveness = play_action_plays.groupby('receiverAlignment')['yardsGained'].mean().sort_values(ascending=False)
cividis = matplotlib.cm.get_cmap('cividis', len(alignment_play_action_effectiveness))

plt.figure(figsize=(12, 6))
colors_cividis = [cividis(i) for i in range(len(alignment_play_action_effectiveness))]
plt.bar(alignment_play_action_effectiveness.index, alignment_play_action_effectiveness.values, color=colors_cividis, edgecolor="black")
plt.title("Average Yards Gained per Play-Action Play by Wide Receiver Alignment", fontsize=18, fontweight="bold")
plt.xlabel("Wide Receiver Alignment", fontsize=14, fontweight="bold")
plt.ylabel("Average Yards Gained", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


run_pass_tendency = plays.groupby(['offenseFormation', 'isDropback']).size().unstack().fillna(0)


# Plot 3: Run vs Pass Tendencies by Offensive Formation
plt.figure(figsize=(12, 6))
run_pass_tendency.plot(kind='bar', stacked=True, color=["#FF7F50", "#4682B4"], edgecolor="black")
plt.title("Run vs Pass Tendencies by Offensive Formation", fontsize=18, fontweight="bold")
plt.xlabel("Offensive Formation", fontsize=14, fontweight="bold")
plt.ylabel("Play Count", fontsize=14, fontweight="bold")
plt.legend(["Run Play", "Pass Play"], title="Play Type", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# Specify the format explicitly for parsing gameTimeEastern
games['gameHour'] = pd.to_datetime(games['gameTimeEastern'], format='%H:%M:%S').dt.hour

# Categorize time of day
def categorize_time(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

games['timeOfDay'] = games['gameHour'].apply(categorize_time)

# Calculate the average home and visitor scores for each time category
average_scores = games.groupby('timeOfDay')[['homeFinalScore', 'visitorFinalScore']].mean()

# Plotting the impact of time of day
plt.figure(figsize=(8, 5))
average_scores.plot(kind='bar', color=['orange', 'skyblue'])
plt.title('Impact of Time of Day on Average Scores')
plt.xlabel('Time of Day')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Filter plays for those occurring in the red zone (inside the opponent's 20-yard line)
red_zone_plays = plays[plays['absoluteYardlineNumber'] <= 20]

# Group by offensive formation and calculate average yards gained and average expected points added
formation_stats = red_zone_plays.groupby('offenseFormation').agg(
    averageYardsGained=('yardsGained', 'mean'),
    averageExpectedPointsAdded=('expectedPointsAdded', 'mean')
).reset_index()

# Plotting average yards gained by formation
plt.figure(figsize=(10, 6))
plt.bar(formation_stats['offenseFormation'], formation_stats['averageYardsGained'], color='lightgreen')
plt.title('Average Yards Gained by Formation in the Red Zone')
plt.xlabel('Offensive Formation')
plt.ylabel('Average Yards Gained')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting average expected points added by formation
plt.figure(figsize=(10, 6))
plt.bar(formation_stats['offenseFormation'], formation_stats['averageExpectedPointsAdded'], color='coral')
plt.title('Average Expected Points Added by Formation in the Red Zone')
plt.xlabel('Offensive Formation')
plt.ylabel('Expected Points Added')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the average yards gained for each offemsive formation and coverage concept
avg_yards = plays.groupby(['offenseFormation', 'pff_passCoverage'])['yardsGained'].mean().unstack()

# Heatmap of each combination to see outliers
plt.figure(figsize=(10, 6))
sns.heatmap(avg_yards, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Average Yards Gained'})
plt.title('Average Yards Gained by Offensive and Defensive Formations')
plt.xlabel('Defensive Formation (pff_passCoverage)')
plt.ylabel('Offensive Formation')
plt.show()


# Calculate the average yards gained for each defensive formation
avg_yards_per_coverage = plays.groupby('pff_passCoverage')['yardsGained'].mean().sort_values(ascending=False).reset_index()

# Histogram
plt.figure(figsize=(10, 6))
plt.bar(avg_yards_per_coverage['pff_passCoverage'], avg_yards_per_coverage['yardsGained'], color='skyblue')
plt.xlabel('Defensive Coverage Concept (pff_passCoverage)')
plt.ylabel('Average Yards Gained')
plt.title('Average Yards Gained for Each Defensive Coverage Concept')
plt.xticks(rotation=45, ha='right')
plt.show()


# Calculate the average yards gained for each coverage type (man or zone)
avg_yards_per_manzone = plays.groupby('pff_manZone')['yardsGained'].mean().sort_values(ascending=False).reset_index()

# Histogram
plt.figure(figsize=(8, 5))
plt.bar(avg_yards_per_manzone['pff_manZone'], avg_yards_per_manzone['yardsGained'], color='skyblue')
plt.xlabel('Coverage Type (pff_manZone)')
plt.ylabel('Average Yards Gained')
plt.title('Average Yards Gained by Coverage Type (Man vs. Zone)')
plt.show()


#Play-Action Success Factors: Does the success of play-action plays depend on the down and distance, or is it equally effective across all situations?

# Filter for play-action plays
play_action_plays = plays[plays['playAction'] == True]

# Create distance categories
def categorize_distance(distance):
    if distance <= 3:
        return "Short (1-3 yds)"
    elif distance <= 7:
        return "Medium (4-7 yds)"
    else:
        return "Long (8+ yds)"

play_action_plays['distanceCategory'] = play_action_plays['yardsToGo'].apply(categorize_distance)

# Group by down and distance category to calculate average yards gained
play_action_analysis = play_action_plays.groupby(['down', 'distanceCategory'])['yardsGained'].mean().unstack()

# Plotting the results
plt.figure(figsize=(12, 6))
play_action_analysis.plot(kind='bar', figsize=(10, 6), cmap="viridis")
plt.title("Average Yards Gained on Play-Action Plays by Down and Distance", fontsize=16)
plt.xlabel("Down", fontsize=12)
plt.ylabel("Average Yards Gained", fontsize=12)
plt.legend(title="Distance Category", fontsize=10)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Filter for play-action plays
play_action_plays = plays[plays['playAction'] == True]

# Create distance categories
def categorize_distance(distance):
    if distance <= 3:
        return "Short (1-3 yds)"
    elif distance <= 7:
        return "Medium (4-7 yds)"
    else:
        return "Long (8+ yds)"

play_action_plays['distanceCategory'] = play_action_plays['yardsToGo'].apply(categorize_distance)

# Group by down, distance category, and defensive formation to calculate average yards gained
play_action_defense_analysis = (
    play_action_plays.groupby(['down', 'distanceCategory', 'pff_passCoverage'])['yardsGained']
    .mean()
    .reset_index()
)

# Pivot the data for visualization
pivot_table = play_action_defense_analysis.pivot_table(
    index=['down', 'distanceCategory'],
    columns='pff_passCoverage',
    values='yardsGained',
    aggfunc='mean'
)

# Plotting the results as a heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Average Yards Gained'}
)
plt.title("Play-Action Success by Down, Distance, and Defensive Formation", fontsize=16)
plt.xlabel("Defensive Formation", fontsize=12)
plt.ylabel("Down and Distance Category", fontsize=12)
plt.tight_layout()
plt.show()

# Filter for third-and-long plays
third_and_long_plays = plays[(plays['down'] == 3) & (plays['yardsToGo'] >= 8)]

# Filter for red zone plays
red_zone_plays = plays[plays['absoluteYardlineNumber'] <= 20]

# Group by offensive formation and calculate average yards gained for third-and-long
third_and_long_effectiveness = third_and_long_plays.groupby('offenseFormation')['yardsGained'].mean().sort_values(ascending=False)

# Group by offensive formation and calculate average yards gained for red zone
red_zone_effectiveness = red_zone_plays.groupby('offenseFormation')['yardsGained'].mean().sort_values(ascending=False)

# Plotting third-and-long effectiveness
plt.figure(figsize=(10, 6))
plt.bar(third_and_long_effectiveness.index, third_and_long_effectiveness.values, color='skyblue')
plt.title("Formation Effectiveness on Third-and-Long")
plt.xlabel("Offensive Formation")
plt.ylabel("Average Yards Gained")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting red zone effectiveness
plt.figure(figsize=(10, 6))
plt.bar(red_zone_effectiveness.index, red_zone_effectiveness.values, color='lightgreen')
plt.title("Formation Effectiveness in the Red Zone")
plt.xlabel("Offensive Formation")
plt.ylabel("Average Yards Gained")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Group by offensive and defensive formations, calculate average yards gained
formation_interaction = red_zone_plays.groupby(['offenseFormation', 'pff_passCoverage'])['yardsGained'].mean().unstack()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    formation_interaction, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Avg Yards Gained'}
)
plt.title("Offensive vs Defensive Formations: Average Yards Gained in Red Zone", fontsize=16)
plt.xlabel("Defensive Formation", fontsize=12)
plt.ylabel("Offensive Formation", fontsize=12)
plt.tight_layout()
plt.show()


# Create a success metric based on yardsGained meeting or exceeding yardsToGo
red_zone_plays['isTouchdown'] = red_zone_plays['yardsGained'] >= red_zone_plays['yardsToGo']

# Check the distribution of success metric
print(red_zone_plays['isTouchdown'].value_counts())

# Group by play type and calculate success rates
play_type_success = red_zone_plays.groupby(['isDropback'])['isTouchdown'].mean()

# Map isDropback values to play types
play_type_success.index = play_type_success.index.map({False: 'Run', True: 'Pass'})

# Plot success rates for run vs. pass
plt.figure(figsize=(8, 6))
play_type_success.plot(kind='bar', color=['skyblue', 'orange'], edgecolor='black')
plt.title("Success Rate of Run vs Pass Plays in the Red Zone", fontsize=16)
plt.xlabel("Play Type", fontsize=12)
plt.ylabel("Success Rate (Touchdown %)", fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()



# Filter play-action and non-play-action plays
play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]
non_play_action_plays = red_zone_plays[red_zone_plays['playAction'] == False]

# Calculate success rates
play_action_success = play_action_plays['isTouchdown'].mean()
non_play_action_success = non_play_action_plays['isTouchdown'].mean()

# Combine into a DataFrame
play_action_data = pd.DataFrame({
    'Play Type': ['Play-Action', 'Non-Play-Action'],
    'Success Rate': [play_action_success, non_play_action_success]
})

# Plot side-by-side bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Play Type', y='Success Rate', data=play_action_data, palette=['green', 'gray'])
plt.title("Success Rate of Play-Action vs Non-Play-Action Plays in Red Zone", fontsize=16)
plt.xlabel("Play Type", fontsize=12)
plt.ylabel("Success Rate (Touchdown %)", fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# Group by down, distance category, and play type to calculate success rates

situational_success = (
    red_zone_plays.groupby(['down', 'distanceCategory', 'isDropback'])['isTouchdown']
    .mean()
    .unstack()
)

# Rename columns for better readability
situational_success.columns = ['Run', 'Pass']

# Plot the results as a clustered bar chart
situational_success.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'orange'], edgecolor='black')
plt.title("Success Rate by Down, Distance, and Play Type in Red Zone", fontsize=16)
plt.xlabel("Down and Distance Category", fontsize=12)
plt.ylabel("Success Rate (Touchdown %)", fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.legend(title="Play Type", fontsize=12)
plt.tight_layout()
plt.show()


# Group by team and play type to calculate success rates
team_success = (
    red_zone_plays.groupby(['possessionTeam', 'isDropback'])['isTouchdown']
    .mean()
    .unstack()
)

# Rename columns for better readability
team_success.columns = ['Run Success Rate', 'Pass Success Rate']

# Sort teams by overall success rate (average of run and pass)
team_success['Overall Success Rate'] = team_success.mean(axis=1)
team_success = team_success.sort_values('Overall Success Rate', ascending=False)

# Plot team success rates
team_success.plot(kind='bar', figsize=(14, 8), color=['green', 'blue', 'purple'], edgecolor='black')
plt.title("Red Zone Success Rates by Team and Play Type", fontsize=16)
plt.xlabel("Team", fontsize=12)
plt.ylabel("Success Rate (Touchdown %)", fontsize=12)
plt.legend(title="Metric", fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.tight_layout()
plt.show()


# Group by team and play type; calculate touchdown rate
team_success = (
    red_zone_plays.groupby(['possessionTeam', 'isDropback'])['isTouchdown']
    .mean()
    .reset_index()
)

# Map play types for clarity
team_success['isDropback'] = team_success['isDropback'].map({False: 'Run', True: 'Pass'})

# Pivot for visualization
team_success_pivot = team_success.pivot(index='possessionTeam', columns='isDropback', values='isTouchdown')

# Plot bar chart
team_success_pivot.plot(kind='bar', figsize=(14, 8), color=['skyblue', 'orange'], edgecolor='black')
plt.title("Team-Specific Success Rates in the Red Zone by Play Type", fontsize=16)
plt.xlabel("Team", fontsize=14)
plt.ylabel("Success Rate (Touchdown %)", fontsize=14)
plt.legend(title="Play Type", fontsize=12)
plt.tight_layout()
plt.show()


# Group by team and play type to calculate success rates
team_success = (
    red_zone_plays.groupby(['possessionTeam', 'isDropback'])['isTouchdown']
    .mean()
    .unstack()
)

# Rename columns for clarity
team_success.columns = ['Run Success Rate', 'Pass Success Rate']
team_success['Overall Success Rate'] = team_success[['Run Success Rate', 'Pass Success Rate']].mean(axis=1)
team_success['Team'] = team_success.index
team_success = team_success.sort_values('Overall Success Rate', ascending=True)
team_success['Position'] = range(len(team_success))

# Scatterplot
plt.figure(figsize=(14, 8))
plt.scatter(
    team_success['Overall Success Rate'], team_success['Position'],
    s=300,  
    c=np.where(team_success['Run Success Rate'] > team_success['Pass Success Rate'], 'green', 'blue'), alpha=0.7
)

for i, team in enumerate(team_success['Team']):
    plt.text(
        team_success['Overall Success Rate'].iloc[i] + 0.01,  #
        team_success['Position'].iloc[i],
        team, fontsize=10, va='center'
    )

plt.scatter([], [], c='green', label='Run Success > Pass Success', s=100)
plt.scatter([], [], c='blue', label='Pass Success > Run Success', s=100)
plt.legend(title="Key", fontsize=10, loc='upper left')
plt.title("Red Zone Success Rates by Team (Scatter View)", fontsize=16)
plt.xlabel("Overall Success Rate", fontsize=14)
plt.yticks(team_success['Position'], [])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Group by play type (run/pass) and defensive formation; calculate touchdown rate
play_defense_success = (
    red_zone_plays.groupby(['isDropback', 'pff_passCoverage'])['isTouchdown']
    .mean()
    .reset_index()
)

play_defense_success['isDropback'] = play_defense_success['isDropback'].map({False: 'Run', True: 'Pass'})

play_defense_pivot = play_defense_success.pivot(index='pff_passCoverage', columns='isDropback', values='isTouchdown')

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    play_defense_pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Success Rate'}
)
plt.title("Play Type Success by Defensive Formation in the Red Zone", fontsize=16, pad=20)
plt.xlabel("Play Type", fontsize=14)
plt.ylabel("Defensive Formation", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()



# Group by down, play type, and distance category; calculate touchdown rate
down_play_distance_success = (
    red_zone_plays.groupby(['down', 'isDropback', 'distanceCategory'])['isTouchdown']
    .mean()
    .reset_index()
)

# Map play types for clarity
down_play_distance_success['isDropback'] = down_play_distance_success['isDropback'].map({False: 'Run', True: 'Pass'})

# Pivot for heatmap visualization
down_play_distance_pivot = down_play_distance_success.pivot_table(
    index=['down', 'distanceCategory'], columns='isDropback', values='isTouchdown'
)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    down_play_distance_pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Success Rate'}
)
plt.title("Success Rate by Down, Play Type, and Distance in Red Zone", fontsize=16, pad=20)
plt.xlabel("Play Type", fontsize=14)
plt.ylabel("Down and Distance Category", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Group by formation and distance category, then calculate success rates
formation_distance_success = red_zone_plays.groupby(['offenseFormation', 'distanceCategory'])['isTouchdown'].mean().unstack()

# Plot success rates for offensive formations by distance category
formation_distance_success.plot(kind='bar', figsize=(14, 8), colormap='viridis', edgecolor='black')
plt.title("Formation Effectiveness by Distance Category in Red Zone", fontsize=16)
plt.xlabel("Offensive Formation", fontsize=14)
plt.ylabel("Success Rate (Touchdown %)", fontsize=14)
plt.legend(title="Distance Category", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# Filter for play-action plays in the red zone
red_zone_play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]

# Group by team and calculate average yards gained
team_play_action_yards = red_zone_play_action_plays.groupby('possessionTeam')['yardsGained'].mean().sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x=team_play_action_yards.index, y=team_play_action_yards.values, palette="viridis")
plt.title("Average Yards Gained from Play-Action Plays in the Endzone by Team", fontsize=16)
plt.xlabel("Team", fontsize=14)
plt.ylabel("Average Yards Gained", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Group by defensive formation and calculate touchdown success rates
defense_play_action_success = red_zone_play_action_plays.groupby('pff_passCoverage')['isTouchdown'].mean().sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x=defense_play_action_success.index, y=defense_play_action_success.values, palette="rocket")
plt.title("Defensive Formation Vulnerability to Play-Action in the Endzone", fontsize=16)
plt.xlabel("Defensive Formation", fontsize=14)
plt.ylabel("Success Rate (Touchdown %)", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Group by receiver alignment to calculate average yards gained and touchdown success rates
alignment_analysis = red_zone_play_action_plays.groupby('receiverAlignment').agg(
    avgYardsGained=('yardsGained', 'mean'),
    successRate=('isTouchdown', 'mean')
).sort_values('avgYardsGained', ascending=False)

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot average yards gained on the primary y-axis
color1 = 'tab:blue'
ax1.bar(alignment_analysis.index, alignment_analysis['avgYardsGained'], color=color1, alpha=0.7, label='Avg Yards Gained')
ax1.set_xlabel("Receiver Alignment", fontsize=14)
ax1.set_ylabel("Avg Yards Gained", fontsize=14, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.tick_params(axis='x', rotation=45)
ax1.set_title("Receiver Alignment Impact on Yardage and Touchdown Success in Play-Action Plays", fontsize=16)

# Add a secondary y-axis for success rate
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.plot(alignment_analysis.index, alignment_analysis['successRate'], color=color2, marker='o', linewidth=2, label='Touchdown Success Rate')
ax2.set_ylabel("Touchdown Success Rate (%)", fontsize=14, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()


# Calculate play-action usage rates by team
team_play_action_usage = (
    red_zone_plays.groupby('possessionTeam')['playAction']
    .mean()
    .reset_index()
    .sort_values('playAction', ascending=False)
)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='possessionTeam', y='playAction', data=team_play_action_usage, palette='plasma')
plt.title("Play-Action Usage Rates in the Red Zone by Team", fontsize=16)
plt.xlabel("Team", fontsize=14)
plt.ylabel("Play-Action Usage Rate", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Filter for play-action plays
play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]

# Group by down, calculate average yards gained
down_success = (
    play_action_plays.groupby('down')['yardsGained']
    .mean()
    .reset_index()
)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='down', y='yardsGained', data=down_success, palette='magma')
plt.title("Impact of Play-Action on Yards Gained by Down in the Red Zone", fontsize=16)
plt.xlabel("Down", fontsize=14)
plt.ylabel("Average Yards Gained", fontsize=14)
plt.tight_layout()
plt.show()


# Filter for play-action plays
play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]

# Group by offensive formation, calculate touchdown success rates
formation_success = (
    play_action_plays.groupby('offenseFormation')['isTouchdown']
    .mean()
    .reset_index()
    .sort_values('isTouchdown', ascending=False)
)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='offenseFormation', y='isTouchdown', data=formation_success, palette='coolwarm')
plt.title("Touchdown Success Rate by Offensive Formation in Play-Action Plays in Red Zone", fontsize=16)
plt.xlabel("Offensive Formation", fontsize=14)
plt.ylabel("Touchdown Success Rate", fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Filter for play-action plays
play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]

# Group by receiver alignment and defensive formation, calculate average yards gained
alignment_defense_stats = (
    play_action_plays.groupby(['receiverAlignment', 'pff_passCoverage'])['yardsGained']
    .mean()
    .unstack()
)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    alignment_defense_stats, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Avg Yards Gained'}
)
plt.title("Receiver Alignment vs Defensive Formation: Play-Action Yardage in Red Zone", fontsize=16)
plt.xlabel("Defensive Formation", fontsize=14)
plt.ylabel("Receiver Alignment", fontsize=14)
plt.tight_layout()
plt.show()




# Create distance categories for red zone
red_zone_plays['distanceCategory'] = pd.cut(
    red_zone_plays['absoluteYardlineNumber'], bins=[0, 5, 10, 15, 20], 
    labels=["0-5 yds", "6-10 yds", "11-15 yds", "16-20 yds"]
)

# Filter for play-action plays
play_action_plays = red_zone_plays[red_zone_plays['playAction'] == True]

# Calculate success rate by distance category
distance_success = (
    play_action_plays.groupby('distanceCategory')['isTouchdown']
    .mean()
    .reset_index()
)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='distanceCategory', y='isTouchdown', data=distance_success, palette='viridis')
plt.title("Play-Action Success Rates by Distance to Goal Line", fontsize=16)
plt.xlabel("Distance to Goal Line", fontsize=14)
plt.ylabel("Touchdown Success Rate", fontsize=14)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
