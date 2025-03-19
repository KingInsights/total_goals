import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
clf = joblib.load("football_model_goals_scored.pkl")

# Load the CSV file directly when the app starts
df_2024_2025_stats = pd.read_csv('2024-2025-Premier-League_season_stats_1_x_2_rfc.csv')

# Function to get the latest stats for home and away teams
def get_latest_stats(home_team, away_team, df_stats):
    """
    Auto-populates df_predictions with the latest stats for the given teams.
    """
    home_team_stats = df_stats[df_stats["home"] == home_team].sort_values("date", ascending=False).head(1)
    away_team_stats = df_stats[df_stats["away"] == away_team].sort_values("date", ascending=False).head(1)

    if home_team_stats.empty or away_team_stats.empty:
        return None  # Return None if stats are missing for any team

    prediction_row = pd.DataFrame(columns=df_stats.columns)
    prediction_row.loc[0] = home_team_stats.iloc[0]
	
    away_columns = [col for col in df_stats.columns if col.startswith("away_")]
    for col in away_columns:
    	prediction_row.loc[0, col] = away_team_stats.iloc[0][col]

    

    prediction_row["home"] = home_team
    prediction_row["away"] = away_team

    return prediction_row

# Display title
st.title("Total Goals Scored Predictor")

# Extract unique teams for dropdown
home_teams = df_2024_2025_stats["home"].unique()
away_teams = df_2024_2025_stats["away"].unique()
unique_teams = set(home_teams) | set(away_teams)

default_option = "Choose Your Team"

# Team selection dropdowns
home_team = st.selectbox("Select the Home Team:", options=[default_option] + list(unique_teams))
away_team = st.selectbox("Select the Away Team:", options=[default_option] + list(unique_teams))

if home_team == away_team and home_team != default_option:
    st.write("❌ Home and Away Teams are the same, not going to happen")

predict_button = st.button("Predict")

# Prediction logic
if predict_button and home_team != away_team and home_team != default_option and away_team != default_option:
    df_predictions = get_latest_stats(home_team, away_team, df_2024_2025_stats)

    if df_predictions is not None:
        # Exclude unnecessary columns for model prediction
        exclude_cols = ["home_gls_srd", "away_gls_srd", "match_result", 'date', 'wk', 'day', 'day_id',
                        'target_ttl_gls_srd', 'away', 'away_gls_cnd', 'home', 'home_gls_cnd', 'time', 'time_id',
                        'goal_bins']  # ⬅️ ADDED 'goal_bins' HERE

        feature_cols = [col for col in df_predictions.columns if col not in exclude_cols]
        X_pred = df_predictions[feature_cols]

        # Make prediction
        y_pred = clf.predict(X_pred)
        y_proba = clf.predict_proba(X_pred)
        class_labels = clf.classes_

        # Display prediction and bar chart
        st.write(f"Predicted Total Goals for {home_team} vs {away_team}: **{y_pred[0]}**")

        # Convert probabilities to percentages
        probs = y_proba[0] * 100
        top_prob = probs.max()

        # Plot probability distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(class_labels, probs, color='skyblue')
        ax.set_title(f"{home_team} vs {away_team} — Total Goals Probabilities")
        ax.set_xlabel("Goals Scored")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, top_prob + 10)  # ⬅️ Scales to highest probability
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Add percentage labels on bars
        for x, p in zip(class_labels, probs):
            ax.text(x, p + 1, f"{round(p, 1)}%", ha='center', fontsize=9)

        st.pyplot(fig)

    else:
        st.write(f"❌ Missing stats for {home_team} or {away_team}.")
else:
    st.write("Please select different home and away teams and click Predict.")
