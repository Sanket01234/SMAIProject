import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from train import process_match_chunk

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model_info = joblib.load(model_path)
    return model_info['model'], model_info['scaler'], model_info['optimal_threshold'], model_info['feature_names']

def load_and_prepare_data(ball_by_ball_path, match_info_path):
    print("Loading data...")
    ball_by_ball = pd.read_csv(ball_by_ball_path)
    match_info = pd.read_csv(match_info_path)
    match_info['date'] = pd.to_datetime(match_info['date'])
    return pd.merge(ball_by_ball, match_info[['match_id', 'date', 'venue']], on='match_id')

def precompute_feature_dfs(data):
    print("Precomputing historical features...")
    feature_dfs = {}
    
    pvp_stats = []
    for (batsman, bowler), group in data.groupby(['batter', 'bowler']):
        balls_faced = len(group)
        runs_scored = group['batsman_runs'].sum()
        dismissals = group[(group['is_wicket'] == 1) & (group['player_dismissed'] == batsman)].shape[0]
        strike_rate = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
        
        pvp_stats.append({
            'batsman': batsman,
            'bowler': bowler,
            'balls_faced': balls_faced,
            'runs_scored': runs_scored,
            'dismissals': dismissals,
            'strike_rate': strike_rate
        })
    
    feature_dfs['player_vs_player'] = pd.DataFrame(pvp_stats)
    
    bat_venue_stats = []
    for (batsman, venue), group in data.groupby(['batter', 'venue']):
        balls_faced = len(group)
        runs_scored = group['batsman_runs'].sum()
        dismissals = group[(group['is_wicket'] == 1) & (group['player_dismissed'] == batsman)].shape[0]
        average = runs_scored / dismissals if dismissals > 0 else runs_scored
        strike_rate = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
        
        bat_venue_stats.append({
            'batsman': batsman,
            'venue': venue,
            'balls_faced': balls_faced,
            'runs_scored': runs_scored,
            'dismissals': dismissals,
            'average': average,
            'strike_rate': strike_rate
        })
    
    feature_dfs['batsman_venue'] = pd.DataFrame(bat_venue_stats)
    
    bowl_venue_stats = []
    for (bowler, venue), group in data.groupby(['bowler', 'venue']):
        balls_bowled = len(group)
        runs_conceded = group['total_runs'].sum()
        wickets = group['is_wicket'].sum()
        bowling_average = runs_conceded / wickets if wickets > 0 else runs_conceded
        bowling_strike_rate = balls_bowled / wickets if wickets > 0 else balls_bowled
        
        bowl_venue_stats.append({
            'bowler': bowler,
            'venue': venue,
            'balls_bowled': balls_bowled,
            'runs_conceded': runs_conceded,
            'wickets': wickets,
            'bowling_average': bowling_average,
            'bowling_strike_rate': bowling_strike_rate
        })
    
    feature_dfs['bowler_venue'] = pd.DataFrame(bowl_venue_stats)
    
    data['phase'] = data['over'].apply(
        lambda x: 'Phase1_0-5' if x < 6 else ('Phase2_6-14' if x < 15 else 'Phase3_15-19'))
    
    bat_phase_stats = []
    for (batsman, phase), group in data.groupby(['batter', 'phase']):
        balls_faced = len(group)
        runs_scored = group['batsman_runs'].sum()
        dismissals = group[(group['is_wicket'] == 1) & (group['player_dismissed'] == batsman)].shape[0]
        average = runs_scored / dismissals if dismissals > 0 else runs_scored
        strike_rate = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
        
        bat_phase_stats.append({
            'batsman': batsman,
            'phase': phase,
            'balls_faced': balls_faced,
            'runs_scored': runs_scored,
            'dismissals': dismissals,
            'average': average,
            'strike_rate': strike_rate
        })
    
    feature_dfs['batsman_phase'] = pd.DataFrame(bat_phase_stats)
    
    bowl_phase_stats = []
    for (bowler, phase), group in data.groupby(['bowler', 'phase']):
        balls_bowled = len(group)
        runs_conceded = group['total_runs'].sum()
        wickets = group['is_wicket'].sum()
        bowling_average = runs_conceded / wickets if wickets > 0 else runs_conceded
        bowling_strike_rate = balls_bowled / wickets if wickets > 0 else balls_bowled
        
        bowl_phase_stats.append({
            'bowler': bowler,
            'phase': phase,
            'balls_bowled': balls_bowled,
            'runs_conceded': runs_conceded,
            'wickets': wickets,
            'bowling_average': bowling_average,
            'bowling_strike_rate': bowling_strike_rate
        })
    
    feature_dfs['bowler_phase'] = pd.DataFrame(bowl_phase_stats)
    
    bat_vs_team_stats = []
    for (batsman, opposing_team), group in data.groupby(['batter', 'bowling_team']):
        balls_faced = len(group)
        runs_scored = group['batsman_runs'].sum()
        dismissals = group[(group['is_wicket'] == 1) & (group['player_dismissed'] == batsman)].shape[0]
        average = runs_scored / dismissals if dismissals > 0 else runs_scored
        strike_rate = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
        
        bat_vs_team_stats.append({
            'batsman': batsman,
            'opposing_team': opposing_team,
            'balls_faced': balls_faced,
            'runs_scored': runs_scored,
            'dismissals': dismissals,
            'average': average,
            'strike_rate': strike_rate
        })
    
    feature_dfs['batsman_vs_team'] = pd.DataFrame(bat_vs_team_stats)
    
    bowl_vs_team_stats = []
    for (bowler, opposing_team), group in data.groupby(['bowler', 'batting_team']):
        balls_bowled = len(group)
        runs_conceded = group['total_runs'].sum()
        wickets = group['is_wicket'].sum()
        bowling_average = runs_conceded / wickets if wickets > 0 else runs_conceded
        bowling_strike_rate = balls_bowled / wickets if wickets > 0 else balls_bowled
        
        bowl_vs_team_stats.append({
            'bowler': bowler,
            'opposing_team': opposing_team,
            'balls_bowled': balls_bowled,
            'runs_conceded': runs_conceded,
            'wickets': wickets,
            'bowling_average': bowling_average,
            'bowling_strike_rate': bowling_strike_rate
        })
    
    feature_dfs['bowler_vs_team'] = pd.DataFrame(bowl_vs_team_stats)

    return feature_dfs

def process_single_delivery(match_id, over, ball,inning, data_paths):
    # Load full dataset
    data = load_and_prepare_data(data_paths['deliveries'], data_paths['matches'])
    
    # Get context for this specific match
    match_data = data[data['match_id'] == match_id].copy()
    if match_data.empty:
        raise ValueError(f"Match {match_id} not found in dataset")
    
    # Precompute all historical features
    feature_dfs = precompute_feature_dfs(data)
    
    # Process match to get delivery-level features
    match_data = match_data.sort_values(['inning', 'over', 'ball'])
    features, _, _ = process_match_chunk(match_data, feature_dfs)
    
    # Find our specific delivery
    for f in features:
        if (f['inning'] == inning and 
            f['over'] == over and 
            f['ball'] == ball):
            return f
    
    raise ValueError("Delivery not found in processed features")

def get_feature_df(features, feature_names):
    df = pd.DataFrame([features])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_names]

def predict_wicket_probability(model, scaler, threshold, features, feature_names):
    X = pd.DataFrame([features])[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    return proba, proba >= threshold

def plot_wicket_probability_gauge(probability, threshold, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    gauge_height = 0.25
    
    ax.add_patch(Rectangle((0, 0.3), 1, gauge_height, facecolor='lightgray', edgecolor='gray'))
    ax.add_patch(Rectangle((0, 0.3), probability, gauge_height, facecolor='red', alpha=0.7))
    
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
    
    ax.text(threshold, 0.6, f'Threshold\n{threshold:.3f}', ha='center', va='center', fontsize=12)
    ax.text(probability, 0.15, f'Wicket Probability: {probability:.3f}', ha='center', va='center', fontsize=14, weight='bold')
    
    if probability >= threshold:
        verdict = "HIGH WICKET ALERT!"
    else:
        verdict = "Low wicket probability"
    
    ax.text(0.5, 0.8, verdict, ha='center', va='center', fontsize=16, weight='bold', 
            color='darkred' if probability >= threshold else 'darkgreen')
    
    return ax

def format_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:10]]
    
    return top_features

def analyze_key_factors(features, model, feature_names):
    importances = model.feature_importances_
    feature_imp_dict = dict(zip(feature_names, importances))
    
    top_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    feature_values = []
    for feature_name, importance in top_features:
        if feature_name in features:
            feature_values.append((feature_name, features[feature_name], importance))
    
    return feature_values

def main():
    parser = argparse.ArgumentParser(description='Predict wicket probability from raw match data')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--deliveries', default='deliveries.csv', help='Path to deliveries data')
    parser.add_argument('--matches', default='matches.csv', help='Path to matches data')
    parser.add_argument('--match_id', type=int, required=True, help='Match ID')
    parser.add_argument('--over', type=int, required=True, help='Over number')
    parser.add_argument('--ball', type=int, required=True, help='Ball number')
    parser.add_argument('--inning', type=int, required=True, help='Inning Number')
    parser.add_argument('--output', default='prediction.png', help='Output image path')
    
    args = parser.parse_args()

    # Load model
    model, scaler, threshold, feature_names = load_model(args.model)
    
    # Prepare features
    data_paths = {'deliveries': args.deliveries, 'matches': args.matches}
    features = process_single_delivery(
        args.match_id, args.over, args.ball, args.inning ,data_paths
    )
    
    # Make prediction
    proba, is_wicket = predict_wicket_probability(model, scaler, threshold, features, feature_names)
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    plot_wicket_probability_gauge(proba, threshold)
    plt.savefig(args.output, bbox_inches='tight')
    
    # Print results
    print(f"\nPrediction for Match {args.match_id} | ")
    print(f"Over {args.over}.{args.ball} | Probability: {proba:.3f} | Wicket Predicted: {'YES' if is_wicket else 'NO'}")
    
    # Show key factors
    key_factors = analyze_key_factors(features, model, feature_names)
    print("\nKey Decision Factors:")
    for name, value, imp in key_factors[:5]:
        print(f"- {name}: {value:.2f} (Impact: {imp:.3f})")

if __name__ == "__main__":
    main()
