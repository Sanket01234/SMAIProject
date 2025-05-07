import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
import joblib
from time import time
warnings.filterwarnings('ignore')

def load_and_prepare_data(ball_by_ball_path, match_info_path, cutoff_date=None):
    print("Loading data...")
    ball_by_ball = pd.read_csv(ball_by_ball_path)
    match_info = pd.read_csv(match_info_path)
    
    match_info['date'] = pd.to_datetime(match_info['date'])
    
    data = pd.merge(ball_by_ball, match_info[['match_id', 'date', 'venue']], on='match_id')
    
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        data = data[data['date'] <= cutoff]
    
    return data

def precompute_feature_dfs(data):
    print("Precomputing feature dataframes...")
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

def process_match_chunk(match_group, feature_dfs):
    match_id = match_group['match_id'].iloc[0]
    match_date = match_group['date'].iloc[0]
    
    match_group = match_group.sort_values(by=['inning', 'over', 'ball'])
    
    features_list = []
    targets = []
    identifiers_list = []
    
    for i, row in match_group.iterrows():
        inning = row['inning']
        over = row['over']
        ball_num = row['ball']
        batsman = row['batter']
        bowler = row['bowler']
        batting_team = row['batting_team']
        bowling_team = row['bowling_team']
        venue = row['venue']
        
        innings_data = match_group[(match_group['inning'] == inning) & 
                                  ((match_group['over'] < over) | 
                                   ((match_group['over'] == over) & (match_group['ball'] < ball_num)))]
        
        features = {}
        
        features['inning'] = inning
        features['over'] = over
        features['ball'] = ball_num
        
        total_runs = innings_data['total_runs'].sum()
        wickets_fallen = innings_data['is_wicket'].sum()
        balls_bowled = len(innings_data)
        current_run_rate = total_runs / (balls_bowled/6) if balls_bowled > 0 else 0
        
        features['total_runs'] = total_runs
        features['wickets_fallen'] = wickets_fallen
        features['run_rate'] = current_run_rate
        features['balls_remaining'] = 120 - balls_bowled
        
        recent_balls = innings_data.tail(12)
        recent_runs = recent_balls['total_runs'].sum() if not recent_balls.empty else 0
        recent_dots = len(recent_balls[recent_balls['total_runs'] == 0]) if not recent_balls.empty else 0
        recent_boundaries = len(recent_balls[(recent_balls['batsman_runs'] == 4) | (recent_balls['batsman_runs'] == 6)]) if not recent_balls.empty else 0
        
        features['recent_runs'] = recent_runs
        features['recent_dots'] = recent_dots
        features['recent_boundaries'] = recent_boundaries
        
        if over < 6:
            phase = 'Phase1_0-5'
        elif over < 15:
            phase = 'Phase2_6-14'
        else:
            phase = 'Phase3_15-19'
        
        pvp_df = feature_dfs['player_vs_player']
        matchup = pvp_df[(pvp_df['batsman'] == batsman) & (pvp_df['bowler'] == bowler)]
        features['pvp_balls_faced'] = matchup['balls_faced'].values[0] if not matchup.empty else 0
        features['pvp_runs_scored'] = matchup['runs_scored'].values[0] if not matchup.empty else 0
        features['pvp_dismissals'] = matchup['dismissals'].values[0] if not matchup.empty else 0
        features['pvp_strike_rate'] = matchup['strike_rate'].values[0] if not matchup.empty else 0
        
        features['pvp_dismissals'] = np.log1p(features['pvp_dismissals'])
        
        bat_venue_df = feature_dfs['batsman_venue']
        bat_venue = bat_venue_df[(bat_venue_df['batsman'] == batsman) & (bat_venue_df['venue'] == venue)]
        features['bat_venue_runs'] = bat_venue['runs_scored'].values[0] if not bat_venue.empty else 0
        features['bat_venue_balls'] = bat_venue['balls_faced'].values[0] if not bat_venue.empty else 0
        features['bat_venue_avg'] = bat_venue['average'].values[0] if not bat_venue.empty else 0
        features['bat_venue_sr'] = bat_venue['strike_rate'].values[0] if not bat_venue.empty else 0
        
        bowl_venue_df = feature_dfs['bowler_venue']
        bowl_venue = bowl_venue_df[(bowl_venue_df['bowler'] == bowler) & (bowl_venue_df['venue'] == venue)]
        features['bowl_venue_wickets'] = bowl_venue['wickets'].values[0] if not bowl_venue.empty else 0
        features['bowl_venue_balls'] = bowl_venue['balls_bowled'].values[0] if not bowl_venue.empty else 0
        features['bowl_venue_runs'] = bowl_venue['runs_conceded'].values[0] if not bowl_venue.empty else 0
        features['bowl_venue_avg'] = bowl_venue['bowling_average'].values[0] if not bowl_venue.empty else 0
        features['bowl_venue_sr'] = bowl_venue['bowling_strike_rate'].values[0] if not bowl_venue.empty else 0
        
        features['bowl_venue_wickets'] = np.log1p(features['bowl_venue_wickets'])
        
        bat_phase_df = feature_dfs['batsman_phase']
        bat_phase = bat_phase_df[(bat_phase_df['batsman'] == batsman) & (bat_phase_df['phase'] == phase)]
        features['bat_phase_runs'] = bat_phase['runs_scored'].values[0] if not bat_phase.empty else 0
        features['bat_phase_balls'] = bat_phase['balls_faced'].values[0] if not bat_phase.empty else 0
        features['bat_phase_avg'] = bat_phase['average'].values[0] if not bat_phase.empty else 0
        features['bat_phase_sr'] = bat_phase['strike_rate'].values[0] if not bat_phase.empty else 0
        
        bowl_phase_df = feature_dfs['bowler_phase']
        bowl_phase = bowl_phase_df[(bowl_phase_df['bowler'] == bowler) & (bowl_phase_df['phase'] == phase)]
        features['bowl_phase_wickets'] = bowl_phase['wickets'].values[0] if not bowl_phase.empty else 0
        features['bowl_phase_balls'] = bowl_phase['balls_bowled'].values[0] if not bowl_phase.empty else 0
        features['bowl_phase_runs'] = bowl_phase['runs_conceded'].values[0] if not bowl_phase.empty else 0
        features['bowl_phase_avg'] = bowl_phase['bowling_average'].values[0] if not bowl_phase.empty else 0
        features['bowl_phase_sr'] = bowl_phase['bowling_strike_rate'].values[0] if not bowl_phase.empty else 0
        
        bat_vs_team_df = feature_dfs['batsman_vs_team']
        bat_vs_team = bat_vs_team_df[(bat_vs_team_df['batsman'] == batsman) & (bat_vs_team_df['opposing_team'] == bowling_team)]
        features['bat_vs_team_runs'] = bat_vs_team['runs_scored'].values[0] if not bat_vs_team.empty else 0
        features['bat_vs_team_balls'] = bat_vs_team['balls_faced'].values[0] if not bat_vs_team.empty else 0
        features['bat_vs_team_avg'] = bat_vs_team['average'].values[0] if not bat_vs_team.empty else 0
        features['bat_vs_team_sr'] = bat_vs_team['strike_rate'].values[0] if not bat_vs_team.empty else 0
        
        bowl_vs_team_df = feature_dfs['bowler_vs_team']
        bowl_vs_team = bowl_vs_team_df[(bowl_vs_team_df['bowler'] == bowler) & (bowl_vs_team_df['opposing_team'] == batting_team)]
        features['bowl_vs_team_wickets'] = bowl_vs_team['wickets'].values[0] if not bowl_vs_team.empty else 0
        features['bowl_vs_team_balls'] = bowl_vs_team['balls_bowled'].values[0] if not bowl_vs_team.empty else 0
        features['bowl_vs_team_runs'] = bowl_vs_team['runs_conceded'].values[0] if not bowl_vs_team.empty else 0
        features['bowl_vs_team_avg'] = bowl_vs_team['bowling_average'].values[0] if not bowl_vs_team.empty else 0
        features['bowl_vs_team_sr'] = bowl_vs_team['bowling_strike_rate'].values[0] if not bowl_vs_team.empty else 0
        
        if len(innings_data) >= 12:
            recent_balls_sequence = innings_data.tail(12)['total_runs'].tolist()
            dot_ball_count = 0
            for run in reversed(recent_balls_sequence):
                if run == 0:
                    dot_ball_count += 1
                else:
                    break
            features['consecutive_dot_balls'] = dot_ball_count
            
            features['dot_ball_percentage'] = sum(1 for run in recent_balls_sequence if run == 0) / len(recent_balls_sequence)
            features['boundary_percentage'] = sum(1 for run in recent_balls_sequence if run >= 4) / len(recent_balls_sequence)
        else:
            features['consecutive_dot_balls'] = 0
            features['dot_ball_percentage'] = 0
            features['boundary_percentage'] = 0
        
        last_dismissal = innings_data[(innings_data['is_wicket'] == 1) & (innings_data['player_dismissed'] == batsman)]
        features['balls_since_last_dismissal'] = balls_bowled - last_dismissal.index[-1] if not last_dismissal.empty else balls_bowled
        
        features_list.append(features)
        targets.append(1 if row['is_wicket'] == 1 else 0)
        identifiers_list.append({
            'match_id': match_id,
            'inning': inning,
            'over': over,
            'ball': ball_num,
            'batter': batsman,
            'bowler': bowler
        })
    
    return features_list, targets, identifiers_list

def prepare_training_data(ball_by_ball_path, match_info_path, cutoff_date=None, sample_frac=1.0):
    start_time = time()
    data = load_and_prepare_data(ball_by_ball_path, match_info_path, cutoff_date)
    data = data.sort_values(by=['date', 'match_id', 'inning', 'over', 'ball'])
    
    if sample_frac < 1.0:
        unique_matches = data['match_id'].unique()
        sampled_matches = np.random.choice(unique_matches, size=int(len(unique_matches) * sample_frac), replace=False)
        data = data[data['match_id'].isin(sampled_matches)]
        print(f"Sampled {len(sampled_matches)} matches out of {len(unique_matches)}")
    
    feature_dfs = precompute_feature_dfs(data)
    
    all_features = []
    all_targets = []
    all_identifiers = []
    
    match_count = len(data['match_id'].unique())
    print(f"Processing {match_count} matches...")
    
    for i, (match_id, match_group) in enumerate(data.groupby('match_id')):
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{match_count} matches... ({(i+1)/match_count*100:.1f}%)")
        
        features, targets, identifiers = process_match_chunk(match_group, feature_dfs)
        all_features.extend(features)
        all_targets.extend(targets)
        all_identifiers.extend(identifiers)
    
    X = pd.DataFrame(all_features)
    y = np.array(all_targets)
    identifiers = pd.DataFrame(all_identifiers)
    
    print(f"Data preparation completed in {(time() - start_time)/60:.2f} minutes")
    return X, y, identifiers

def train_and_save_model(X, y, identifiers, model_path):
    start_time = time()
    print("Training XGBoost model with optimized parameters for class imbalance...")

    class_ratio = sum(y==0) / sum(y==1)
    print(f"Class imbalance ratio (non-wickets:wickets): {class_ratio:.2f}:1")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    ids_test = identifiers.iloc[X_test.index]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the XGBoost classifier with optimized parameters
    model = xgb.XGBClassifier(
        scale_pos_weight=35,
        n_estimators=200,
        learning_rate=0.07,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    
    # Define the evaluation set using scaled data
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    
    # Train the model with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=eval_set,
        verbose=100
    )
    
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nTop 20 Feature Importances:")
    print(feature_importance.head(20))

    # Save results
    results = ids_test.copy()
    results['actual_wicket'] = y_test
    results['predicted_wicket'] = y_pred
    results['wicket_probability'] = y_pred_proba
    results.to_csv('test_predictions.csv', index=False)
    
    # Save model and scaler
    model_info = {
        'model': model,
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'feature_names': X.columns.tolist()
    }
    
    joblib.dump(model_info, model_path)
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print(f"Model training completed in {(time() - start_time)/60:.2f} minutes")
    print(f"Model saved to {model_path}")
    
    plot_precision_recall_curve(y_test, y_pred_proba)
    
    return model, optimal_threshold

def find_optimal_threshold(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def plot_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    print("Saved precision-recall curve to precision_recall_curve.png")
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png')
    print("Saved threshold optimization plot to threshold_optimization.png")

if __name__ == "__main__":
    print("T20 Cricket Wicket Prediction Model - Training Script")
    
    ball_by_ball_path = 'deliveries.csv'
    match_info_path = 'matches.csv'
    model_path = 'wicket_probability_model.pkl'
    cutoff_date = None
    sample_frac = 1.0
    
    start_time = time()
    
    X, y, identifiers = prepare_training_data(
        ball_by_ball_path, 
        match_info_path,
        cutoff_date,
        sample_frac
    )
    
    print(f"\nClass distribution: {np.bincount(y)}")
    print(f"Wicket percentage: {sum(y)/len(y)*100:.2f}%")
    
    model, threshold = train_and_save_model(X, y, identifiers, model_path)
    
    print(f"Total execution time: {(time() - start_time)/60:.2f} minutes")
