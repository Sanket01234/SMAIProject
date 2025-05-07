import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, f1_score,precision_score, recall_score
import matplotlib.pyplot as plt
import warnings
from time import time
warnings.filterwarnings('ignore')
import os
import shutil

def load_and_prepare_data(ball_by_ball_path, match_info_path, start_date=None, end_date=None):
    print("Loading data...")
    ball_by_ball = pd.read_csv(ball_by_ball_path)
    match_info = pd.read_csv(match_info_path)
    
    match_info['date'] = pd.to_datetime(match_info['date'])
    
    data = pd.merge(ball_by_ball, match_info[['match_id', 'date', 'venue']], on='match_id')
    
    if start_date:
        start = pd.to_datetime(start_date)
        data = data[data['date'] >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        data = data[data['date'] <= end]
    
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

def prepare_evaluation_data(ball_by_ball_path, match_info_path, start_date=None, end_date=None):
    start_time = time()
    data = load_and_prepare_data(ball_by_ball_path, match_info_path, start_date, end_date)
    data = data.sort_values(by=['date', 'match_id', 'inning', 'over', 'ball'])
    
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

def plot_precision_recall_curve(y_true, y_pred_proba, output_prefix='eval'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_precision_recall_curve.png')
    print(f"Saved precision-recall curve to {output_prefix}_precision_recall_curve.png")
    
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
    plt.savefig(f'{output_prefix}_threshold_optimization.png')
    print(f"Saved threshold optimization plot to {output_prefix}_threshold_optimization.png")
    
    return optimal_threshold

def evaluate_model(model_path, ball_by_ball_path, match_info_path, start_date=None, end_date=None, output_prefix='eval'):
    start_time = time()
    print(f"T20 Cricket Wicket Prediction Model - Evaluation Script")
    print(f"Evaluating model: {model_path}")
    print(f"Evaluation period: {start_date or 'beginning'} to {end_date or 'end'}")
    
    model_info = joblib.load(model_path)
    model = model_info['model']
    scaler = model_info['scaler']
    original_threshold = model_info['optimal_threshold']
    feature_names = model_info['feature_names']
    
    X, y, identifiers = prepare_evaluation_data(ball_by_ball_path, match_info_path, start_date, end_date)
    
    print(f"\nClass distribution: {np.bincount(y)}")
    print(f"Wicket percentage: {sum(y)/len(y)*100:.2f}%")
    
    if set(X.columns) != set(feature_names):
        missing_cols = set(feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(feature_names)
        
        if missing_cols:
            print(f"Warning: Missing columns in evaluation data: {missing_cols}")
            for col in missing_cols:
                X[col] = 0
        
        if extra_cols:
            print(f"Warning: Extra columns in evaluation data that will be ignored: {extra_cols}")
            X = X[feature_names]
    
    X = X[feature_names]
    
    print("Scaling features...")
    X_scaled = scaler.transform(X)
    
    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Create output directory
    output_dir = f"{output_prefix}_threshold_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Threshold testing with saving
    thresholds = [0.7, 0.75, 0.8, 0.82, 0.85]
    results = []
    
    for thresh in thresholds:
        print(f"\nProcessing threshold {thresh:.2f}")
        thresh_dir = os.path.join(output_dir, f"thresh_{thresh:.2f}")
        os.makedirs(thresh_dir, exist_ok=True)
        
        # Predictions
        y_pred = (y_pred_proba >= thresh).astype(int)
        res = {
            'threshold': thresh,
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        # Save threshold-specific results
        threshold_results = identifiers.copy()
        threshold_results['predicted'] = y_pred
        threshold_results['probability'] = y_pred_proba
        threshold_results.to_csv(os.path.join(thresh_dir, 'predictions.csv'), index=False)
        
        # Save classification report
        report = classification_report(y, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(thresh_dir, 'classification_report.csv'))
        
        # Save threshold-specific model info
        model_info['optimal_threshold'] = thresh
        joblib.dump(model_info, os.path.join(thresh_dir, f'model_thresh_{thresh:.2f}.pkl'))
        
        results.append(res)
    
    # Save comparative analysis
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'threshold_comparison.csv'), index=False)
    
    # Copy best model (based on F1)
    best_thresh = max(results, key=lambda x: x['f1'])['threshold']
    shutil.copyfile(
        os.path.join(output_dir, f'thresh_{best_thresh:.2f}', f'model_thresh_{best_thresh:.2f}.pkl'),
        f'best_model_{output_prefix}.pkl'
    )
    
    print("Evaluating with original threshold...")
    y_pred_original = (y_pred_proba >= original_threshold).astype(int)
    
    print("Original model threshold:", original_threshold)
    print(classification_report(y, y_pred_original))
    auc = roc_auc_score(y, y_pred_proba)
    ap = average_precision_score(y, y_pred_proba)
    original_f1 = f1_score(y, y_pred_original)
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"F1 Score: {original_f1:.4f}")
    
    print("\nRe-optimizing threshold for current data...")
    optimal_threshold = plot_precision_recall_curve(y, y_pred_proba, output_prefix)
    
    print(f"Re-optimized threshold: {optimal_threshold:.4f}")
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    print("\nEvaluating with re-optimized threshold...")
    print(classification_report(y, y_pred_optimized))
    optimized_f1 = f1_score(y, y_pred_optimized)
    print(f"F1 Score: {optimized_f1:.4f}")
    
    results = identifiers.copy()
    results['actual_wicket'] = y
    results['predicted_wicket_original'] = y_pred_original
    results['predicted_wicket_optimized'] = y_pred_optimized
    results['wicket_probability'] = y_pred_proba
    results.to_csv(f'{output_prefix}_predictions.csv', index=False)
    
    eval_summary = {
        'metrics': {
            'roc_auc': auc,
            'average_precision': ap,
            'original_threshold': original_threshold,
            'original_f1': original_f1,
            'optimized_threshold': optimal_threshold,
            'optimized_f1': optimized_f1,
            'total_deliveries': len(y),
            'total_wickets': sum(y),
            'wicket_percentage': sum(y)/len(y)*100
        },
        'class_distribution': {
            'no_wicket': int(np.sum(y == 0)),
            'wicket': int(np.sum(y == 1))
        }
    }
    
    pd.DataFrame([eval_summary['metrics']]).to_csv(f'{output_prefix}_metrics_summary.csv', index=False)
    
    print(f"\nEvaluation completed in {(time() - start_time)/60:.2f} minutes")
    print(f"Results saved with prefix '{output_prefix}'")
    
    return eval_summary

if __name__ == "__main__":
    model_path = 'wicket_probability_model.pkl'
    ball_by_ball_path = 'deliveries.csv'
    match_info_path = 'matches.csv'
    
    start_date = '2023-01-01'
    end_date = None
    output_prefix = 'eval_2023'
    
    evaluate_model(model_path, ball_by_ball_path, match_info_path, start_date, end_date, output_prefix)