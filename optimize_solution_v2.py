#!/usr/bin/env python3
"""
Advanced solution optimizer - Round 2
Based on seminar materials analysis
"""

import os
import warnings
warnings.simplefilter("ignore")

import implicit
import rectools
import pandas as pd
import numpy as np
from rectools import models, dataset, metrics, Columns
from catboost import CatBoostRanker

# Load data
print("=" * 80)
print("LOADING DATA...")
print("=" * 80)

data_path = os.environ.get("DATA_PATH", "data_original")

users = pd.read_csv(os.path.join(data_path, "users.csv"))
items = pd.read_csv(os.path.join(data_path, "items.csv"))
users = users.sample(frac=0.1, random_state=42)

interactions = (
    pd.read_csv(os.path.join(data_path, "interactions.csv"), parse_dates=["last_watch_dt"])
    .rename(columns={'total_dur': Columns.Weight, 'last_watch_dt': Columns.Datetime})
)
interactions = interactions[interactions["user_id"].isin(users["user_id"])]

# Train-test split
N_DAYS = 7
max_date = interactions['datetime'].max()
train = interactions[(interactions['datetime'] <= max_date - pd.Timedelta(days=N_DAYS))]
test = interactions[(interactions['datetime'] > max_date - pd.Timedelta(days=N_DAYS))]

catalog = train[Columns.Item].unique()
test_users = test[Columns.User].unique()
cold_users = set(test_users) - set(train[Columns.User])
test = test[~test[Columns.User].isin(cold_users)]
hot_users = test[Columns.User].unique()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Hot users: {len(hot_users)}")
print()


def scorer(map_score: float):
    """Calculate score from MAP"""
    UPPER_BOUND = 0.089
    LOWER_BOUND = 0.071
    score = int(min(max((map_score - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND), 0), 1) * 80)
    return score


def evaluate_solution(name, solution_func, train_df, users_df, items_df, test_df):
    """Evaluate a solution and return MAP score"""
    print(f"\n{'=' * 80}")
    print(f"TESTING: {name}")
    print(f"{'=' * 80}")
    
    try:
        import time
        start_time = time.time()
        
        recs = solution_func(train_df.copy(), users_df.copy(), items_df.copy())
        
        elapsed = time.time() - start_time
        
        map_score = metrics.MAP(10).calc(recs, test_df)
        final_score = scorer(map_score)
        
        print(f"✓ MAP@10: {map_score:.6f}")
        print(f"✓ Final Score: {final_score}/80")
        print(f"✓ Time: {elapsed:.1f}s")
        
        return {
            'name': name,
            'map': map_score,
            'score': final_score,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'map': 0.0,
            'score': 0,
            'time': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# ADVANCED SOLUTIONS - Round 2
# ============================================================================

def solution_v2_1_als_no_weights(train, users, items):
    """ALS without aggressive weighting - let the model learn"""
    # Simple binary weights
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=100,
            iterations=15,
            regularization=0.1,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_2_als_light_weights(train, users, items):
    """ALS with light weighting based on completion"""
    # Light weighting - only distinguish fully watched vs not
    train[Columns.Weight] = np.where(train['watched_pct'] >= 90, 3, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=128,
            iterations=15,
            regularization=0.1,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_3_ease_light_weights(train, users, items):
    """EASE with light weighting"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 90, 3, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.EASEModel(regularization=300)
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_4_popular_baseline(train, users, items):
    """Popular baseline - sometimes simple works"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_5_als_duration_weights(train, users, items):
    """ALS using total_dur (original weight) instead of watched_pct"""
    # Use original duration weights
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=128,
            iterations=15,
            regularization=0.1,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_6_als_log_weights(train, users, items):
    """ALS with logarithmic weighting"""
    # Log scale weights
    train[Columns.Weight] = np.log1p(train['watched_pct'])
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=128,
            iterations=15,
            regularization=0.1,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_7_ensemble_simple(train, users, items):
    """Simple ensemble without normalization issues"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    # Two models
    als_model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=100,
            iterations=15,
            regularization=0.1,
            random_state=42
        )
    )
    als_model.fit(ds)
    
    ease_model = models.EASEModel(regularization=300)
    ease_model.fit(ds)
    
    # Get top 50 from each
    als_recs = als_model.recommend(users=hot_users, dataset=ds, k=50, filter_viewed=True)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=50, filter_viewed=True)
    
    # Combine: give each item score based on its rank
    als_recs['score'] = 50 - als_recs['rank']
    ease_recs['score'] = 50 - ease_recs['rank']
    
    # Merge and sum scores
    combined = pd.concat([
        als_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    # Top 10
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v2_8_als_higher_reg(train, users, items):
    """ALS with higher regularization to avoid overfitting"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=100,
            iterations=15,
            regularization=0.5,  # Higher regularization
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_9_ease_higher_reg(train, users, items):
    """EASE with different regularization values"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.EASEModel(regularization=800)  # Higher reg
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v2_10_als_more_factors(train, users, items):
    """ALS with even more factors"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=300,
            iterations=20,
            regularization=0.1,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


# ============================================================================
# RUN ALL SOLUTIONS
# ============================================================================

solutions = [
    ("V2.1: ALS No Weights", solution_v2_1_als_no_weights),
    ("V2.2: ALS Light Weights", solution_v2_2_als_light_weights),
    ("V2.3: EASE Light Weights", solution_v2_3_ease_light_weights),
    ("V2.4: Popular Baseline", solution_v2_4_popular_baseline),
    ("V2.5: ALS Duration Weights", solution_v2_5_als_duration_weights),
    ("V2.6: ALS Log Weights", solution_v2_6_als_log_weights),
    ("V2.7: Simple Ensemble", solution_v2_7_ensemble_simple),
    ("V2.8: ALS Higher Reg", solution_v2_8_als_higher_reg),
    ("V2.9: EASE Higher Reg", solution_v2_9_ease_higher_reg),
    ("V2.10: ALS More Factors", solution_v2_10_als_more_factors),
]

results = []
for name, solution_func in solutions:
    result = evaluate_solution(name, solution_func, train, users, items, test)
    results.append(result)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS - ROUND 2")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('map', ascending=False)

print("\nRanking by MAP@10:")
print("-" * 80)
for idx, row in results_df.iterrows():
    status = "✓" if row['success'] else "✗"
    print(f"{status} {row['name']:35s} | MAP: {row['map']:.6f} | Score: {row['score']:2d}/80 | Time: {row['time']:.1f}s")

print("\n" + "=" * 80)
if len(results_df[results_df['success']]) > 0:
    best = results_df[results_df['success']].iloc[0]
    print(f"🏆 BEST SOLUTION: {best['name']}")
    print(f"   MAP@10: {best['map']:.6f}")
    print(f"   Score: {best['score']}/80")
    
    if best['map'] >= 0.071:
        print("   🎉 TARGET REACHED!")
    else:
        print(f"   Need {0.071 - best['map']:.6f} more to reach minimum target")
else:
    print("❌ No successful solutions")

print("=" * 80)

