#!/usr/bin/env python3
"""
Advanced solution optimizer - Round 3
Improving upon Popular Baseline (MAP 0.078)
Goal: Reach upper bound 0.089 for 80/80 score
"""

import os
import warnings
warnings.simplefilter("ignore")

import implicit
import rectools
import pandas as pd
import numpy as np
from rectools import models, dataset, metrics, Columns

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
# ADVANCED SOLUTIONS - Round 3: Popular + Personalization
# ============================================================================

def solution_v3_1_popular_baseline(train, users, items):
    """Pure Popular baseline - our best so far"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_2_popular_weighted(train, users, items):
    """Popular with watched_pct weighting"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_3_popular_high_engagement(train, users, items):
    """Popular considering only high engagement"""
    train_filtered = train[train['watched_pct'] >= 70].copy()
    train_filtered[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train_filtered)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_4_popular_recency(train, users, items):
    """Popular with recency weighting - recent items more important"""
    # Give more weight to recent interactions
    train_sorted = train.sort_values('datetime')
    train_sorted[Columns.Weight] = 1
    
    # Apply recency weight
    days_diff = (train_sorted['datetime'].max() - train_sorted['datetime']).dt.days
    recency_weight = 1.0 / (1.0 + days_diff / 30.0)  # Decay over months
    train_sorted[Columns.Weight] = recency_weight * 10
    
    ds = dataset.Dataset.construct(train_sorted)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_5_popular_plus_ease(train, users, items):
    """Combine Popular (70%) + EASE (30%)"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    # Popular model
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    # EASE model  
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    # Combine with rank-based scoring
    pop_recs['score'] = 30 - pop_recs['rank']
    ease_recs['score'] = 30 - ease_recs['rank']
    
    # Weight popular higher
    pop_recs['score'] = pop_recs['score'] * 0.7
    ease_recs['score'] = ease_recs['score'] * 0.3
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v3_6_popular_plus_als(train, users, items):
    """Combine Popular (80%) + ALS (20%)"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    # Popular model
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    # ALS model
    als_model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=100,
            iterations=10,
            regularization=0.1,
            random_state=42
        )
    )
    als_model.fit(ds)
    als_recs = als_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    # Combine
    pop_recs['score'] = 30 - pop_recs['rank']
    als_recs['score'] = 30 - als_recs['rank']
    
    pop_recs['score'] = pop_recs['score'] * 0.8
    als_recs['score'] = als_recs['score'] * 0.2
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        als_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v3_7_popular_quality_filter(train, users, items):
    """Popular with quality threshold - only well-watched items"""
    # Calculate average watched_pct per item
    item_quality = train.groupby('item_id')['watched_pct'].agg(['mean', 'count']).reset_index()
    
    # Keep items with good average watched_pct and enough interactions
    good_items = item_quality[(item_quality['mean'] >= 60) & (item_quality['count'] >= 3)]['item_id']
    
    train_filtered = train[train['item_id'].isin(good_items)].copy()
    train_filtered[Columns.Weight] = 1
    
    ds = dataset.Dataset.construct(train_filtered)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_8_popular_90_ease_10(train, users, items):
    """Combine Popular (90%) + EASE (10%) - more conservative"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=20, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=20, filter_viewed=True)
    
    pop_recs['score'] = 20 - pop_recs['rank']
    ease_recs['score'] = 20 - ease_recs['rank']
    
    pop_recs['score'] = pop_recs['score'] * 0.9
    ease_recs['score'] = ease_recs['score'] * 0.1
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v3_9_popular_by_duration(train, users, items):
    """Popular by total duration instead of count"""
    # Use duration as weight
    train[Columns.Weight] = train[Columns.Weight]  # original duration
    ds = dataset.Dataset.construct(train)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_v3_10_popular_recent_only(train, users, items):
    """Popular but only from recent data (last month)"""
    # Use only last 30 days
    recent_cutoff = train['datetime'].max() - pd.Timedelta(days=30)
    train_recent = train[train['datetime'] >= recent_cutoff].copy()
    train_recent[Columns.Weight] = 1
    
    ds = dataset.Dataset.construct(train_recent)
    
    model = models.PopularModel()
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


# ============================================================================
# RUN ALL SOLUTIONS
# ============================================================================

solutions = [
    ("V3.1: Popular Baseline", solution_v3_1_popular_baseline),
    ("V3.2: Popular Weighted", solution_v3_2_popular_weighted),
    ("V3.3: Popular High Engagement", solution_v3_3_popular_high_engagement),
    ("V3.4: Popular Recency", solution_v3_4_popular_recency),
    ("V3.5: Popular 70% + EASE 30%", solution_v3_5_popular_plus_ease),
    ("V3.6: Popular 80% + ALS 20%", solution_v3_6_popular_plus_als),
    ("V3.7: Popular Quality Filter", solution_v3_7_popular_quality_filter),
    ("V3.8: Popular 90% + EASE 10%", solution_v3_8_popular_90_ease_10),
    ("V3.9: Popular by Duration", solution_v3_9_popular_by_duration),
    ("V3.10: Popular Recent Only", solution_v3_10_popular_recent_only),
]

results = []
for name, solution_func in solutions:
    result = evaluate_solution(name, solution_func, train, users, items, test)
    results.append(result)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS - ROUND 3")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('map', ascending=False)

print("\nRanking by MAP@10:")
print("-" * 80)
for idx, row in results_df.iterrows():
    status = "✓" if row['success'] else "✗"
    badge = ""
    if row['success']:
        if row['map'] >= 0.089:
            badge = " 🎯 PERFECT!"
        elif row['map'] >= 0.071:
            badge = " ✅ PASSED"
    print(f"{status} {row['name']:40s} | MAP: {row['map']:.6f} | Score: {row['score']:2d}/80 {badge} | Time: {row['time']:.1f}s")

print("\n" + "=" * 80)
if len(results_df[results_df['success']]) > 0:
    best = results_df[results_df['success']].iloc[0]
    print(f"🏆 BEST SOLUTION: {best['name']}")
    print(f"   MAP@10: {best['map']:.6f}")
    print(f"   Score: {best['score']}/80")
    
    if best['map'] >= 0.089:
        print("   🎉🎉🎉 PERFECT SCORE ACHIEVED!")
    elif best['map'] >= 0.071:
        print(f"   ✅ Target reached! Gap to perfect: {0.089 - best['map']:.6f}")
    else:
        print(f"   Need {0.071 - best['map']:.6f} more to reach minimum target")
else:
    print("❌ No successful solutions")

print("=" * 80)

