#!/usr/bin/env python3
"""
Advanced solution optimizer - Round 5
Exploring third model options for Popular + EASE + ???
Current best: Pop 85% + EASE 10% + BPR 5% = MAP 0.082
Target: MAP 0.089 for 80/80 score
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
        return {
            'name': name,
            'map': 0.0,
            'score': 0,
            'time': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# EXPLORING THIRD MODEL OPTIONS
# ============================================================================

def solution_v5_1_pop_ease_bpr_baseline(train, users, items):
    """Pop 85% + EASE 10% + BPR 5% (V4 baseline)"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.05
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_2_pop_ease_bpr_87_10_3(train, users, items):
    """Pop 87% + EASE 10% + BPR 3%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.87
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.03
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_3_pop_ease_bpr_83_10_7(train, users, items):
    """Pop 83% + EASE 10% + BPR 7%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.83
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.07
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_4_pop_ease_svd(train, users, items):
    """Pop 88% + EASE 10% + SVD 2%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    svd_model = models.PureSVDModel(factors=100)
    svd_model.fit(ds)
    svd_recs = svd_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.88
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    svd_recs['score'] = (30 - svd_recs['rank']) * 0.02
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          svd_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_5_pop_ease_als(train, users, items):
    """Pop 87% + EASE 10% + ALS 3%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    als_model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(factors=100, iterations=10, regularization=0.1, random_state=42)
    )
    als_model.fit(ds)
    als_recs = als_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.87
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    als_recs['score'] = (30 - als_recs['rank']) * 0.03
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          als_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_6_pop_ease_ease(train, users, items):
    """Pop 85% + EASE(reg=800) 10% + EASE(reg=400) 5%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease1_model = models.EASEModel(regularization=800)
    ease1_model.fit(ds)
    ease1_recs = ease1_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease2_model = models.EASEModel(regularization=400)
    ease2_model.fit(ds)
    ease2_recs = ease2_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease1_recs['score'] = (30 - ease1_recs['rank']) * 0.10
    ease2_recs['score'] = (30 - ease2_recs['rank']) * 0.05
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease1_recs[['user_id', 'item_id', 'score']],
                          ease2_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_7_pop_ease_popular_recent(train, users, items):
    """Pop 85% + EASE 10% + PopRecent 5%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    # Popular from recent data
    recent_cutoff = train['datetime'].max() - pd.Timedelta(days=30)
    train_recent = train[train['datetime'] >= recent_cutoff].copy()
    train_recent[Columns.Weight] = 1
    ds_recent = dataset.Dataset.construct(train_recent)
    
    pop_recent_model = models.PopularModel()
    pop_recent_model.fit(ds_recent)
    pop_recent_recs = pop_recent_model.recommend(users=hot_users, dataset=ds_recent, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    pop_recent_recs['score'] = (30 - pop_recent_recs['rank']) * 0.05
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          pop_recent_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_8_pop_ease_12_bpr_3(train, users, items):
    """Pop 85% + EASE 12% + BPR 3%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.12
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.03
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_9_pop_ease_8_bpr_7(train, users, items):
    """Pop 85% + EASE 8% + BPR 7%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.08
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.07
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v5_10_pop_weighted_ease_bpr(train, users, items):
    """Pop(weighted) 85% + EASE 10% + BPR 5%"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    train[Columns.Weight] = 1
    ds_unweighted = dataset.Dataset.construct(train)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds_unweighted)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds_unweighted, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(factors=100, iterations=50, random_state=42)
    )
    bpr_model.fit(ds_unweighted)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds_unweighted, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.05
    
    combined = pd.concat([pop_recs[['user_id', 'item_id', 'score']],
                          ease_recs[['user_id', 'item_id', 'score']],
                          bpr_recs[['user_id', 'item_id', 'score']]])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    return result[['user_id', 'item_id', 'score', 'rank']]


# ============================================================================
# RUN ALL SOLUTIONS
# ============================================================================

solutions = [
    ("V5.1: Pop 85% + EASE 10% + BPR 5% (baseline)", solution_v5_1_pop_ease_bpr_baseline),
    ("V5.2: Pop 87% + EASE 10% + BPR 3%", solution_v5_2_pop_ease_bpr_87_10_3),
    ("V5.3: Pop 83% + EASE 10% + BPR 7%", solution_v5_3_pop_ease_bpr_83_10_7),
    ("V5.4: Pop 88% + EASE 10% + SVD 2%", solution_v5_4_pop_ease_svd),
    ("V5.5: Pop 87% + EASE 10% + ALS 3%", solution_v5_5_pop_ease_als),
    ("V5.6: Pop 85% + EASE 10% + EASE 5%", solution_v5_6_pop_ease_ease),
    ("V5.7: Pop 85% + EASE 10% + PopRecent 5%", solution_v5_7_pop_ease_popular_recent),
    ("V5.8: Pop 85% + EASE 12% + BPR 3%", solution_v5_8_pop_ease_12_bpr_3),
    ("V5.9: Pop 85% + EASE 8% + BPR 7%", solution_v5_9_pop_ease_8_bpr_7),
    ("V5.10: Pop(weighted) 85% + EASE 10% + BPR 5%", solution_v5_10_pop_weighted_ease_bpr),
]

results = []
for name, solution_func in solutions:
    result = evaluate_solution(name, solution_func, train, users, items, test)
    results.append(result)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS - ROUND 5: THIRD MODEL EXPLORATION")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('map', ascending=False)

print("\nRanking by MAP@10:")
print("-" * 80)
BEST_V4 = 0.081920
for idx, row in results_df.iterrows():
    status = "✓" if row['success'] else "✗"
    badge = ""
    if row['success']:
        if row['map'] >= 0.089:
            badge = " 🎯 PERFECT!"
        elif row['map'] >= 0.071:
            badge = " ✅"
    improvement = ""
    if row['success'] and row['map'] > BEST_V4:
        improvement = f" (+{row['map'] - BEST_V4:.6f})"
    elif row['success']:
        improvement = f" ({row['map'] - BEST_V4:+.6f})"
    print(f"{status} {row['name']:50s} | MAP: {row['map']:.6f}{improvement:15s} | Score: {row['score']:2d}/80 {badge}")

print("\n" + "=" * 80)
if len(results_df[results_df['success']]) > 0:
    best = results_df[results_df['success']].iloc[0]
    print(f"🏆 BEST SOLUTION: {best['name']}")
    print(f"   MAP@10: {best['map']:.6f}")
    print(f"   Score: {best['score']}/80")
    print(f"   Improvement over V4: {best['map'] - BEST_V4:+.6f}")
    
    if best['map'] >= 0.089:
        print("   🎉🎉🎉 PERFECT SCORE ACHIEVED!")
    elif best['map'] >= 0.071:
        print(f"   ✅ Target reached! Gap to perfect: {0.089 - best['map']:.6f}")
else:
    print("❌ No successful solutions")

print("=" * 80)

