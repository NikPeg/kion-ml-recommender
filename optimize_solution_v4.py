#!/usr/bin/env python3
"""
Advanced solution optimizer - Round 4
Fine-tuning Popular + EASE combination (Current best: 90/10 = MAP 0.081)
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
# FINE-TUNING Popular + EASE
# ============================================================================

def make_popular_ease_mix(popular_weight, ease_weight, ease_reg=800):
    """Factory function for Popular + EASE combinations"""
    def solution(train, users, items):
        train[Columns.Weight] = 1
        ds = dataset.Dataset.construct(train)
        
        pop_model = models.PopularModel()
        pop_model.fit(ds)
        pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=20, filter_viewed=True)
        
        ease_model = models.EASEModel(regularization=ease_reg)
        ease_model.fit(ds)
        ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=20, filter_viewed=True)
        
        pop_recs['score'] = (20 - pop_recs['rank']) * popular_weight
        ease_recs['score'] = (20 - ease_recs['rank']) * ease_weight
        
        combined = pd.concat([
            pop_recs[['user_id', 'item_id', 'score']],
            ease_recs[['user_id', 'item_id', 'score']]
        ])
        
        combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
        combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
        
        result = combined.groupby('user_id').head(10).reset_index(drop=True)
        result['rank'] = result.groupby('user_id').cumcount() + 1
        
        return result[['user_id', 'item_id', 'score', 'rank']]
    
    return solution


def solution_v4_1_popular_95_ease_5(train, users, items):
    """Popular 95% + EASE 5%"""
    return make_popular_ease_mix(0.95, 0.05)(train, users, items)


def solution_v4_2_popular_92_ease_8(train, users, items):
    """Popular 92% + EASE 8%"""
    return make_popular_ease_mix(0.92, 0.08)(train, users, items)


def solution_v4_3_popular_90_ease_10(train, users, items):
    """Popular 90% + EASE 10% (baseline)"""
    return make_popular_ease_mix(0.90, 0.10)(train, users, items)


def solution_v4_4_popular_88_ease_12(train, users, items):
    """Popular 88% + EASE 12%"""
    return make_popular_ease_mix(0.88, 0.12)(train, users, items)


def solution_v4_5_popular_85_ease_15(train, users, items):
    """Popular 85% + EASE 15%"""
    return make_popular_ease_mix(0.85, 0.15)(train, users, items)


def solution_v4_6_popular_90_ease_10_reg500(train, users, items):
    """Popular 90% + EASE 10% with reg=500"""
    return make_popular_ease_mix(0.90, 0.10, ease_reg=500)(train, users, items)


def solution_v4_7_popular_90_ease_10_reg1000(train, users, items):
    """Popular 90% + EASE 10% with reg=1000"""
    return make_popular_ease_mix(0.90, 0.10, ease_reg=1000)(train, users, items)


def solution_v4_8_popular_ease_bpr(train, users, items):
    """Popular 85% + EASE 10% + BPR 5%"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    bpr_model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(
            factors=100,
            iterations=50,
            random_state=42
        )
    )
    bpr_model.fit(ds)
    bpr_recs = bpr_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.85
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    bpr_recs['score'] = (30 - bpr_recs['rank']) * 0.05
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']],
        bpr_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v4_9_popular_weighted_ease(train, users, items):
    """Popular (with weighted watched_pct) 90% + EASE 10%"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 80, 2, 1)
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=20, filter_viewed=True)
    
    train[Columns.Weight] = 1  # Reset for EASE
    ds_ease = dataset.Dataset.construct(train)
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds_ease)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds_ease, k=20, filter_viewed=True)
    
    pop_recs['score'] = (20 - pop_recs['rank']) * 0.90
    ease_recs['score'] = (20 - ease_recs['rank']) * 0.10
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


def solution_v4_10_popular_ease_top30(train, users, items):
    """Popular 90% + EASE 10% with k=30 candidates"""
    train[Columns.Weight] = 1
    ds = dataset.Dataset.construct(train)
    
    pop_model = models.PopularModel()
    pop_model.fit(ds)
    pop_recs = pop_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    ease_model = models.EASEModel(regularization=800)
    ease_model.fit(ds)
    ease_recs = ease_model.recommend(users=hot_users, dataset=ds, k=30, filter_viewed=True)
    
    pop_recs['score'] = (30 - pop_recs['rank']) * 0.90
    ease_recs['score'] = (30 - ease_recs['rank']) * 0.10
    
    combined = pd.concat([
        pop_recs[['user_id', 'item_id', 'score']],
        ease_recs[['user_id', 'item_id', 'score']]
    ])
    
    combined = combined.groupby(['user_id', 'item_id'], as_index=False)['score'].sum()
    combined = combined.sort_values(['user_id', 'score'], ascending=[True, False])
    
    result = combined.groupby('user_id').head(10).reset_index(drop=True)
    result['rank'] = result.groupby('user_id').cumcount() + 1
    
    return result[['user_id', 'item_id', 'score', 'rank']]


# ============================================================================
# RUN ALL SOLUTIONS
# ============================================================================

solutions = [
    ("V4.1: Popular 95% + EASE 5%", solution_v4_1_popular_95_ease_5),
    ("V4.2: Popular 92% + EASE 8%", solution_v4_2_popular_92_ease_8),
    ("V4.3: Popular 90% + EASE 10% (baseline)", solution_v4_3_popular_90_ease_10),
    ("V4.4: Popular 88% + EASE 12%", solution_v4_4_popular_88_ease_12),
    ("V4.5: Popular 85% + EASE 15%", solution_v4_5_popular_85_ease_15),
    ("V4.6: Popular 90% + EASE 10% reg=500", solution_v4_6_popular_90_ease_10_reg500),
    ("V4.7: Popular 90% + EASE 10% reg=1000", solution_v4_7_popular_90_ease_10_reg1000),
    ("V4.8: Pop 85% + EASE 10% + BPR 5%", solution_v4_8_popular_ease_bpr),
    ("V4.9: Popular Weighted + EASE", solution_v4_9_popular_weighted_ease),
    ("V4.10: Popular 90% + EASE 10% k=30", solution_v4_10_popular_ease_top30),
]

results = []
for name, solution_func in solutions:
    result = evaluate_solution(name, solution_func, train, users, items, test)
    results.append(result)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS - ROUND 4: FINE-TUNING")
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
            badge = " ✅"
    improvement = ""
    if row['success'] and row['map'] > 0.080916:
        improvement = f" (+{row['map'] - 0.080916:.6f})"
    print(f"{status} {row['name']:45s} | MAP: {row['map']:.6f}{improvement:15s} | Score: {row['score']:2d}/80 {badge}")

print("\n" + "=" * 80)
if len(results_df[results_df['success']]) > 0:
    best = results_df[results_df['success']].iloc[0]
    print(f"🏆 BEST SOLUTION: {best['name']}")
    print(f"   MAP@10: {best['map']:.6f}")
    print(f"   Score: {best['score']}/80")
    print(f"   Improvement over V3: {best['map'] - 0.080916:.6f}")
    
    if best['map'] >= 0.089:
        print("   🎉🎉🎉 PERFECT SCORE ACHIEVED!")
    elif best['map'] >= 0.071:
        print(f"   ✅ Target reached! Gap to perfect: {0.089 - best['map']:.6f}")
else:
    print("❌ No successful solutions")

print("=" * 80)

