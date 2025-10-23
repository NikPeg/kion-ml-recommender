#!/usr/bin/env python3
"""
Automated solution optimizer for RecSys project
Tests multiple approaches and finds the best one
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
# SOLUTION VARIANTS
# ============================================================================

def solution_1_simple_als(train, users, items):
    """Simple ALS with basic weights"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 50, 5, 1)
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=100,
            iterations=15,
            regularization=0.01,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_2_als_high_factors(train, users, items):
    """ALS with more factors and iterations"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 70, 10,
                                     np.where(train['watched_pct'] >= 30, 3, 1))
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=200,
            iterations=20,
            regularization=0.01,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_3_als_high_alpha(train, users, items):
    """ALS with high alpha for implicit feedback"""
    train[Columns.Weight] = train['watched_pct'] / 10.0  # Proportional weights
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=150,
            iterations=20,
            regularization=0.01,
            alpha=20.0,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_4_ease_optimized(train, users, items):
    """EASE with optimized regularization"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 70, 10,
                                     np.where(train['watched_pct'] >= 40, 5,
                                     np.where(train['watched_pct'] >= 15, 2, 1)))
    ds = dataset.Dataset.construct(train)
    
    model = models.EASEModel(regularization=500)
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_5_pure_svd(train, users, items):
    """PureSVD approach"""
    train[Columns.Weight] = np.where(train['watched_pct'] >= 60, 8,
                                     np.where(train['watched_pct'] >= 30, 4, 1))
    ds = dataset.Dataset.construct(train)
    
    model = models.PureSVDModel(factors=150)
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_6_bpr(train, users, items):
    """BPR approach"""
    train[Columns.Weight] = 1  # BPR ignores weights
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitBPRWrapperModel(
        model=implicit.bpr.BayesianPersonalizedRanking(
            factors=150,
            iterations=100,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


def solution_7_als_smart_weights(train, users, items):
    """ALS with smart weight distribution"""
    # Give exponential weights to high completion
    train[Columns.Weight] = np.power(train['watched_pct'] / 100.0, 2) * 10 + 0.1
    ds = dataset.Dataset.construct(train)
    
    model = models.ImplicitALSWrapperModel(
        model=implicit.als.AlternatingLeastSquares(
            factors=180,
            iterations=25,
            regularization=0.05,
            alpha=15.0,
            random_state=42
        )
    )
    model.fit(ds)
    return model.recommend(users=hot_users, dataset=ds, k=10, filter_viewed=True)


# ============================================================================
# RUN ALL SOLUTIONS
# ============================================================================

solutions = [
    ("1. Simple ALS", solution_1_simple_als),
    ("2. ALS High Factors", solution_2_als_high_factors),
    ("3. ALS High Alpha", solution_3_als_high_alpha),
    ("4. EASE Optimized", solution_4_ease_optimized),
    ("5. PureSVD", solution_5_pure_svd),
    ("6. BPR", solution_6_bpr),
    ("7. ALS Smart Weights", solution_7_als_smart_weights),
]

results = []
for name, solution_func in solutions:
    result = evaluate_solution(name, solution_func, train, users, items, test)
    results.append(result)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('map', ascending=False)

print("\nRanking by MAP@10:")
print("-" * 80)
for idx, row in results_df.iterrows():
    status = "✓" if row['success'] else "✗"
    print(f"{status} {row['name']:30s} | MAP: {row['map']:.6f} | Score: {row['score']:2d}/80 | Time: {row['time']:.1f}s")

print("\n" + "=" * 80)
best = results_df.iloc[0]
print(f"🏆 BEST SOLUTION: {best['name']}")
print(f"   MAP@10: {best['map']:.6f}")
print(f"   Score: {best['score']}/80")
print("=" * 80)

