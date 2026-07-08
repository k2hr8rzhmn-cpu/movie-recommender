# -*- coding: utf-8 -*-
"""
离线预处理脚本：把大文件压缩成网站可用的小数据
运行一次即可，生成 data/ 目录下的所有文件
"""
import sys, os, io, json, pickle, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import scipy.sparse as sp

# ============ 路径配置 ============
# 数据源（王雅萱交付的文件夹）
MODEL_DIR = r'c:/Users/18433/WorkBuddy/20260629093110/model_code/模型过程_王雅萱/7-6任务交付__Content-Based 推荐_王雅萱'
# 用户之前提供的 SVD 微型模型
SVD_PATH = r'D:/18433/微信文件/xwechat_files/wxid_pitez8j7e0so21_86d2/msg/file/2026-07/svd_features.pkl'
# 输出目录
DATA_DIR = r'c:/Users/18433/WorkBuddy/20260629093110/movie_recommender/data'
os.makedirs(DATA_DIR, exist_ok=True)

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}')

# ============ Step 1: 电影信息表 ============
log('Step 1: 处理电影信息表...')
movie_content = pd.read_csv(os.path.join(MODEL_DIR, 'movie_content.csv'))
movie_info_list = []
for _, row in movie_content.iterrows():
    genres = str(row.get('genres_space', '')).split() if pd.notna(row.get('genres_space')) else []
    movie_info_list.append({
        'movieId': int(row['movieId']),
        'title': str(row['title']),
        'year': int(row['year']) if pd.notna(row['year']) else 0,
        'genres': genres,
    })

movie_id_to_info = {m['movieId']: m for m in movie_info_list}
with open(os.path.join(DATA_DIR, 'movie_info.json'), 'w', encoding='utf-8') as f:
    json.dump(movie_info_list, f, ensure_ascii=False)
log(f'  -> {len(movie_info_list)} 部电影信息已保存')

# ============ Step 2: 用户历史 + 热门电影 ============
log('Step 2: 处理训练集（用户历史 + 热门排行）...')
train = pd.read_csv(os.path.join(MODEL_DIR, 'train.csv'), usecols=['userId', 'movieId', 'rating', 'is_positive'])
log(f'  训练集: {len(train):,} 行')

# 2a: 每个用户喜欢的电影（is_positive=1）
train_pos = train[train['is_positive'] == 1]
user_liked = train_pos.groupby('userId')['movieId'].apply(list).to_dict()
# 转成 {userId: [movieId, ...]} 的紧凑格式
with open(os.path.join(DATA_DIR, 'user_liked.pkl'), 'wb') as f:
    pickle.dump(user_liked, f, protocol=pickle.HIGHEST_PROTOCOL)
log(f'  -> {len(user_liked):,} 个用户的喜好历史已保存')

# 2b: 每个用户看过的所有电影（用于过滤）
user_seen = train.groupby('userId')['movieId'].apply(set).to_dict()
with open(os.path.join(DATA_DIR, 'user_seen.pkl'), 'wb') as f:
    pickle.dump(user_seen, f, protocol=pickle.HIGHEST_PROTOCOL)
log(f'  -> {len(user_seen):,} 个用户的观看历史已保存')

# 2c: 热门电影排行（按评分次数）
movie_rating_count = train.groupby('movieId').agg(
    count=('rating', 'count'),
    mean_rating=('rating', 'mean')
).sort_values('count', ascending=False)

popular_movies = []
for movie_id, row in movie_rating_count.head(100).iterrows():
    info = movie_id_to_info.get(int(movie_id), {})
    popular_movies.append({
        'movieId': int(movie_id),
        'title': info.get('title', 'Unknown'),
        'year': info.get('year', 0),
        'genres': info.get('genres', []),
        'rating_count': int(row['count']),
        'mean_rating': round(float(row['mean_rating']), 2),
        'score': round(float(row['count']) / 1000, 3),  # 归一化分数
    })

with open(os.path.join(DATA_DIR, 'popular_movies.json'), 'w', encoding='utf-8') as f:
    json.dump(popular_movies, f, ensure_ascii=False)
log(f'  -> Top 100 热门电影已保存')

# ============ Step 3: 复制模型文件 ============
log('Step 3: 复制模型文件...')
import shutil

# TF-IDF 矩阵（4MB）
src = os.path.join(MODEL_DIR, 'tfidf_matrix.npz')
dst = os.path.join(DATA_DIR, 'tfidf_matrix.npz')
shutil.copy2(src, dst)
log(f'  -> tfidf_matrix.npz ({os.path.getsize(dst)/1024/1024:.1f} MB)')

# movie_to_idx 映射（241KB）
src = os.path.join(MODEL_DIR, 'movie_to_idx.pkl')
dst = os.path.join(DATA_DIR, 'movie_to_idx.pkl')
shutil.copy2(src, dst)
log(f'  -> movie_to_idx.pkl ({os.path.getsize(dst)/1024:.0f} KB)')

# SVD 微型模型（3.3MB）
if os.path.exists(SVD_PATH):
    dst = os.path.join(DATA_DIR, 'svd_features.pkl')
    shutil.copy2(SVD_PATH, dst)
    log(f'  -> svd_features.pkl ({os.path.getsize(dst)/1024/1024:.1f} MB)')
else:
    log('  -> svd_features.pkl 不存在，跳过（将使用纯内容推荐）')

# ============ Step 4: 预计算内容推荐示例（前1000个用户）============
log('Step 4: 预计算内容推荐（抽样用户）...')

# 加载 TF-IDF 矩阵和映射
tfidf_matrix = sp.load_npz(os.path.join(DATA_DIR, 'tfidf_matrix.npz'))
with open(os.path.join(DATA_DIR, 'movie_to_idx.pkl'), 'rb') as f:
    movie_to_idx = pickle.load(f)

idx_to_movie = np.array([mid for mid, idx in sorted(movie_to_idx.items(), key=lambda x: x[1])])
movie_vectors = sp.csr_matrix(tfidf_matrix)
# L2 归一化，方便用点积算余弦相似度
from sklearn.preprocessing import normalize
movie_vectors_norm = normalize(movie_vectors, norm='l2', axis=1)

# 为前 200 个有历史的用户预计算推荐
sample_users = list(user_liked.keys())[:200]
precomputed_recs = {}

for i, uid in enumerate(sample_users):
    liked_movies = user_liked[uid]
    # 只保留有 TF-IDF 向量的电影
    liked_indices = [movie_to_idx[mid] for mid in liked_movies if mid in movie_to_idx]
    if not liked_indices:
        continue

    # 用户画像 = 喜欢电影的向量平均
    user_vector = movie_vectors[liked_indices].mean(axis=0)
    user_vector = np.asarray(user_vector).flatten()
    # 归一化
    norm = np.linalg.norm(user_vector)
    if norm > 0:
        user_vector = user_vector / norm

    # 计算与所有电影的相似度
    scores = movie_vectors_norm @ user_vector
    scores = np.asarray(scores).flatten()

    # 排除已看过的
    seen = user_seen.get(uid, set())
    seen_indices = set()
    for mid in seen:
        if mid in movie_to_idx:
            seen_indices.add(movie_to_idx[mid])

    # 排序，取 Top-10
    top_indices = np.argsort(scores)[::-1]
    rec_list = []
    for idx in top_indices:
        if len(rec_list) >= 10:
            break
        if idx in seen_indices:
            continue
        mid = int(idx_to_movie[idx])
        info = movie_id_to_info.get(mid, {})
        rec_list.append({
            'movieId': mid,
            'title': info.get('title', 'Unknown'),
            'year': info.get('year', 0),
            'genres': info.get('genres', []),
            'score': round(float(scores[idx]), 4),
        })
    
    if rec_list:
        precomputed_recs[str(uid)] = rec_list

    if (i + 1) % 50 == 0:
        log(f'  已处理 {i+1}/{len(sample_users)} 个用户')

with open(os.path.join(DATA_DIR, 'precomputed_recs.json'), 'w', encoding='utf-8') as f:
    json.dump(precomputed_recs, f, ensure_ascii=False)
log(f'  -> {len(precomputed_recs)} 个用户的预计算推荐已保存')

# ============ Step 5: 按类型统计电影（冷启动用）============
log('Step 5: 按类型整理电影索引...')
genre_movies = {}
for m in movie_info_list:
    for g in m['genres']:
        if g and g != 'Unknown':
            if g not in genre_movies:
                genre_movies[g] = []
            genre_movies[g].append(m['movieId'])

# 每个类型只保留前 50 部（按热门度排序）
genre_top = {}
for genre, mids in genre_movies.items():
    # 按评分次数排序
    rated = [(mid, movie_rating_count.loc[mid, 'count']) for mid in mids if mid in movie_rating_count.index]
    rated.sort(key=lambda x: x[1], reverse=True)
    genre_top[genre] = [mid for mid, _ in rated[:50]]

with open(os.path.join(DATA_DIR, 'genre_movies.json'), 'w', encoding='utf-8') as f:
    json.dump(genre_top, f, ensure_ascii=False)
log(f'  -> {len(genre_top)} 个类型的电影索引已保存')

# ============ 汇总 ============
log('\n========== 预处理完成 ==========')
total_size = 0
for f in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, f)
    size = os.path.getsize(fpath)
    total_size += size
    if size > 1024*1024:
        log(f'  {f}: {size/1024/1024:.1f} MB')
    else:
        log(f'  {f}: {size/1024:.0f} KB')
log(f'  总计: {total_size/1024/1024:.1f} MB')
log('================================')
