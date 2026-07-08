# -*- coding: utf-8 -*-
"""
电影推荐系统 - Flask 后端
启动: python app.py
访问: http://localhost:5000
"""
import os, json, pickle, time, re
import numpy as np

# numpy 版本兼容：pickle 在 numpy 2.x 生成，兼容 numpy 1.x
import sys
if not hasattr(np, '_core') and hasattr(np, 'core'):
    sys.modules['numpy._core'] = np.core
    np._core = np.core

import scipy.sparse as sp
from sklearn.preprocessing import normalize
from flask import Flask, render_template, jsonify, request
import requests
import urllib.parse
import pandas as pd

app = Flask(__name__)

# ============ 数据目录 ============
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# ============ 启动时加载所有数据 ============
print('正在加载数据...')

# 1. 电影信息
with open(os.path.join(DATA_DIR, 'movie_info.json'), 'r', encoding='utf-8') as f:
    movie_info_list = json.load(f)
movie_id_to_info = {m['movieId']: m for m in movie_info_list}
print(f'  电影信息: {len(movie_info_list)} 部')

# 2. 热门电影
with open(os.path.join(DATA_DIR, 'popular_movies.json'), 'r', encoding='utf-8') as f:
    popular_movies = json.load(f)
print(f'  热门电影: {len(popular_movies)} 部')

# 3. 预计算推荐
with open(os.path.join(DATA_DIR, 'precomputed_recs.json'), 'r', encoding='utf-8') as f:
    precomputed_recs = json.load(f)
print(f'  预计算推荐: {len(precomputed_recs)} 个用户')

# 4. 用户历史
with open(os.path.join(DATA_DIR, 'user_liked.pkl'), 'rb') as f:
    user_liked = pickle.load(f)
with open(os.path.join(DATA_DIR, 'user_seen.pkl'), 'rb') as f:
    user_seen = pickle.load(f)
print(f'  用户喜好历史: {len(user_liked)} 个用户')
print(f'  用户观看历史: {len(user_seen)} 个用户')

# 5. TF-IDF 矩阵 + 映射
tfidf_matrix = sp.load_npz(os.path.join(DATA_DIR, 'tfidf_matrix.npz'))
with open(os.path.join(DATA_DIR, 'movie_to_idx.pkl'), 'rb') as f:
    movie_to_idx = pickle.load(f)
idx_to_movie = np.array([mid for mid, idx in sorted(movie_to_idx.items(), key=lambda x: x[1])])
movie_vectors_norm = normalize(tfidf_matrix, norm='l2', axis=1)
print(f'  TF-IDF 矩阵: {tfidf_matrix.shape}')

# 6. 类型索引
with open(os.path.join(DATA_DIR, 'genre_movies.json'), 'r', encoding='utf-8') as f:
    genre_movies = json.load(f)
print(f'  电影类型: {len(genre_movies)} 种')

# 7. SVD 模型
svd_data = None
svd_path = os.path.join(DATA_DIR, 'svd_features.pkl')
if os.path.exists(svd_path):
    try:
        with open(svd_path, 'rb') as f:
            svd_data = pickle.load(f)
        print(f'  SVD 模型: {len(svd_data.get("user_factors", []))} 个活跃用户')
    except Exception as e:
        print(f'  SVD 模型加载失败（{e}），将禁用 SVD 推荐')

# 8. 电影标签和内容关键词（从 movie_content.csv）
movie_tags = {}
movie_content_text = {}
mc_path = os.path.join(DATA_DIR, 'movie_content.csv')
if os.path.exists(mc_path):
    mc_df = pd.read_csv(mc_path, usecols=['movieId', 'tags_text', 'content'])
    for _, row in mc_df.iterrows():
        mid = int(row['movieId'])
        tags = str(row.get('tags_text', '')).split() if pd.notna(row.get('tags_text')) else []
        content = str(row.get('content', '')).split() if pd.notna(row.get('content')) else []
        movie_tags[mid] = tags[:20]
        movie_content_text[mid] = content[:30]
    print(f'  电影标签: {len(movie_tags)} 部')

# 9. 电影详情缓存（中文名、海报、简介、评分）
DETAILS_CACHE_PATH = os.path.join(DATA_DIR, 'movie_details_cache.json')
movie_details = {}
if os.path.exists(DETAILS_CACHE_PATH):
    with open(DETAILS_CACHE_PATH, 'r', encoding='utf-8') as f:
        movie_details = json.load(f)
    print(f'  电影详情缓存: {len(movie_details)} 部')

# 9b. 独立中文名映射（不受服务器缓存覆写影响）
CN_TITLES_PATH = os.path.join(DATA_DIR, 'chinese_titles.json')
cn_titles_map = {}
if os.path.exists(CN_TITLES_PATH):
    with open(CN_TITLES_PATH, 'r', encoding='utf-8') as f:
        cn_titles_map = json.load(f)
    print(f'  中文名映射: {len(cn_titles_map)} 部')

# 10. 旧版 TMDB 缓存（兼容）
TMDB_CACHE_PATH = os.path.join(DATA_DIR, 'tmdb_cache.json')
tmdb_cache = {}
if os.path.exists(TMDB_CACHE_PATH):
    with open(TMDB_CACHE_PATH, 'r', encoding='utf-8') as f:
        tmdb_cache = json.load(f)

# Hybrid 最优 alpha（王雅萱 7-8 调参结果）
HYBRID_ALPHA = 0.7

print('数据加载完成!\n')


# ============ 工具函数 ============

def _strip_year(title):
    """去掉标题中的年份括号"""
    return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

def _normalize_title(title):
    """修正 'X, The' -> 'The X'"""
    clean = _strip_year(title)
    if clean.endswith(', The'):
        clean = 'The ' + clean[:-5]
    elif clean.endswith(', A'):
        clean = 'A ' + clean[:-3]
    elif clean.endswith(', An'):
        clean = 'An ' + clean[:-4]
    return clean

def _get_chinese_title(movie_id):
    """获取中文标题：优先详情缓存，其次独立映射"""
    mid_str = str(movie_id)
    # 1. 从详情缓存获取
    if mid_str in movie_details:
        cn = movie_details[mid_str].get('chinese_title', '')
        if cn and not cn.isascii():
            return cn
    # 2. 从独立映射获取（不受缓存覆写影响）
    if mid_str in cn_titles_map:
        return cn_titles_map[mid_str]
    return ''

def _get_bilingual_title(movie_id):
    """获取中英文双语标题：中文名（英文名）"""
    info = get_movie_info(movie_id)
    en_title = _normalize_title(info['title'])
    cn_title = _get_chinese_title(movie_id)
    if cn_title:
        return f'{cn_title}（{en_title}）'
    return en_title

def _get_movie_detail(movie_id):
    """从缓存获取电影详情（中文名、海报、简介、评分）"""
    mid_str = str(movie_id)
    if mid_str in movie_details:
        return movie_details[mid_str]
    # 旧版 TMDB 缓存兼容
    if mid_str in tmdb_cache:
        return tmdb_cache[mid_str]
    return None

def get_movie_info(movie_id):
    """获取电影基本信息"""
    return movie_id_to_info.get(int(movie_id), {
        'movieId': int(movie_id),
        'title': f'Movie #{movie_id}',
        'year': 0,
        'genres': [],
    })

def _build_movie_record(movie_id, score=0, model=''):
    """构建带双语标题和详情的电影记录"""
    info = get_movie_info(movie_id)
    detail = _get_movie_detail(movie_id)
    
    record = {
        'movieId': movie_id,
        'title': info['title'],
        'bilingual_title': _get_bilingual_title(movie_id),
        'year': info['year'],
        'genres': info['genres'],
        'score': score,
        'model': model,
    }
    
    # 从详情缓存补充海报、简介、评分
    cn_from_map = _get_chinese_title(movie_id)
    if detail:
        record['poster_url'] = detail.get('poster_url', '')
        record['overview'] = detail.get('overview', '')
        record['douban_rating'] = detail.get('douban_rating', '')
        record['imdb_rating'] = detail.get('imdb_rating', '')
        record['chinese_title'] = detail.get('chinese_title', '') or cn_from_map
        record['douban_url'] = detail.get('douban_url', '')
        record['genre_cn'] = detail.get('genre_cn', '')
    else:
        record['poster_url'] = ''
        record['overview'] = ''
        record['douban_rating'] = ''
        record['chinese_title'] = cn_from_map
        clean_title = _normalize_title(info['title'])
        record['douban_url'] = f'https://search.douban.com/movie/subject_search?search_text={urllib.parse.quote(clean_title)}&cat=1002'
    
    # 本地标签
    record['tags'] = movie_tags.get(int(movie_id), [])
    
    return record


# ============ 推荐算法 ============

def content_based_recommend(user_id, top_n=10):
    """实时内容推荐"""
    if user_id not in user_liked:
        return None
    liked_movies = user_liked[user_id]
    liked_indices = [movie_to_idx[mid] for mid in liked_movies if mid in movie_to_idx]
    if not liked_indices:
        return None

    user_vector = tfidf_matrix[liked_indices].mean(axis=0)
    user_vector = np.asarray(user_vector).flatten()
    norm = np.linalg.norm(user_vector)
    if norm > 0:
        user_vector = user_vector / norm

    scores = movie_vectors_norm @ user_vector
    scores = np.asarray(scores).flatten()

    seen = user_seen.get(user_id, set())
    seen_indices = set()
    for mid in seen:
        if mid in movie_to_idx:
            seen_indices.add(movie_to_idx[mid])

    top_indices = np.argsort(scores)[::-1]
    results = []
    for idx in top_indices:
        if len(results) >= top_n:
            break
        if idx in seen_indices:
            continue
        mid = int(idx_to_movie[idx])
        results.append(_build_movie_record(mid, round(float(scores[idx]), 4), 'Content-Based'))
    return results


def svd_recommend(user_id, top_n=10):
    """SVD 推荐"""
    if svd_data is None:
        return None
    user_factors = svd_data['user_factors']
    movie_factors = svd_data['movie_factors']
    svd_user_to_idx = svd_data['user_to_idx']
    svd_movie_list = svd_data.get('movie_list', list(range(movie_factors.shape[0])))

    uid_key = user_id
    if uid_key not in svd_user_to_idx:
        uid_key = np.int64(user_id)
    if uid_key not in svd_user_to_idx:
        return None

    row = svd_user_to_idx[uid_key]
    user_vec = user_factors[row]
    pred_scores = movie_factors @ user_vec
    min_s, max_s = pred_scores.min(), pred_scores.max()
    if max_s > min_s:
        pred_scores = (pred_scores - min_s) / (max_s - min_s)

    seen = user_seen.get(user_id, set())
    top_indices = np.argsort(pred_scores)[::-1]
    results = []
    for idx in top_indices:
        if len(results) >= top_n:
            break
        if idx < len(svd_movie_list):
            mid = int(svd_movie_list[idx])
        else:
            continue
        if mid in seen:
            continue
        results.append(_build_movie_record(mid, round(float(pred_scores[idx]), 4), 'SVD'))
    return results


def hybrid_recommend(user_id, alpha=HYBRID_ALPHA, top_n=10):
    """Hybrid 推荐：SVD + Content-Based 加权融合 (α=0.7 最优)"""
    svd_recs = svd_recommend(user_id, top_n=50)
    cb_recs = content_based_recommend(user_id, top_n=50)

    if svd_recs is None and cb_recs is None:
        return None
    if svd_recs is None:
        for r in cb_recs[:top_n]:
            r['model'] = 'Hybrid (Content only)'
        return cb_recs[:top_n]
    if cb_recs is None:
        for r in svd_recs[:top_n]:
            r['model'] = 'Hybrid (SVD only)'
        return svd_recs[:top_n]

    svd_dict = {r['movieId']: r['score'] for r in svd_recs}
    cb_dict = {r['movieId']: r['score'] for r in cb_recs}
    all_movies = set(svd_dict.keys()) | set(cb_dict.keys())

    hybrid_scores = []
    for mid in all_movies:
        svd_s = svd_dict.get(mid, 0)
        cb_s = cb_dict.get(mid, 0)
        final = alpha * svd_s + (1 - alpha) * cb_s
        hybrid_scores.append((mid, final))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for mid, score in hybrid_scores[:top_n]:
        results.append(_build_movie_record(mid, round(float(score), 4), 'Hybrid'))
    return results


def cold_start_recommend(genres, top_n=10):
    """冷启动推荐：按类型筛选热门电影"""
    candidate_ids = set()
    for g in genres:
        if g in genre_movies:
            if not candidate_ids:
                candidate_ids = set(genre_movies[g])
            else:
                candidate_ids &= set(genre_movies[g])
    if not candidate_ids:
        for g in genres:
            if g in genre_movies:
                candidate_ids.update(genre_movies[g])

    results = []
    for mid in candidate_ids:
        score = 0
        for i, pm in enumerate(popular_movies):
            if pm['movieId'] == mid:
                score = round(1.0 - i * 0.01, 4)
                break
        if score == 0:
            score = 0.1
        results.append(_build_movie_record(mid, score, 'Cold Start'))

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def similar_movies(movie_id, top_n=10):
    """找相似电影"""
    if movie_id not in movie_to_idx:
        return []
    idx = movie_to_idx[movie_id]
    movie_vec = np.asarray(movie_vectors_norm[idx].todense()).flatten()
    scores = movie_vectors_norm @ movie_vec
    scores = np.asarray(scores).flatten()

    top_indices = np.argsort(scores)[::-1]
    results = []
    for i in top_indices:
        if len(results) >= top_n:
            break
        if i == idx:
            continue
        mid = int(idx_to_movie[i])
        results.append(_build_movie_record(mid, round(float(scores[i]), 4), 'Similar'))
    return results


# ============ 实时豆瓣查询（按需） ============

DB_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://movie.douban.com/',
    'Accept': 'application/json, text/plain, */*',
}

def _fetch_douban_live(title, year=0):
    """实时从豆瓣 suggest API 查询电影信息"""
    clean = _normalize_title(title)
    try:
        resp = requests.get(
            'https://movie.douban.com/j/subject_suggest',
            params={'q': clean},
            headers=DB_HEADERS,
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                best = None
                for item in data:
                    item_year = item.get('year', '')
                    if year and item_year and str(year) == str(item_year):
                        best = item
                        break
                if not best:
                    for item in data:
                        item_year = item.get('year', '')
                        if year and item_year and abs(int(item_year) - year) <= 2:
                            best = item
                            break
                if not best:
                    best = data[0]
                return {
                    'chinese_title': best.get('title', ''),
                    'douban_id': best.get('id', ''),
                    'poster_url': best.get('img', ''),
                    'douban_url': best.get('url', ''),
                }
        return None
    except:
        return None


def _fetch_wmdb_live(chinese_title, english_title, year=0):
    """实时从 wmdb.tv 查询电影简介（限流：10秒1次）"""
    search_q = chinese_title if chinese_title else _normalize_title(english_title)
    try:
        resp = requests.get(
            'https://api.wmdb.tv/api/v1/movie/search',
            params={'q': search_q, 'limit': 5, 'lang': 'Cn'},
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=12
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('data', [])
            if results:
                best = None
                for r in results:
                    r_year = r.get('year', '')
                    if year and r_year and str(year) in str(r_year):
                        best = r
                        break
                if not best:
                    best = results[0]
                data_items = best.get('data', [])
                overview = ''
                if data_items and len(data_items) > 0:
                    overview = data_items[0].get('description', '')
                return {
                    'overview': overview[:500] if overview else '',
                    'douban_rating': best.get('doubanRating', ''),
                    'imdb_rating': best.get('imdbRating', ''),
                    'genre_cn': data_items[0].get('genre', '') if data_items else '',
                }
        return None
    except:
        return None


def _save_details_cache():
    """保存详情缓存"""
    with open(DETAILS_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(movie_details, f, ensure_ascii=False)


# ============ API 路由 ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """推荐接口"""
    user_id = request.args.get('user_id', type=int)
    model = request.args.get('model', 'hybrid')
    if user_id is None:
        return jsonify({'error': '请提供 user_id'}), 400

    start_time = time.time()
    has_history = user_id in user_liked
    results = None
    model_used = model

    if model == 'popular':
        results = [_build_movie_record(m['movieId'], m['score'], 'Popular') for m in popular_movies[:10]]
        model_used = '热门推荐'
    elif model == 'content':
        results = content_based_recommend(user_id, top_n=10)
        model_used = 'Content-Based'
    elif model == 'svd':
        results = svd_recommend(user_id, top_n=10)
        model_used = 'SVD'
    elif model == 'hybrid':
        if str(user_id) in precomputed_recs:
            cached = precomputed_recs[str(user_id)][:10]
            results = []
            for r in cached:
                results.append(_build_movie_record(r['movieId'], r.get('score', 0), 'Hybrid (预计算)'))
            model_used = 'Hybrid (预计算)'
        else:
            results = hybrid_recommend(user_id, alpha=HYBRID_ALPHA, top_n=10)
            model_used = f'Hybrid (α={HYBRID_ALPHA})'

    elapsed = round(time.time() - start_time, 3)

    if results is None:
        return jsonify({
            'error': '该用户没有历史评分记录，请使用冷启动推荐或新用户引导',
            'user_id': user_id,
            'has_history': False,
            'need_cold_start': True,
        })

    return jsonify({
        'user_id': user_id,
        'has_history': has_history,
        'model': model_used,
        'count': len(results),
        'elapsed': elapsed,
        'recommendations': results,
    })


@app.route('/api/cold_start', methods=['GET'])
def api_cold_start():
    """冷启动推荐"""
    genres_str = request.args.get('genres', '')
    genres = [g.strip() for g in genres_str.split(',') if g.strip()]
    if not genres:
        return jsonify({'error': '请选择至少一个类型'}), 400
    results = cold_start_recommend(genres, top_n=10)
    return jsonify({'genres': genres, 'count': len(results), 'recommendations': results})


@app.route('/api/similar', methods=['GET'])
def api_similar():
    """相似电影"""
    movie_id = request.args.get('movie_id', type=int)
    if movie_id is None:
        return jsonify({'error': '请提供 movie_id'}), 400
    results = similar_movies(movie_id, top_n=10)
    return jsonify({'movie_id': movie_id, 'count': len(results), 'recommendations': results})


@app.route('/api/movie_detail', methods=['GET'])
def api_movie_detail():
    """获取电影详情（缓存优先，按需查询豆瓣/wmdb）"""
    movie_id = request.args.get('movie_id', type=int)
    if movie_id is None:
        return jsonify({'error': '请提供 movie_id'}), 400

    record = _build_movie_record(movie_id)
    mid_str = str(movie_id)

    # 如果缓存中已有中文名和简介，直接返回
    if mid_str in movie_details and movie_details[mid_str].get('chinese_title') and movie_details[mid_str].get('overview'):
        return jsonify(record)

    # 按需查询豆瓣（快速获取中文名和海报）
    if not record.get('chinese_title'):
        db_data = _fetch_douban_live(record['title'], record.get('year', 0))
        if db_data:
            record['chinese_title'] = db_data['chinese_title']
            record['poster_url'] = db_data['poster_url']
            record['douban_url'] = db_data['douban_url']
            # 更新双语标题
            if db_data['chinese_title']:
                en_title = _normalize_title(record['title'])
                record['bilingual_title'] = f'{db_data["chinese_title"]}（{en_title}）'
            # 缓存
            movie_details[mid_str] = {
                'movieId': movie_id,
                'title': record['title'],
                'clean_title': _strip_year(record['title']),
                'year': record.get('year', 0),
                'chinese_title': db_data['chinese_title'],
                'douban_id': db_data['douban_id'],
                'poster_url': db_data['poster_url'],
                'douban_url': db_data['douban_url'],
                'overview': '',
                'douban_rating': '',
            }
            _save_details_cache()

    return jsonify(record)


@app.route('/api/search', methods=['GET'])
def api_search():
    """搜索电影：支持中英文关键词"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': '请输入搜索关键词'}), 400

    query_lower = query.lower()
    results = []

    # 搜索本地数据库（中文名 + 英文名）
    for mid_str, detail in movie_details.items():
        cn = detail.get('chinese_title', '').lower()
        en = detail.get('clean_title', _normalize_title(detail.get('title', ''))).lower()
        if query_lower in cn or query_lower in en:
            mid = int(mid_str)
            results.append(_build_movie_record(mid, 0, '搜索'))
            if len(results) >= 20:
                break

    # 如果本地缓存不够，搜索 movie_info_list
    if len(results) < 10:
        for m in movie_info_list:
            if len(results) >= 20:
                break
            en_title = _normalize_title(m['title']).lower()
            if query_lower in en_title:
                mid = m['movieId']
                # 避免重复
                if not any(r['movieId'] == mid for r in results):
                    results.append(_build_movie_record(mid, 0, '搜索'))

    return jsonify({
        'query': query,
        'count': len(results),
        'recommendations': results,
    })


@app.route('/api/homepage', methods=['GET'])
def api_homepage():
    """首页推荐：精选电影 + 按类型推荐"""
    # 精选电影（热门 Top 12，带详情）
    featured = []
    for pm in popular_movies[:12]:
        featured.append(_build_movie_record(pm['movieId'], pm.get('score', 0), '热门'))

    # 按类型推荐（每个类型 Top 6）
    genre_picks = {}
    top_genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Animation', 'Thriller']
    for genre in top_genres:
        if genre in genre_movies:
            picks = []
            for mid in genre_movies[genre][:6]:
                picks.append(_build_movie_record(mid, 0, genre))
            genre_picks[genre] = picks

    # 最新热门（2000年后）
    recent_popular = []
    for pm in popular_movies:
        if pm.get('year', 0) >= 2000:
            recent_popular.append(_build_movie_record(pm['movieId'], pm.get('score', 0), '近期热门'))
            if len(recent_popular) >= 6:
                break

    return jsonify({
        'featured': featured,
        'genre_picks': genre_picks,
        'recent_popular': recent_popular,
    })


@app.route('/api/user_info', methods=['GET'])
def api_user_info():
    """获取用户信息"""
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({'error': '请提供 user_id'}), 400

    has_liked = user_id in user_liked
    has_seen = user_id in user_seen
    liked_count = len(user_liked.get(user_id, []))
    seen_count = len(user_seen.get(user_id, set()))

    liked_movies = []
    if has_liked:
        for mid in user_liked[user_id][:5]:
            liked_movies.append(_build_movie_record(mid))

    in_svd = svd_data is not None and (user_id in svd_data.get('user_to_idx', {}) or np.int64(user_id) in svd_data.get('user_to_idx', {}))
    has_precomputed = str(user_id) in precomputed_recs

    return jsonify({
        'user_id': user_id,
        'exists': has_liked or has_seen,
        'liked_count': liked_count,
        'seen_count': seen_count,
        'in_svd_model': in_svd,
        'has_precomputed': has_precomputed,
        'sample_liked': liked_movies,
    })


@app.route('/api/genres', methods=['GET'])
def api_genres():
    """获取所有电影类型"""
    genres = sorted(genre_movies.keys())
    genre_counts = {g: len(genre_movies[g]) for g in genres}
    return jsonify({'genres': genres, 'counts': genre_counts})


@app.route('/api/model_results', methods=['GET'])
def api_model_results():
    """返回模型评估对比结果（含 α 调参数据）"""
    return jsonify({
        'models': [
            {'name': '热门推荐', 'label': 'Popular (基线)', 'precision': 0.0433, 'recall': 0.0632, 'ndcg': 0.0645, 'color': '#95a5a6'},
            {'name': '全局均值', 'label': 'Global Avg (基线)', 'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'color': '#bdc3c7'},
            {'name': 'Content-Based', 'label': 'Content-Based', 'precision': 0.0443, 'recall': 0.0535, 'ndcg': 0.0648, 'color': '#3498db'},
            {'name': 'SVD', 'label': 'SVD', 'precision': 0.0633, 'recall': 0.1007, 'ndcg': 0.0977, 'color': '#e74c3c'},
            {'name': 'Hybrid α=0.5', 'label': 'Hybrid (α=0.5)', 'precision': 0.073, 'recall': 0.1033, 'ndcg': 0.1095, 'color': '#2ecc71'},
            {'name': 'Hybrid α=0.7', 'label': 'Hybrid (α=0.7 最优)', 'precision': 0.076, 'recall': 0.1104, 'ndcg': 0.1122, 'color': '#f1c40f'},
        ],
        'alpha_tuning': [
            {'alpha': 0.1, 'precision': 0.0506, 'recall': 0.0624, 'ndcg': 0.0744},
            {'alpha': 0.3, 'precision': 0.0628, 'recall': 0.0833, 'ndcg': 0.0944},
            {'alpha': 0.5, 'precision': 0.073, 'recall': 0.1033, 'ndcg': 0.1095},
            {'alpha': 0.6, 'precision': 0.0757, 'recall': 0.1091, 'ndcg': 0.1125},
            {'alpha': 0.7, 'precision': 0.076, 'recall': 0.1104, 'ndcg': 0.1122},
            {'alpha': 0.8, 'precision': 0.0749, 'recall': 0.1085, 'ndcg': 0.11},
            {'alpha': 0.9, 'precision': 0.0732, 'recall': 0.1061, 'ndcg': 0.1072},
        ],
    })


@app.route('/api/onboarding_movies', methods=['GET'])
def api_onboarding_movies():
    """新用户引导：返回多样化热门电影（偏好2000年后）"""
    import random
    target_genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance',
                     'Thriller', 'Animation', 'Adventure', 'Crime', 'Fantasy']
    selected = []
    seen_ids = set()

    for genre in target_genres:
        if genre not in genre_movies:
            continue
        # 优先选2000年后的电影
        candidates = []
        for mid in genre_movies[genre][:30]:
            info = get_movie_info(mid)
            if info.get('year', 0) >= 2000:
                candidates.append(mid)
        # 不足则补充2000年前的
        if len(candidates) < 3:
            for mid in genre_movies[genre][:20]:
                if mid not in candidates:
                    candidates.append(mid)
                if len(candidates) >= 5:
                    break

        picks = random.sample(candidates, min(2, len(candidates)))
        for mid in picks:
            if mid in seen_ids:
                continue
            selected.append(_build_movie_record(mid, 0, ''))
            seen_ids.add(mid)
            if len(selected) >= 18:
                break
        if len(selected) >= 18:
            break

    # 不足时从2000年后的热门电影补充
    if len(selected) < 12:
        for pm in popular_movies:
            if pm['movieId'] in seen_ids:
                continue
            if pm.get('year', 0) >= 2000:
                selected.append(_build_movie_record(pm['movieId'], pm.get('score', 0), ''))
                seen_ids.add(pm['movieId'])
                if len(selected) >= 18:
                    break

    return jsonify({'movies': selected, 'total': len(selected)})


@app.route('/api/new_user_recommend', methods=['POST'])
def api_new_user_recommend():
    """基于新用户评分实时构建画像并生成推荐"""
    data = request.get_json()
    if not data or 'ratings' not in data:
        return jsonify({'error': '请提交评分数据'}), 400

    ratings = data['ratings']
    if not ratings or not isinstance(ratings, list):
        return jsonify({'error': '请至少评价一部电影'}), 400

    start_time = time.time()
    liked_movie_ids = [int(r['movieId']) for r in ratings if float(r.get('rating', 0)) >= 3.5]
    disliked_movie_ids = [int(r['movieId']) for r in ratings if 0 < float(r.get('rating', 0)) < 3.5]
    all_rated_ids = set(int(r['movieId']) for r in ratings if float(r.get('rating', 0)) > 0)

    if not liked_movie_ids:
        return jsonify({
            'error': '请至少给一部电影 3.5 星以上的好评，这样我们才能了解你的口味偏好',
            'liked_count': 0,
            'rated_count': len(all_rated_ids),
        })

    liked_indices = [movie_to_idx[mid] for mid in liked_movie_ids if mid in movie_to_idx]
    if not liked_indices:
        return jsonify({'error': '无法处理这些电影的数据，请尝试评价更多电影'}), 400

    user_vector = tfidf_matrix[liked_indices].mean(axis=0)
    user_vector = np.asarray(user_vector).flatten()

    disliked_indices = [movie_to_idx[mid] for mid in disliked_movie_ids if mid in movie_to_idx]
    if disliked_indices:
        disliked_vector = tfidf_matrix[disliked_indices].mean(axis=0)
        disliked_vector = np.asarray(disliked_vector).flatten()
        user_vector = user_vector - 0.3 * disliked_vector

    norm = np.linalg.norm(user_vector)
    if norm > 0:
        user_vector = user_vector / norm

    scores = movie_vectors_norm @ user_vector
    scores = np.asarray(scores).flatten()

    top_indices = np.argsort(scores)[::-1]
    results = []
    for idx in top_indices:
        if len(results) >= 10:
            break
        mid = int(idx_to_movie[idx])
        if mid in all_rated_ids:
            continue
        results.append(_build_movie_record(mid, round(float(scores[idx]), 4), '新用户推荐 (Content-Based)'))

    elapsed = round(time.time() - start_time, 3)

    # 口味画像
    genre_pref = {}
    for mid in liked_movie_ids:
        info = get_movie_info(mid)
        for g in info.get('genres', []):
            genre_pref[g] = genre_pref.get(g, 0) + 1
    top_genres = sorted(genre_pref.items(), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        'liked_count': len(liked_movie_ids),
        'disliked_count': len(disliked_movie_ids),
        'rated_count': len(all_rated_ids),
        'taste_profile': [g for g, _ in top_genres],
        'model': '新用户推荐 (Content-Based)',
        'count': len(results),
        'elapsed': elapsed,
        'recommendations': results,
    })


@app.route('/api/add_rating', methods=['POST'])
def api_add_rating():
    """老用户补评：已有用户对新电影评分，更新推荐"""
    data = request.get_json()
    if not data or 'user_id' not in data or 'ratings' not in data:
        return jsonify({'error': '请提供 user_id 和 ratings'}), 400

    user_id = data['user_id']
    new_ratings = data['ratings']

    if user_id not in user_liked:
        user_liked[user_id] = []
    if user_id not in user_seen:
        user_seen[user_id] = set()

    added_count = 0
    for r in new_ratings:
        mid = int(r['movieId'])
        rating = float(r.get('rating', 0))
        if rating >= 3.5 and mid not in user_liked[user_id]:
            user_liked[user_id].append(mid)
            added_count += 1
        user_seen[user_id].add(mid)

    # 实时重新计算推荐
    results = hybrid_recommend(user_id, alpha=HYBRID_ALPHA, top_n=10)

    return jsonify({
        'user_id': user_id,
        'added_count': added_count,
        'total_liked': len(user_liked[user_id]),
        'total_seen': len(user_seen[user_id]),
        'model': f'Hybrid (α={HYBRID_ALPHA}, 已更新偏好)',
        'count': len(results) if results else 0,
        'recommendations': results or [],
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    print('\n========================================')
    print('  电影推荐系统已启动')
    print(f'  Hybrid α = {HYBRID_ALPHA} (调参最优)')
    print(f'  访问地址: http://localhost:{port}')
    print(f'  Debug: {debug_mode}')
    print('  按 Ctrl+C 停止')
    print('========================================\n')
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
