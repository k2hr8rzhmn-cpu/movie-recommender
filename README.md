# MovieLens 个性化电影推荐系统

## 项目简介

基于 MovieLens ml-latest 数据集构建的个性化电影推荐网站，支持四种推荐模型和冷启动推荐。

## 技术架构

```
movie_recommender/
├── precompute.py          # 离线预处理脚本（运行一次）
├── app.py                 # Flask 后端
├── requirements.txt       # Python 依赖
├── data/                  # 预处理生成的数据文件（~28MB）
│   ├── movie_info.json        # 34208 部电影信息
│   ├── popular_movies.json    # Top 100 热门电影
│   ├── precomputed_recs.json  # 200 个用户的预计算推荐
│   ├── user_liked.pkl         # 46303 个用户的喜好历史
│   ├── user_seen.pkl          # 47043 个用户的观看历史
│   ├── genre_movies.json      # 19 种类型的电影索引
│   ├── tfidf_matrix.npz       # TF-IDF 矩阵 (34208×5000)
│   ├── movie_to_idx.pkl       # movieId → 行索引映射
│   └── svd_features_k30.pkl   # 微型 SVD 模型 (1000用户×30维)
├── templates/
│   └── index.html         # 前端页面
└── test_api.py            # API 测试脚本
```

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 离线预处理（只需运行一次）
```bash
python precompute.py
```
> 此步骤会读取原始训练数据，生成 data/ 目录下的所有文件。

### 3. 启动网站
```bash
python app.py
```

### 4. 访问
浏览器打开 http://localhost:5000

## 功能说明

### 推荐模型
| 模型 | 说明 | 适用场景 |
|------|------|---------|
| Hybrid (α=0.5) | SVD + Content-Based 加权融合 | 有评分历史的老用户 |
| Content-Based | 基于 TF-IDF 内容相似度 | 有历史喜好的用户 |
| SVD | 矩阵分解协同过滤 | SVD 模型中的 1000 个活跃用户 |
| 热门推荐 | 全局评分次数排序 | 基线对比 / 兜底 |
| 冷启动 | 按用户选择的类型筛选热门 | 无评分记录的新用户 |

### API 接口
| 接口 | 参数 | 说明 |
|------|------|------|
| GET /api/recommend | user_id, model | 获取推荐 |
| GET /api/cold_start | genres | 冷启动推荐 |
| GET /api/similar | movie_id | 相似电影 |
| GET /api/user_info | user_id | 用户信息 |
| GET /api/genres | - | 所有电影类型 |
| GET /api/model_results | - | 五模型评估对比 |

## 模型评估结果

| 模型 | Precision@10 | Recall@10 | NDCG@10 |
|------|-------------|-----------|---------|
| 热门推荐（基线） | 0.0433 | 0.0632 | 0.0645 |
| 全局均值（基线） | 0.0000 | 0.0000 | 0.0000 |
| Content-Based | 0.0443 | 0.0535 | 0.0648 |
| SVD | 0.0633 | 0.1007 | 0.0977 |
| **Hybrid (α=0.5)** | **0.0730** | **0.1033** | **0.1095** |

## 数据来源

- 原始数据：MovieLens ml-latest（22,884,377 条评分，34,208 部电影，247,753 名用户）
- 清洗后：过滤低活跃用户 + 20% 采样 → 4,579,789 条评分，47,043 名用户
- 训练/测试集：按用户时间切分（前 80% 训练，后 20% 测试）
