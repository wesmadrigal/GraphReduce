# RelBench Rel-Stack with DuckDB (User Engagement)

[![RelBench rel-stack user-engagement graphreduce flow](relbench_rel_stack_user_engagement_duckdb_overview.svg)](relbench_rel_stack_user_engagement_duckdb_overview.svg)

Open full-size: [SVG](relbench_rel_stack_user_engagement_duckdb_overview.svg)

This example implements the RelBench
[user-engagement task](https://relbench.stanford.edu/datasets/rel-stack/#user-engagement):

* parent node: `Users.csv`
* label node: `Posts.csv`
* target: whether user has **any engagement** in next 90 days
  * engagement = post OR vote OR comment
* train/eval cut date: `2020-10-01`
* out-of-time holdout cut date: `2021-01-01`
* lookback period: `10 years` (`3650` days)
* active-user constraint: keep only users with any historical activity
  * active means at least one post, vote, or comment at any point in history

## Complete Example

### Data Preparation

```python
import datetime
from pathlib import Path
from urllib.request import urlretrieve

import duckdb

from graphreduce.graph_reduce import GraphReduce
from graphreduce.node import DuckdbNode
from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.models import sqlop

BASE_URL = "https://open-relbench.s3.us-east-1.amazonaws.com/rel-stack"
TABLES = [
    "Users.csv",
    "Posts.csv",
    "Badges.csv",
    "PostHistory.csv",
    "PostLinks.csv",
    "Votes.csv",
    "Comments.csv",
    "Tags.csv",
]

data_dir = Path("data/relbench/rel-stack")
data_dir.mkdir(parents=True, exist_ok=True)

for table in TABLES:
    out_path = data_dir / table
    if not out_path.exists():
        urlretrieve(f"{BASE_URL}/{table}", out_path)

con = duckdb.connect()
eval_cut_date = datetime.datetime(2020, 10, 1)
holdout_cut_date = datetime.datetime(2021, 1, 1)
cut_date = eval_cut_date

users_fpath = data_dir / "Users.csv"
posts_fpath = data_dir / "Posts.csv"
votes_fpath = data_dir / "Votes.csv"
comments_fpath = data_dir / "Comments.csv"

user = DuckdbNode(
    fpath=f"'{users_fpath}'",
    prefix="user",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
    table_name="users",
    # Keep only users with any activity at any point in history.
    do_filters_ops=[
        sqlop(
            optype=SQLOpType.where,
            opval=f"""(
                user_CreationDate <= '{cut_date.date()}'
                AND (
                    EXISTS (
                        SELECT 1
                        FROM '{posts_fpath}' p
                        WHERE p.OwnerUserId = user_Id
                          AND p.CreationDate < '{cut_date.date()}'
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM '{votes_fpath}' v
                        WHERE v.UserId = user_Id
                          AND v.CreationDate < '{cut_date.date()}'
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM '{comments_fpath}' c
                        WHERE c.UserId = user_Id
                          AND c.CreationDate < '{cut_date.date()}'
                    )
                )
            )""",
        )
    ],
)

post = DuckdbNode(
    fpath=f"'{posts_fpath}'",
    prefix="post",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "OwnerUserId", "PostTypeId", "AcceptedAnswerId", "ParentId", "Title", "Tags", "Body", "CreationDate"],
    table_name="posts",
)

vote = DuckdbNode(
    fpath=f"'{votes_fpath}'",
    prefix="vote",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    table_name="votes",
    # Explicit label ops so votes also contribute future engagement labels.
    do_labels_ops=[
        sqlop(optype=SQLOpType.aggfunc, opval="count(*) as vote_Id_label"),
        sqlop(optype=SQLOpType.agg, opval="vote_UserId"),
    ],
)

comment = DuckdbNode(
    fpath=f"'{comments_fpath}'",
    prefix="comm",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    table_name="comments",
    # Explicit label ops so comments also contribute future engagement labels.
    do_labels_ops=[
        sqlop(optype=SQLOpType.aggfunc, opval="count(*) as comm_Id_label"),
        sqlop(optype=SQLOpType.agg, opval="comm_UserId"),
    ],
)
post_vote = DuckdbNode(
    fpath=f"'{votes_fpath}'",
    prefix="pvote",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "PostId", "VoteTypeId", "UserId", "CreationDate"],
    table_name="votes",
)

post_comment = DuckdbNode(
    fpath=f"'{comments_fpath}'",
    prefix="pcomm",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "PostId", "Text", "CreationDate", "UserId", "ContentLicense"],
    table_name="comments",
)

post_comment_user = DuckdbNode(
    fpath=f"'{users_fpath}'",
    prefix="pcu",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "DisplayName", "Location", "ProfileImageUrl", "WebsiteUrl", "AboutMe", "CreationDate"],
    table_name="users",
)

post_comment_badge = DuckdbNode(
    fpath=f"'{data_dir / 'Badges.csv'}'",
    prefix="pcbad",
    pk="Id",
    date_key="Date",
    columns=["Id", "UserId", "Class", "Name", "Date"],
    table_name="badges",
)

post_history = DuckdbNode(
    fpath=f"'{data_dir / 'PostHistory.csv'}'",
    prefix="ph",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "PostHistoryTypeId", "PostId", "RevisionGUID", "CreationDate", "UserId", "Text", "Comment", "ContentLicense"],
    table_name="post_history",
)

post_links = DuckdbNode(
    fpath=f"'{data_dir / 'PostLinks.csv'}'",
    prefix="plink",
    pk="Id",
    date_key="CreationDate",
    columns=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
    table_name="post_links",
)

tag = DuckdbNode(
    fpath=f"'{data_dir / 'Tags.csv'}'",
    prefix="tag",
    pk="Id",
    date_key=None,
    columns=["Id", "TagName", "Count", "ExcerptPostId", "WikiPostId"],
    table_name="tags",
)

badge = DuckdbNode(
    fpath=f"'{data_dir / 'Badges.csv'}'",
    prefix="bad",
    pk="Id",
    date_key="Date",
    columns=["Id", "UserId", "Class", "Name", "Date"],
    table_name="badges",
)

gr = GraphReduce(
    name="rel-stack-user-engagement",
    parent_node=user,
    compute_layer=ComputeLayerEnum.duckdb,
    sql_client=con,
    cut_date=cut_date,
    compute_period_val=3650,
    compute_period_unit=PeriodUnit.day,
    auto_features=True,
    auto_labels=True,
    date_filters_on_agg=True,
    label_node=post,
    label_field="Id",
    label_operation="count",
    label_period_val=90,
    label_period_unit=PeriodUnit.day,
    auto_feature_hops_back=4,
    auto_feature_hops_front=0,
)

for node in [user, post, vote, comment, post_vote, post_comment, post_comment_user, post_comment_badge, post_history, post_links, tag, badge]:
    gr.add_node(node)

# User-centric relations.
gr.add_entity_edge(parent_node=user, relation_node=post, parent_key="Id", relation_key="OwnerUserId", reduce=True)
gr.add_entity_edge(parent_node=user, relation_node=vote, parent_key="Id", relation_key="UserId", reduce=True)
gr.add_entity_edge(parent_node=user, relation_node=comment, parent_key="Id", relation_key="UserId", reduce=True)
gr.add_entity_edge(parent_node=user, relation_node=badge, parent_key="Id", relation_key="UserId", reduce=True)
gr.add_entity_edge(parent_node=post, relation_node=post_vote, parent_key="Id", relation_key="PostId", reduce=True)
gr.add_entity_edge(parent_node=post, relation_node=post_comment, parent_key="Id", relation_key="PostId", reduce=True)
gr.add_entity_edge(parent_node=post_comment, relation_node=post_comment_user, parent_key="UserId", relation_key="Id", reduce=True)
gr.add_entity_edge(parent_node=post_comment_user, relation_node=post_comment_badge, parent_key="Id", relation_key="UserId", reduce=True)

# Post branches.
gr.add_entity_edge(parent_node=post, relation_node=post_history, parent_key="Id", relation_key="PostId", reduce=True)
gr.add_entity_edge(parent_node=post, relation_node=post_links, parent_key="Id", relation_key="PostId", reduce=True)
gr.add_entity_edge(parent_node=post, relation_node=tag, parent_key="Id", relation_key="ExcerptPostId", reduce=True)

gr.do_transformations_sql()

out_df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df()
print("rows:", len(out_df))
print("columns:", len(out_df.columns))
df = out_df.copy()

post_label_cols = [c for c in df.columns if c.startswith("post_") and "label" in c.lower()]
vote_label_cols = [c for c in df.columns if c.startswith("vote_") and "label" in c.lower()]
comm_label_cols = [c for c in df.columns if c.startswith("comm_") and "label" in c.lower()]
print("post labels:", post_label_cols)
print("vote labels:", vote_label_cols)
print("comment labels:", comm_label_cols)
print(df.head())

# Build a second compute graph for out-of-time holdout:
# re-instantiate nodes with `cut_date = holdout_cut_date` and run
# `gr.do_transformations_sql()` again to produce `df_future`.
```

### Model Training

```python
import numpy as np
from torch_frame.utils import infer_df_stype
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

label_cols = post_label_cols + vote_label_cols + comm_label_cols
if not label_cols:
    raise ValueError("No engagement label columns were found in df.")

for c in label_cols:
    df[c] = df[c].fillna(0)

# Engagement target = any future post/vote/comment in next 90 days.
df["user_had_engagement"] = (df[label_cols].sum(axis=1) > 0).astype("int8")
target = "user_had_engagement"

stypes = infer_df_stype(df_train)
features = [
    k
    for k, v in stypes.items()
    if str(v) == "numerical"
    and k not in ["user_Id", "user_AccountId"]
    and "label" not in k
    and "had_engagement" not in k
]
features = [c for c in features if c in df_train.columns and c in df_future.columns]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    df_train[features],
    df_train[target],
    test_size=0.20,
    stratify=df[target],
    random_state=42,
)

k = 3
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_aucs = []
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X_train_full))
train_pool = Pool(X_train_full, y_train_full)

for fold, (idx_tr, idx_va) in enumerate(skf.split(X_train_full, y_train_full), 1):
    print(f"\n=== Fold {fold} ===")
    X_tr, X_va = X_train_full.iloc[idx_tr], X_train_full.iloc[idx_va]
    y_tr, y_va = y_train_full.iloc[idx_tr], y_train_full.iloc[idx_va]

    mdl = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        custom_metric=["AUC", "PRAUC", "F1", "Recall", "Precision", "Logloss"],
        use_best_model=True,
        iterations=8000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5.0,
        min_data_in_leaf=20,
        boosting_type="Ordered",
        auto_class_weights="Balanced",
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_strength=0.8,
        rsm=0.8,
        feature_border_type="GreedyLogSum",
        od_type="Iter",
        od_wait=250,
        verbose=200,
    )

    mdl.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
        verbose=200,
    )

    val_pred = mdl.predict_proba(X_va)[:, 1]
    val_auc = roc_auc_score(y_va, val_pred)
    fold_aucs.append(val_auc)
    print(f"Fold {fold} validation AUC : {val_auc:.4f}")

    test_preds += mdl.predict_proba(X_test)[:, 1] / k
    oof_preds[idx_va] = val_pred

print("\n=== CV Summary ===")
print(f"Mean CV AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"Folds AUC   : {[f'{a:.4f}' for a in fold_aucs]}")

test_auc = roc_auc_score(y_test, test_preds)
print(f"\nFinal test AUC (averaged over {k} folds): {test_auc:.4f}")

final_mdl = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    custom_metric=["AUC", "PRAUC", "F1", "Recall", "Precision", "Logloss"],
    iterations=int(mdl.best_iteration_ * 1.1),
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=5.0,
    min_data_in_leaf=20,
    boosting_type="Ordered",
    auto_class_weights="Balanced",
    bootstrap_type="Bayesian",
    bagging_temperature=0.5,
    random_strength=0.8,
    rsm=0.8,
    feature_border_type="GreedyLogSum",
    od_type="Iter",
    od_wait=250,
    verbose=200,
)

final_mdl.fit(df_train[features], df_train[target])
future_pred = final_mdl.predict_proba(df_future[features])[:, 1]
print("out_of_time_auc_2021_01_01:", round(roc_auc_score(df_future[target], future_pred), 4))

con.close()
```

## Notes

* `Posts.csv` is the configured `label_node`.
* `Votes.csv` and `Comments.csv` also define `do_labels_ops` so their future
  activity contributes to `user_had_engagement`.
* The final target is a binary union of future post/vote/comment activity.
* Build two separate GraphReduce compute graphs:
  * train/eval graph at `cut_date=2020-10-01`
  * holdout graph at `cut_date=2021-01-01`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_user_engagement">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-stack User Engagement</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
