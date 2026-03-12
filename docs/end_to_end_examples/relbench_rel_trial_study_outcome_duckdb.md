# RelBench Rel-Trial with DuckDB (Study Outcome Label) (Classification)

This example builds a table-level GraphReduce compute graph for the RelBench
`rel-trial` dataset, rooted at `studies.parquet`, and predicts whether a study
achieves its primary outcome in the next year.

It uses the same split points as RelBench:

* Validation cut date: `2020-01-01`
* Holdout cut date: `2021-01-01`

Local dataset path:
`tests/data/relbench/rel-trial`

Tables used:

* `studies.parquet`
* `outcomes.parquet`
* `outcome_analyses.parquet`
* `drop_withdrawals.parquet`
* `reported_event_totals.parquet`
* `designs.parquet`
* `eligibilities.parquet`
* `interventions.parquet`
* `conditions.parquet`
* `facilities.parquet`
* `sponsors.parquet`
* `interventions_studies.parquet`
* `conditions_studies.parquet`
* `facilities_studies.parquet`
* `sponsors_studies.parquet`

## Complete Example

### Data Preparation + Graph Construction

<details>
<summary>Show Code</summary>

```python
import datetime
from pathlib import Path

import duckdb

from graphreduce.enum import ComputeLayerEnum, PeriodUnit, SQLOpType
from graphreduce.graph_reduce import GraphReduce
from graphreduce.models import sqlop
from graphreduce.node import DuckdbNode

TABLES = [
    "studies.parquet",
    "outcomes.parquet",
    "outcome_analyses.parquet",
    "drop_withdrawals.parquet",
    "reported_event_totals.parquet",
    "designs.parquet",
    "eligibilities.parquet",
    "interventions.parquet",
    "conditions.parquet",
    "facilities.parquet",
    "sponsors.parquet",
    "interventions_studies.parquet",
    "conditions_studies.parquet",
    "facilities_studies.parquet",
    "sponsors_studies.parquet",
]

VAL_TIMESTAMP = datetime.datetime(2020, 1, 1)
TEST_TIMESTAMP = datetime.datetime(2021, 1, 1)
LOOKBACK_START = datetime.datetime(2000, 1, 1)
LABEL_DAYS = 365


def build_frame(con: duckdb.DuckDBPyConnection, data_dir: Path, cut_date: datetime.datetime):
    for table in TABLES:
        out_path = data_dir / table
        if not out_path.exists():
            raise FileNotFoundError(f"Missing local file: {out_path}")

    def prepare_view(view_name: str, table_name: str):
        con.sql(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{data_dir / table_name}')")

    prepare_view("studies_src", "studies.parquet")
    prepare_view("outcomes_src", "outcomes.parquet")
    prepare_view("outcome_analyses_src", "outcome_analyses.parquet")
    prepare_view("drop_withdrawals_src", "drop_withdrawals.parquet")
    prepare_view("reported_event_totals_src", "reported_event_totals.parquet")
    prepare_view("designs_src", "designs.parquet")
    prepare_view("eligibilities_src", "eligibilities.parquet")
    prepare_view("interventions_src", "interventions.parquet")
    prepare_view("conditions_src", "conditions.parquet")
    prepare_view("facilities_src", "facilities.parquet")
    prepare_view("sponsors_src", "sponsors.parquet")
    prepare_view("interventions_studies_src", "interventions_studies.parquet")
    prepare_view("conditions_studies_src", "conditions_studies.parquet")
    prepare_view("facilities_studies_src", "facilities_studies.parquet")
    prepare_view("sponsors_studies_src", "sponsors_studies.parquet")

    studies = DuckdbNode(
        fpath="studies_src",
        prefix="std",
        pk="nct_id",
        date_key="start_date",
        columns=con.sql("select * from studies_src limit 0").to_df().columns.tolist(),
        do_filters_ops=[
            sqlop(optype=SQLOpType.where, opval="std_nct_id is not null"),
            sqlop(optype=SQLOpType.where, opval=f"std_start_date <= '{cut_date.date()}'"),
        ],
    )

    outcomes = DuckdbNode(
        fpath="outcomes_src",
        prefix="out",
        pk="id",
        date_key="date",
        columns=con.sql("select * from outcomes_src limit 0").to_df().columns.tolist(),
    )

    outcome_analyses = DuckdbNode(
        fpath="outcome_analyses_src",
        prefix="oa",
        pk="id",
        date_key="date",
        columns=con.sql("select * from outcome_analyses_src limit 0").to_df().columns.tolist(),
        do_annotate_ops=[
            sqlop(
                optype=SQLOpType.select,
                opval=(
                    "*, "
                    "case when exists ("
                    "select 1 from outcomes_src o "
                    "where o.id = oa_outcome_id "
                    "and lower(cast(o.outcome_type as varchar)) = 'primary'"
                    ") then 1 else 0 end as oa_is_primary, "
                    "case when (oa_p_value_modifier is null or oa_p_value_modifier != '>') "
                    "and try_cast(oa_p_value as double) between 0 and 1 "
                    "then 1 else 0 end as oa_valid_p_value"
                ),
            )
        ],
        do_labels_ops=[
            sqlop(
                optype=SQLOpType.aggfunc,
                opval=(
                    "max(case when oa_is_primary = 1 and oa_valid_p_value = 1 "
                    "and try_cast(oa_p_value as double) <= 0.05 then 1 else 0 end) "
                    "as oa_primary_outcome_success_label"
                ),
            ),
            sqlop(optype=SQLOpType.agg, opval="oa_nct_id"),
        ],
    )

    # Remaining tables are explicit table-level nodes.
    drop_withdrawals = DuckdbNode("drop_withdrawals_src", "drw", "id", "date", con.sql("select * from drop_withdrawals_src limit 0").to_df().columns.tolist())
    reported_event_totals = DuckdbNode("reported_event_totals_src", "evt", "id", "date", con.sql("select * from reported_event_totals_src limit 0").to_df().columns.tolist())
    designs = DuckdbNode("designs_src", "dsg", "id", "date", con.sql("select * from designs_src limit 0").to_df().columns.tolist())
    eligibilities = DuckdbNode("eligibilities_src", "eli", "id", "date", con.sql("select * from eligibilities_src limit 0").to_df().columns.tolist())
    interventions_studies = DuckdbNode("interventions_studies_src", "ist", "id", "date", con.sql("select * from interventions_studies_src limit 0").to_df().columns.tolist())
    conditions_studies = DuckdbNode("conditions_studies_src", "cst", "id", "date", con.sql("select * from conditions_studies_src limit 0").to_df().columns.tolist())
    facilities_studies = DuckdbNode("facilities_studies_src", "fst", "id", "date", con.sql("select * from facilities_studies_src limit 0").to_df().columns.tolist())
    sponsors_studies = DuckdbNode("sponsors_studies_src", "sst", "id", "date", con.sql("select * from sponsors_studies_src limit 0").to_df().columns.tolist())
    interventions = DuckdbNode("interventions_src", "intv", "intervention_id", None, con.sql("select * from interventions_src limit 0").to_df().columns.tolist())
    conditions = DuckdbNode("conditions_src", "cond", "condition_id", None, con.sql("select * from conditions_src limit 0").to_df().columns.tolist())
    facilities = DuckdbNode("facilities_src", "fac", "facility_id", None, con.sql("select * from facilities_src limit 0").to_df().columns.tolist())
    sponsors = DuckdbNode("sponsors_src", "spn", "sponsor_id", None, con.sql("select * from sponsors_src limit 0").to_df().columns.tolist())

    gr = GraphReduce(
        name=f"rel_trial_study_outcome_{cut_date.date()}",
        parent_node=studies,
        compute_layer=ComputeLayerEnum.duckdb,
        sql_client=con,
        cut_date=cut_date,
        compute_period_val=(cut_date - LOOKBACK_START).days + 1,
        compute_period_unit=PeriodUnit.day,
        auto_features=True,
        date_filters_on_agg=True,
        label_node=outcome_analyses,
        label_period_val=LABEL_DAYS,
        label_period_unit=PeriodUnit.day,
        auto_feature_hops_back=4,
        auto_feature_hops_front=0,
    )

    for node in [
        studies,
        outcomes,
        outcome_analyses,
        drop_withdrawals,
        reported_event_totals,
        designs,
        eligibilities,
        interventions_studies,
        conditions_studies,
        facilities_studies,
        sponsors_studies,
        interventions,
        conditions,
        facilities,
        sponsors,
    ]:
        gr.add_node(node)

    # Table-level graph edges rooted at studies.
    gr.add_entity_edge(studies, outcomes, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, outcome_analyses, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, drop_withdrawals, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, reported_event_totals, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, designs, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, eligibilities, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, interventions_studies, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, conditions_studies, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, facilities_studies, parent_key="nct_id", relation_key="nct_id", reduce=True)
    gr.add_entity_edge(studies, sponsors_studies, parent_key="nct_id", relation_key="nct_id", reduce=True)

    gr.add_entity_edge(interventions_studies, interventions, parent_key="intervention_id", relation_key="intervention_id", reduce=True)
    gr.add_entity_edge(conditions_studies, conditions, parent_key="condition_id", relation_key="condition_id", reduce=True)
    gr.add_entity_edge(facilities_studies, facilities, parent_key="facility_id", relation_key="facility_id", reduce=True)
    gr.add_entity_edge(sponsors_studies, sponsors, parent_key="sponsor_id", relation_key="sponsor_id", reduce=True)

    gr.do_transformations_sql()
    df = con.sql(f"select * from {gr.parent_node._cur_data_ref}").to_df().copy()
    label_cols = [c for c in df.columns if c.startswith("oa_") and "label" in c.lower()]
    target = label_cols[0]
    df[target] = df[target].fillna(0).astype("int8")
    return df, target


con = duckdb.connect()
df_val, target = build_frame(con, Path("data/relbench/rel-trial"), VAL_TIMESTAMP)
df_test, target_test = build_frame(con, Path("data/relbench/rel-trial"), TEST_TIMESTAMP)
assert target == target_test

print("val shape:", df_val.shape)
print("test shape:", df_test.shape)
print("target:", target)
```

</details>

### Model Training

<details>
<summary>Show Code</summary>

```python
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

numeric_cols = [c for c in df_val.select_dtypes(include=[np.number]).columns if c != target]
features = [
    c for c in numeric_cols
    if "label" not in c.lower() and not c.lower().endswith("_id") and c != "std_nct_id"
]
features = [c for c in features if c in df_test.columns]

X_train, X_eval, y_train, y_eval = train_test_split(
    df_val[features].fillna(0),
    df_val[target],
    test_size=0.20,
    stratify=df_val[target],
    random_state=42,
)

model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)
model.fit(X_train, y_train)

in_time_auc = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
out_of_time_auc = roc_auc_score(
    df_test[target],
    model.predict_proba(df_test[features].fillna(0))[:, 1],
)

print("in-time AUC (2020-01-01 graph):", f"{in_time_auc:.4f}")
print("out-of-time AUC (2021-01-01 graph):", f"{out_of_time_auc:.4f}")
```

</details>

## Notes

* The graph is explicitly table-level, with one node per table.
* `studies.parquet` is the parent entity table.
* The label is computed from `outcome_analyses.parquet` using primary outcomes and p-value threshold logic.
* Two separate compute graphs are built:
  * validation/train graph at `2020-01-01`
  * holdout graph at `2021-01-01`

## Run Interactive

<div class="modal-runner" data-modal-runner data-api-base="https://runner.13.218.155.128.sslip.io" data-example="relbench_trial_study_outcome">
  <div class="modal-runner-controls">
    <input class="modal-runner-input" data-api-input value="https://runner.13.218.155.128.sslip.io" />
    <button data-save-api-btn>Save API URL</button>
    <button data-run-btn>Run rel-trial Study Outcome</button>
  </div>
  <div class="modal-runner-status" data-status>Idle</div>
  <pre class="modal-runner-log" data-log></pre>
</div>
