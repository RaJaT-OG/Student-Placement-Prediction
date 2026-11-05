"""Placement Studio: data inspection, preparation, training, and prediction UI."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR, OUTPUT_DIR, MODEL_DIR = ROOT / "data", ROOT / "outputs", ROOT / "models"
for folder in (OUTPUT_DIR, MODEL_DIR):
    folder.mkdir(exist_ok=True)

st.set_page_config(page_title="Placement Studio", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#18231f;--muted:#718078;--line:#e6ebe7;--green:#147d5a;--pale:#edf6f0;--paper:#f7f9f7}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:var(--paper)} .block-container{padding-top:2rem;max-width:1420px}
h1,h2,h3{font-family:Manrope,sans-serif!important;letter-spacing:-.04em}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.eyebrow{font:500 11px 'DM Mono';letter-spacing:.14em;text-transform:uppercase;color:var(--green)}
.hero{background:#fff;border:1px solid var(--line);border-radius:18px;padding:28px 32px;margin:10px 0 22px}
.metric{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;min-height:112px}
.metric-label{font:500 11px 'DM Mono';color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.metric-value{font:800 28px Manrope;color:var(--ink);margin-top:9px;letter-spacing:-.05em}
.metric-note{color:var(--muted);font-size:12px;margin-top:2px}
.step{display:flex;align-items:center;gap:11px;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:#fff;margin-bottom:8px}
.step-num{font:500 11px 'DM Mono';color:var(--green);background:var(--pale);border-radius:8px;padding:7px 9px}
.step-title{font-weight:600;font-size:13px}.step-sub{font-size:11px;color:var(--muted)}
div[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:16px;border-radius:12px}
.stButton>button[kind="primary"]{background:#147d5a;border-color:#147d5a}
</style>
""", unsafe_allow_html=True)


def metric(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def load_csv(upload, selected):
    if upload is not None:
        return pd.read_csv(upload), upload.name
    if selected and selected != "Upload a dataset":
        return pd.read_csv(DATA_DIR / selected), selected
    return None, None


def make_pipeline(X, kind, params):
    nums = X.select_dtypes(include="number").columns.tolist()
    cats = X.select_dtypes(exclude="number").columns.tolist()
    transforms = []
    if nums:
        transforms.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums))
    if cats:
        transforms.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), cats))
    model = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=params["C"], random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=params["trees"], min_samples_leaf=params["leaf"], class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=params["trees"], learning_rate=params["rate"], max_depth=params["depth"], random_state=42),
    }[kind]
    return Pipeline([("preprocessor", ColumnTransformer(transforms)), ("model", model)])


if "data" not in st.session_state:
    st.session_state.data = None
if "model" not in st.session_state:
    st.session_state.model = None

with st.sidebar:
    st.markdown("<div class='eyebrow'>PLACEMENT INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("## ◈ Placement Studio")
    st.caption("A complete workflow from raw data to a validated model.")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Data workspace", "Prepare data", "Train & evaluate", "Predict & export"], label_visibility="visible")
    st.divider()
    st.markdown("<div class='eyebrow'>WORKFLOW</div>", unsafe_allow_html=True)
    for i, (name, sub) in enumerate([("Load data", "CSV or project dataset"), ("Inspect & clean", "Quality and target checks"), ("Train models", "Compare validation metrics"), ("Export", "Model and predictions")], 1):
        st.markdown(f"<div class='step'><div class='step-num'>0{i}</div><div><div class='step-title'>{name}</div><div class='step-sub'>{sub}</div></div></div>", unsafe_allow_html=True)
    st.caption("Local-first · Reproducible · No external API")

st.markdown("<div class='eyebrow'>STUDENT OUTCOMES / WORKSPACE</div>", unsafe_allow_html=True)
st.title(page)

if page == "Overview":
    files = sorted(p.name for p in DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    df = st.session_state.data
    if df is None and files:
        try:
            df = pd.read_csv(DATA_DIR / ("student_placement_train.csv" if "student_placement_train.csv" in files else files[0]))
            st.session_state.data = df
            st.session_state.filename = "student_placement_train.csv" if "student_placement_train.csv" in files else files[0]
        except Exception as exc:
            st.error(f"Could not load the default dataset: {exc}")
    rows = len(df) if df is not None else 0
    cols = len(df.columns) if df is not None else 0
    missing = int(df.isna().sum().sum()) if df is not None else 0
    target_candidates = [c for c in (df.columns if df is not None else []) if c.lower() in ("placed", "placementstatus", "placement_status", "placement")]
    placed_pct = "—"
    if df is not None and target_candidates:
        y = df[target_candidates[0]].astype(str).str.lower().str.strip()
        placed_pct = f"{(y.isin(['1','1.0','placed','yes','true']).mean()*100):.1f}%"
    st.markdown("<div class='hero'><div class='eyebrow'>END-TO-END ML WORKFLOW</div><h2 style='margin:8px 0 4px'>Turn student data into placement insights.</h2><div style='color:#718078'>Inspect and prepare your data, train candidate models, then export a reusable pipeline and predictions.</div></div>", unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    with a: metric("Dataset rows", f"{rows:,}", "currently loaded")
    with b: metric("Features", f"{cols:,}", "columns available")
    with c: metric("Missing cells", f"{missing:,}", "review before training")
    with d: metric("Placed rate", placed_pct, "detected target share")
    st.markdown("### Your workflow")
    st.write("Move through each stage in order. The training and prediction pages use the same fitted preprocessing pipeline, so missing values and new categories are handled consistently.")
    cols=st.columns(4)
    for col,(n,title,desc) in zip(cols,[('01','Data workspace','Load CSV, inspect schema, quality and target balance.'),('02','Prepare data','Choose target, remove duplicates and exclude identifiers.'),('03','Train & evaluate','Compare models on a stratified holdout and save the best pipeline.'),('04','Predict & export','Score new rows and download a submission-ready CSV.')]):
        with col: st.markdown(f"<div class='metric'><div class='metric-label'>{n} / WORKFLOW</div><div style='font:700 16px Manrope;margin-top:12px'>{title}</div><div class='metric-note' style='margin-top:8px'>{desc}</div></div>",unsafe_allow_html=True)
    if files:
        st.markdown("### Project datasets")
        st.dataframe(pd.DataFrame([{"Dataset":f,"Size":f"{(DATA_DIR/f).stat().st_size/1024:.0f} KB"} for f in files]),hide_index=True,use_container_width=True)

elif page == "Data workspace":
    local_files = sorted(p.name for p in DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    left,right=st.columns([1,1])
    with left:
        chosen=st.selectbox("Choose a project CSV",["Upload a dataset"]+local_files)
    with right:
        uploaded=st.file_uploader("Or upload a CSV",type=["csv"])
    if st.button("Load dataset",type="primary"):
        try:
            data,name=load_csv(uploaded,chosen)
            if data is None: st.warning("Choose a project file or upload a CSV.")
            elif data.empty: st.error("This CSV has no data rows.")
            else:
                st.session_state.data=data
                st.session_state.filename=name
                st.success(f"Loaded {name}: {len(data):,} rows × {len(data.columns)} columns.")
        except Exception as exc: st.error(f"Unable to read CSV: {exc}")
    df=st.session_state.data
    if df is not None:
        st.markdown(f"### {st.session_state.get('filename','Dataset')}")
        a,b,c,d=st.columns(4)
        with a: metric("Rows",f"{len(df):,}")
        with b: metric("Columns",str(len(df.columns)))
        with c: metric("Duplicates",f"{int(df.duplicated().sum()):,}")
        with d: metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
        t1,t2,t3=st.tabs(["Preview","Column profile","Target balance"])
        with t1: st.dataframe(df.head(100),use_container_width=True,hide_index=True)
        with t2:
            profile=pd.DataFrame({"Column":df.columns,"Type":[str(x) for x in df.dtypes],"Missing":df.isna().sum().values,"Missing %":(df.isna().mean()*100).round(1).values,"Unique":df.nunique(dropna=True).values})
            st.dataframe(profile,use_container_width=True,hide_index=True)
        with t3:
            guess=next((x for x in df.columns if x.lower() in ("placed","placementstatus","placement_status","placement")),None)
            target=st.selectbox("Target column",["(not set)"]+list(df.columns),index=(list(df.columns).index(guess)+1 if guess else 0))
            if target!="(not set)":
                counts=df[target].fillna("Missing").astype(str).value_counts().rename_axis("Class").reset_index(name="Rows")
                st.bar_chart(counts.set_index("Class")); st.dataframe(counts,hide_index=True)

elif page == "Prepare data":
    df=st.session_state.data
    if df is None: st.info("Load a dataset in Data workspace to begin.")
    else:
        st.markdown("Choose the label and exclude identifier or leakage columns. Imputation, encoding and scaling are fitted inside each training split to avoid validation leakage.")
        likely=next((x for x in df.columns if x.lower() in ("placed","placementstatus","placement_status","placement")),df.columns[-1])
        target=st.selectbox("Target column",list(df.columns),index=list(df.columns).index(likely))
        candidates=[c for c in df.columns if c!=target]
        default_ids=[c for c in candidates if c.lower() in ("student_id","id","record_id") or c.lower().endswith("_id")]
        exclude=st.multiselect("Identifier / excluded columns",candidates,default=default_ids)
        split=st.slider("Validation share",0.1,0.4,0.2,0.05)
        if st.button("Apply preparation",type="primary"):
            clean=df.drop_duplicates().dropna(subset=[target]).copy()
            labels=clean[target]
            if labels.nunique()!=2: st.error("This workflow currently expects a binary target. Select a target with exactly two classes.")
            elif len(clean)<10: st.error("At least 10 labeled rows are needed for a stratified train/validation split.")
            else:
                X=clean.drop(columns=[target]+exclude)
                X=X.dropna(axis=1,how="all")
                # Normalize common placement labels while preserving the positive class.
                vals=labels.astype(str).str.strip().str.lower()
                positive=next((v for v in vals.unique() if v in ("1","1.0","placed","yes","true")),sorted(vals.unique())[-1])
                y=(vals==positive).astype(int)
                if y.value_counts().min()<2: st.error("Each target class needs at least two labeled examples.")
                else:
                    strat=y if y.value_counts().min()>=2 else None
                    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=split,random_state=42,stratify=strat)
                    st.session_state.prepared={"X":X,"y":y,"X_train":Xtr,"X_valid":Xv,"y_train":ytr,"y_valid":yv,"target":target,"exclude":exclude,"positive":positive,"clean":clean}
                    st.success(f"Ready: {len(Xtr):,} training rows, {len(Xv):,} validation rows, {X.shape[1]} feature columns. Positive class: {positive}.")
        prep=st.session_state.get("prepared")
        if prep:
            a,b,c=st.columns(3)
            with a: metric("Training rows",f"{len(prep['X_train']):,}")
            with b: metric("Validation rows",f"{len(prep['X_valid']):,}")
            with c: metric("Feature columns",str(prep['X'].shape[1]))

elif page == "Train & evaluate":
    prep=st.session_state.get("prepared")
    if not prep: st.info("Complete the Prepare data step first.")
    else:
        st.markdown("Models are compared on the same stratified holdout. The highest ROC-AUC model is saved with its full preprocessing pipeline.")
        algos=st.multiselect("Candidate models",["Logistic Regression","Random Forest","Gradient Boosting"],default=["Logistic Regression","Random Forest","Gradient Boosting"])
        c1,c2,c3=st.columns(3)
        with c1: cval=st.slider("Logistic C",0.05,5.0,1.0,0.05)
        with c2: trees=st.slider("Ensemble trees",50,500,200,50)
        with c3: depth=st.slider("Boosting depth",1,6,3)
        rate=st.slider("Boosting learning rate",0.01,0.3,0.05,0.01)
        if st.button("Train & compare",type="primary",disabled=not algos):
            metrics=[]; best_auc=-1; best=None; bar=st.progress(0)
            try:
                for i,name in enumerate(algos):
                    pipe=make_pipeline(prep["X_train"],name,{"C":cval,"trees":trees,"leaf":2,"depth":depth,"rate":rate})
                    pipe.fit(prep["X_train"],prep["y_train"])
                    pred=pipe.predict(prep["X_valid"]); prob=pipe.predict_proba(prep["X_valid"])[:,1]
                    row={"Model":name,"Accuracy":accuracy_score(prep["y_valid"],pred),"Precision":precision_score(prep["y_valid"],pred,zero_division=0),"Recall":recall_score(prep["y_valid"],pred,zero_division=0),"F1":f1_score(prep["y_valid"],pred,zero_division=0),"ROC-AUC":roc_auc_score(prep["y_valid"],prob)}
                    metrics.append(row)
                    if row["ROC-AUC"]>best_auc: best_auc=row["ROC-AUC"];best=(name,pipe,pred,prob)
                    bar.progress((i+1)/len(algos),text=f"Fitting {name}…")
                results=pd.DataFrame(metrics).sort_values("ROC-AUC",ascending=False).reset_index(drop=True)
                st.session_state.model={"pipeline":best[1],"name":best[0],"results":results,"metrics":best[2:],"prep":prep}
                results.to_csv(OUTPUT_DIR/"model_comparison.csv",index=False)
                joblib.dump({"pipeline":best[1],"target":prep["target"],"exclude":prep["exclude"],"positive":prep["positive"],"features":prep["X"].columns.tolist()},MODEL_DIR/"best_placement_pipeline.joblib")
                bar.empty();st.success(f"Training complete. Best model: {best[0]} (ROC-AUC {best_auc:.3f}).")
            except Exception as exc: st.error(f"Training failed: {exc}")
        trained=st.session_state.model
        if trained:
            res=trained["results"]
            st.markdown(f"### Validation results · best: {trained['name']}")
            st.dataframe(res.style.format({c:"{:.3f}" for c in res.columns if c!="Model"}),use_container_width=True,hide_index=True)
            bestrow=res.iloc[0]
            cols=st.columns(4)
            for col,(k,label) in zip(cols,[("ROC-AUC","ROC-AUC"),("F1","F1 score"),("Recall","Recall"),("Accuracy","Accuracy")]):
                with col: metric(label,f"{bestrow[k]:.1%}",trained["name"])
            pred,prob=trained["metrics"]
            cm=confusion_matrix(prep["y_valid"],pred,labels=[0,1])
            fig,ax=plt.subplots(figsize=(5,3));ax.imshow(cm,cmap="Greens");ax.set_xticks([0,1],labels=["Not placed","Placed"]);ax.set_yticks([0,1],labels=["Not placed","Placed"]);ax.set_xlabel("Predicted");ax.set_ylabel("Actual")
            for (i,j),v in __import__('numpy').ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center",color="#18231f",fontweight="bold")
            fig.tight_layout();st.pyplot(fig);plt.close(fig)
            st.download_button("Download comparison CSV",res.to_csv(index=False).encode(),"model_comparison.csv","text/csv")

elif page == "Predict & export":
    saved=MODEL_DIR/"best_placement_pipeline.joblib"
    trained=st.session_state.model
    if trained:
        artifact=joblib.load(saved)
    elif saved.exists():
        try:
            artifact=joblib.load(saved)
            if not isinstance(artifact, dict) or "pipeline" not in artifact:
                artifact=None
                st.warning("The existing model was saved in the older format. Train a model from this workspace to create a compatible scoring artifact.")
        except Exception as exc: artifact=None;st.error(f"Could not load model: {exc}")
    else: artifact=None
    if artifact is None: st.info("Train a model first to enable scoring and export.")
    else:
        pipe=artifact["pipeline"]
        st.success(f"Active model: {trained['name'] if trained else 'Saved placement pipeline'}")
        st.markdown("Upload a scoring CSV. Identifier columns are retained in the output but excluded from model features.")
        uploaded=st.file_uploader("Scoring dataset",type=["csv"],key="score_file")
        fallback=st.session_state.data
        score_df=None
        if uploaded:
            try: score_df=pd.read_csv(uploaded)
            except Exception as exc: st.error(f"Could not read upload: {exc}")
        if score_df is not None:
            st.write(f"{len(score_df):,} rows · {len(score_df.columns)} columns")
            st.dataframe(score_df.head(8),use_container_width=True,hide_index=True)
            if st.button("Generate predictions",type="primary"):
                try:
                    target=artifact["target"]; excludes=artifact["exclude"]
                    Xscore=score_df.drop(columns=[target]+excludes,errors="ignore")
                    Xscore=Xscore.reindex(columns=artifact["features"])
                    probabilities=pipe.predict_proba(Xscore)[:,1]; labels=(probabilities>=0.5).astype(int)
                    positive=artifact["positive"]
                    negative=next((str(x) for x in st.session_state.prepared["clean"][target].dropna().unique() if str(x).strip().lower()!=str(positive).lower()),"NotPlaced") if st.session_state.get("prepared") else "NotPlaced"
                    out=score_df.copy();out["placement_probability"]=probabilities;out["placement_prediction"]=pd.Series(labels,index=out.index).map({1:positive,0:negative})
                    st.session_state.predictions=out
                    st.success("Predictions generated.")
                except Exception as exc: st.error(f"Prediction failed. Confirm the scoring CSV has the model's feature columns. Details: {exc}")
        output=st.session_state.get("predictions")
        if output is not None:
            st.dataframe(output.head(100),use_container_width=True,hide_index=True)
            st.download_button("Download predictions CSV",output.to_csv(index=False).encode(),"placement_predictions.csv","text/csv")
