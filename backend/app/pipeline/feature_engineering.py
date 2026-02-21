import pandas as pd

from .helpers import cert_score, minmax


def engineer_buyer_features(buyers: pd.DataFrame) -> pd.DataFrame:
    df = buyers.copy()

    df["cert_score"] = df.get("Certification", "").apply(cert_score)

    rev = pd.to_numeric(df.get("Revenue_Size_USD"), errors="coerce").fillna(0)
    team = pd.to_numeric(df.get("Team_Size"), errors="coerce").fillna(0)
    df["stability_proxy"] = 0.6 * minmax(rev) + 0.4 * minmax(team)

    pay = pd.to_numeric(df.get("Good_Payment_History"), errors="coerce").fillna(0)
    resp = pd.to_numeric(df.get("Prompt_Response"), errors="coerce").fillna(0)
    rp = pd.to_numeric(df.get("Response_Probability"), errors="coerce").fillna(0.35)

    df["buyer_trust"] = 100 * (
        0.35 * pay +
        0.25 * resp +
        0.20 * rp +
        0.10 * df["cert_score"] +
        0.10 * df["stability_proxy"]
    )
    df["buyer_trust"] = df["buyer_trust"].clip(0, 100)

    intent = pd.to_numeric(df.get("Intent_Score"), errors="coerce").fillna(0)
    hiring = pd.to_numeric(df.get("Hiring_Growth"), errors="coerce").fillna(0)
    engage = pd.to_numeric(df.get("Engagement_Spike"), errors="coerce").fillna(0)
    dm_change = pd.to_numeric(df.get("DecisionMaker_Change"), errors="coerce").fillna(0)
    pv = pd.to_numeric(df.get("SalesNav_ProfileVisits"), errors="coerce").fillna(0)
    pv_norm = minmax(pv)

    df["buyer_intent"] = 100 * (
        0.55 * intent +
        0.15 * hiring +
        0.10 * engage +
        0.10 * dm_change +
        0.10 * pv_norm
    )
    df["buyer_intent"] = df["buyer_intent"].clip(0, 100)

    return df


def engineer_exporter_features(exporters: pd.DataFrame) -> pd.DataFrame:
    df = exporters.copy()

    df["cert_score"] = df.get("Certification", "").apply(cert_score)

    rev = pd.to_numeric(df.get("Revenue_Size_USD"), errors="coerce").fillna(0)
    team = pd.to_numeric(df.get("Team_Size"), errors="coerce").fillna(0)
    df["stability_proxy"] = 0.6 * minmax(rev) + 0.4 * minmax(team)

    pay_terms = pd.to_numeric(df.get("Good_Payment_Terms"), errors="coerce").fillna(0)
    resp = pd.to_numeric(df.get("Prompt_Response_Score"), errors="coerce").fillna(0)

    df["exporter_trust"] = 100 * (
        0.35 * pay_terms +
        0.25 * resp +
        0.20 * df["cert_score"] +
        0.20 * df["stability_proxy"]
    )
    df["exporter_trust"] = df["exporter_trust"].clip(0, 100)

    intent = pd.to_numeric(df.get("Intent_Score"), errors="coerce").fillna(0)
    hiring = pd.to_numeric(df.get("Hiring_Signal"), errors="coerce").fillna(0)
    li = pd.to_numeric(df.get("LinkedIn_Activity"), errors="coerce").fillna(0)
    pv = pd.to_numeric(df.get("SalesNav_ProfileViews"), errors="coerce").fillna(0)
    job = pd.to_numeric(df.get("SalesNav_JobChange"), errors="coerce").fillna(0)

    df["exporter_intent"] = 100 * (
        0.55 * intent +
        0.15 * hiring +
        0.10 * minmax(li) +
        0.10 * minmax(pv) +
        0.10 * job
    )
    df["exporter_intent"] = df["exporter_intent"].clip(0, 100)

    return df
