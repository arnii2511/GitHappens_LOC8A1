import pandas as pd

from .helpers import safe_float


def verification_checklist(card: dict, buyer_row: pd.Series) -> dict:
    trust = safe_float(card.get("trust_score", 0), 0)
    risk_penalty = safe_float(card.get("risk_penalty", 0), 0)
    channel = str(buyer_row.get("Preferred_Channel", "")).strip().lower()

    must = [
        "Company email + official address proof",
        "Proforma invoice with exact specs (dimensions, material, quantity, Incoterms)",
        "Company bank account details (no personal account)",
    ]

    docs = [
        "GST / business registration proof (if India-based party)",
        "IEC proof (if claiming export history)",
        "Certification proofs (ISO / EU-GMP / SOC2 etc.)",
    ]

    quality = [
        "Request real photos + video with date",
        "Order 1–2 samples before bulk order",
        "Packaging photos + labeling details",
    ]

    payment = [
        "Use escrow / platform payment if possible",
        "30/70 payment split (advance + after inspection)",
        "Third-party inspection before final payment (for large orders)",
    ]

    if risk_penalty >= 10:
        must.append("Add delay penalty clause + shipment timeline in invoice")
        docs.append("Country of origin + Certificate of Origin requirement")
        payment.append("Mandatory third-party inspection before payment release")

    if trust < 45:
        must.append("Verify company on LinkedIn + Google Maps (same name/address)")
        docs.append("Request past shipment proof (BL copy with sensitive info masked)")
        payment.append("Avoid 100% advance transfer")

    if channel == "whatsapp":
        must.append("Ask for email confirmation of all terms (avoid only WhatsApp)")

    return {
        "must_do": must[:6],
        "documents": docs[:6],
        "quality_checks": quality[:5],
        "payment_safety": payment[:5],
    }
