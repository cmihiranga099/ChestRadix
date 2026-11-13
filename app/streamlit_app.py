"""Streamlit front end for the triage API.

Run the API first (``python -m uvicorn src.api:app --port 8000``), then::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import base64
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("TRIAGE_API", "http://127.0.0.1:8000")
COLOURS = {"IMMEDIATE": "#c62828", "URGENT": "#ef6c00", "ROUTINE": "#2e7d32"}

st.set_page_config(page_title="Chest X-ray triage", layout="wide")
st.title("Chest X-ray triage")
st.warning("Research prototype for decision support only. Not a medical device and not validated "
           "for clinical use. A qualified clinician must review every case. Do not upload images "
           "that identify a patient.")


@st.cache_data(show_spinner=False)
def findings() -> list[str]:
    return requests.get(f"{API}/findings", timeout=10).json()["findings"]


try:
    names = findings()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API}. Start it with: python -m uvicorn src.api:app --port 8000")
    st.stop()

upload = st.file_uploader("Chest radiograph (PNG or JPEG)", type=["png", "jpg", "jpeg"])
choice = st.selectbox("Explain which finding (heat map)", ["Finding that decided the tier"] + names)
show_anyway = st.checkbox("Show the heat map even when no finding passed the thresholds (weak evidence)")

if upload is not None and st.button("Triage", type="primary"):
    default_finding = choice.startswith("Finding that")
    params = {} if default_finding else {"finding": choice}
    with st.spinner("Running the models..."):
        try:
            reply = requests.post(f"{API}/predict", params=params, timeout=300,
                                  files={"file": (upload.name, upload.getvalue(), upload.type)})
        except requests.RequestException as error:
            st.error(f"Request failed: {error}")
            st.stop()
    if reply.status_code != 200:
        st.error(reply.json().get("detail", reply.text))
        st.stop()
    result = reply.json()
    unc = result["uncertainty"]
    routine = result["tier"] == "ROUTINE"
    colour = COLOURS[result["tier"]]
    st.markdown(f"<div style='padding:14px;border-radius:8px;background:{colour};color:white;"
                f"font-size:28px;font-weight:700'>{result['tier']}</div>", unsafe_allow_html=True)
    if unc["flagged"]:
        st.warning("Low confidence: the models disagree on this image. A human should look at it first, "
                   "whatever the tier says.")
    else:
        st.success("The models agree on this image.")
    left, right = st.columns(2)
    left.image(base64.b64decode(result["images"]["original_png_base64"]), caption="Input",
               use_container_width=True)
    explained = result["explained_finding"]
    p_explained = result["probabilities"][explained]
    if routine and default_finding and not show_anyway:
        right.info("No finding passed the triage thresholds, so no heat map is shown: a heat map only "
                   "means something when the model has found something to point at. Pick a finding in "
                   "the list above to see where the model looked anyway.")
    else:
        cam_caption = (f"Grad-CAM-style heat map (LayerCAM) for {explained} ({p_explained:.1%}): "
                       f"where the model looked, not a diagnosis.")
        if routine and default_finding:
            cam_caption += " No finding passed the thresholds, so this map is weak evidence."
        if p_explained < 0.20:
            cam_caption += " The probability is low, so treat this map as unreliable."
        right.image(base64.b64decode(result["images"]["gradcam_overlay_png_base64"]),
                    caption=cam_caption, use_container_width=True)
    st.subheader("Finding probabilities (calibrated)")
    frame = pd.DataFrame({"finding": list(result["probabilities"]),
                          "probability": list(result["probabilities"].values())})
    chart = alt.Chart(frame).mark_bar().encode(
        x=alt.X("finding:N", sort="-y", title=None),
        y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
        tooltip=["finding", alt.Tooltip("probability:Q", format=".3f")])
    st.altair_chart(chart, use_container_width=True)
    top = result["deciding_finding"]
    if routine:
        st.caption(f"No finding passed the triage thresholds. Highest-scoring finding: "
                   f"{top} ({result['probabilities'][top]:.1%}).")
    else:
        st.caption(f"Tier decided by: {top}.")
    st.caption(f"Disagreement {unc['value']:.4f} (flag cut-off {unc['cutoff']:.4f}, "
               f"about {unc['target_flag_rate']:.0%} of validation cases flagged).")