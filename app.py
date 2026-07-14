import streamlit as st
import requests

API = "http://localhost:8000/chat"

st.set_page_config(
    page_title=" Assessment Recommender",
    layout="wide",
)

st.title("  Assessment Recommendation Agent")

query = st.text_area(
    "Describe your hiring requirements",
    height=180,
)

top_k = st.slider(
    "Top K",
    1,
    10,
    5,
)

if st.button("Recommend"):

    with st.spinner("Searching..."):

        response = requests.post(
            API,
            json={
                "query": query,
                "top_k": top_k,
            },
        )

        data = response.json()

    st.success(
        f"Found {len(data['recommendations'])} recommendations"
    )

    for r in data["recommendations"]:

        with st.container():

            st.subheader(r["assessment_name"])

            st.write(r["description"])

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Rerank Score",
                f"{r['rerank_score']:.3f}",
            )

            c2.metric(
                "Duration",
                r["duration"],
            )

            c3.metric(
                "Remote",
                r["remote"],
            )

            st.write("### Why this recommendation?")

            for reason in r["reason"]:

                st.write("•", reason)

            st.markdown("---")
