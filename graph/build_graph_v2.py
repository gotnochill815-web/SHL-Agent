import json
import pickle
import re

import networkx as nx

# ------------------------------------------------------
# Dictionaries
# ------------------------------------------------------

SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "sql",
    "mysql",
    "postgres",
    "oracle",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "linux",
    "react",
    "angular",
    "spring",
    "excel",
    "word",
    "power bi",
    "sap",
    "salesforce",
    "networking",
    "hipaa",
]

INDUSTRIES = [
    "finance",
    "banking",
    "healthcare",
    "manufacturing",
    "industrial",
    "sales",
    "retail",
    "customer service",
    "contact center",
    "contact centre",
    "government",
    "education",
]

COMPETENCIES = [
    "leadership",
    "communication",
    "teamwork",
    "problem solving",
    "critical thinking",
    "dependability",
    "safety",
    "customer service",
    "decision making",
    "adaptability",
]

ASSESSMENT_TYPES = [
    "technical",
    "knowledge",
    "simulation",
    "personality",
    "ability",
    "aptitude",
    "behavior",
    "behaviour",
    "coding",
]

# ------------------------------------------------------
# Load Catalog
# ------------------------------------------------------

with open("data/catalog/catalog.json", encoding="utf8") as f:
    catalog = json.load(f)

G = nx.DiGraph()

# ------------------------------------------------------
# Build Graph
# ------------------------------------------------------

for assessment in catalog:

    aid = assessment["assessment_id"]

    G.add_node(
        aid,
        node_type="assessment",
        name=assessment["name"],
    )

    # ---------------- Category ----------------

    for category in assessment.get("category", []):

        node = f"category::{category}"

        G.add_node(node, node_type="category")

        G.add_edge(aid, node)
        G.add_edge(node, aid)

    # ---------------- Job Level ----------------

    for level in assessment.get("job_levels", []):

        node = f"job::{level}"

        G.add_node(node, node_type="job_level")

        G.add_edge(aid, node)
        G.add_edge(node, aid)

    # ---------------- Languages ----------------

    for language in assessment.get("languages", []):

        node = f"language::{language}"

        G.add_node(node, node_type="language")

        G.add_edge(aid, node)
        G.add_edge(node, aid)

    # ---------------- Duration ----------------

    duration = assessment.get("duration_minutes")

    if duration is not None:

        if duration <= 15:
            bucket = "0-15"

        elif duration <= 30:
            bucket = "15-30"

        elif duration <= 45:
            bucket = "30-45"

        else:
            bucket = "45+"

        node = f"duration::{bucket}"

        G.add_node(node, node_type="duration")

        G.add_edge(aid, node)
        G.add_edge(node, aid)

    # --------------------------------------------------
    # Description Mining
    # --------------------------------------------------

    text = (
        assessment["name"]
        + " "
        + assessment["description"]
    ).lower()

    # ---------------- Skills ----------------

    for skill in SKILLS:

        if skill in text:

            node = f"skill::{skill}"

            G.add_node(node, node_type="skill")

            G.add_edge(aid, node)
            G.add_edge(node, aid)

    # ---------------- Industry ----------------

    for industry in INDUSTRIES:

        if industry in text:

            node = f"industry::{industry}"

            G.add_node(node, node_type="industry")

            G.add_edge(aid, node)
            G.add_edge(node, aid)

    # ---------------- Competencies ----------------

    for competency in COMPETENCIES:

        if competency in text:

            node = f"competency::{competency}"

            G.add_node(node, node_type="competency")

            G.add_edge(aid, node)
            G.add_edge(node, aid)

    # ---------------- Assessment Types ----------------

    for assessment_type in ASSESSMENT_TYPES:

        if assessment_type in text:

            node = f"type::{assessment_type}"

            G.add_node(node, node_type="assessment_type")

            G.add_edge(aid, node)
            G.add_edge(node, aid)

# ------------------------------------------------------
# Save
# ------------------------------------------------------

with open(
    "graph/assessment_graph_v2.pkl",
    "wb",
) as f:

    pickle.dump(G, f)

print("=" * 60)

print("Graph V2 Built")

print("Nodes :", G.number_of_nodes())

print("Edges :", G.number_of_edges())