import json
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.fusion import ReciprocalRankFusion
from src.metadata_retriever import MetadataRetriever
from src.query_expander import QueryExpander
from src.recommendation_rules import RecommendationRules      


class HybridRetriever:
    def __init__(
        self,
        catalog_path,
        graph_path,
        embedding_path,
        faiss_path,
        ids_path,
    ):
        # --------------------------------------------------
        # Catalog
        # --------------------------------------------------
        self.catalog = self.load_catalog(catalog_path)
        
        # Filter out Job Solutions
        self.catalog = self.filter_job_solutions(self.catalog)

        self.catalog_lookup = {
            assessment.get("entity_id", assessment.get("assessment_id", "")): assessment
            for assessment in self.catalog
        }

        # --------------------------------------------------
        # Components
        # --------------------------------------------------
        self.metadata = MetadataRetriever(self.catalog)
        self.rules = RecommendationRules()
        self.expander = QueryExpander()
        self.fusion = ReciprocalRankFusion()

        # --------------------------------------------------
        # Graph
        # --------------------------------------------------
        self.graph = self.load_graph(graph_path)

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------
        self.embeddings = np.load(embedding_path)
        self.index = faiss.read_index(faiss_path)

        with open(ids_path) as f:
            self.ids = json.load(f)

        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        self.build_bm25()

    # ======================================================
    # Loading
    # ======================================================
    def load_catalog(self, path):
        with open(path) as f:
            return json.load(f)
    
    def filter_job_solutions(self, catalog):
        filtered = []
        
        for item in catalog:
            name = item.get("name", "").lower()
            description = item.get("description", "").lower()
            keys = " ".join(item.get("keys", [])).lower()
            
            job_keywords = ["job solution", "pre-packaged", "bundle", "package"]
            
            is_job = False
            for kw in job_keywords:
                if kw in name or kw in description or kw in keys:
                    is_job = True
                    break
            
            if not is_job:
                filtered.append(item)
        
        print(f"Filtered out Job Solutions. Kept {len(filtered)} Individual Test Solutions.")
        return filtered

    def load_graph(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ======================================================
    # Get Assessment by Name
    # ======================================================
    def get_assessment_by_name(self, name):
        if not name:
            return None
        
        name_lower = name.lower().strip()
        
        # Handle abbreviations
        abbreviation_map = {
            "opq": "Occupational Personality Questionnaire OPQ32r",
            "opq32r": "Occupational Personality Questionnaire OPQ32r",
            "gsa": "Global Skills Assessment",
            "mq": "Motivation Questionnaire MQM5",
            "dsi": "Dependability and Safety Instrument (DSI)",
        }
        
        # Try abbreviation match first
        if name_lower in abbreviation_map:
            full_name = abbreviation_map[name_lower]
            for assessment in self.catalog:
                assessment_name = assessment.get("name", "").lower()
                if full_name.lower() in assessment_name or assessment_name in full_name.lower():
                    return assessment
        
        # Try partial match
        for assessment in self.catalog:
            assessment_name = assessment.get("name", "").lower()
            
            if name_lower == assessment_name:
                return assessment
            if name_lower in assessment_name:
                return assessment
            if assessment_name in name_lower:
                return assessment
        
        return None

    # ======================================================
    # BM25
    # ======================================================
    def tokenize(self, text):
        return text.lower().split()

    def build_bm25(self):
        corpus = []

        for assessment in self.catalog:
            text = f"""
            {assessment.get("name","")}
            {' '.join(assessment.get("job_levels", []))}
            {assessment.get("description","")}
            """

            corpus.append(
                self.tokenize(text)
            )

        self.bm25 = BM25Okapi(corpus)

    # ======================================================
    # Semantic Search
    # ======================================================
    def semantic_search(
        self,
        query,
        top_k=20,
    ):
        if not query or not query.strip():
            return []
        
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, ids = self.index.search(
            embedding,
            min(top_k, len(self.catalog)),
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.catalog):
                continue

            results.append(
                {
                    "assessment": self.catalog[idx],
                    "score": float(score),
                    "source": "semantic",
                }
            )

        return results

    # ======================================================
    # BM25 Search
    # ======================================================
    def keyword_search(
        self,
        query,
        top_k=20,
    ):
        if not query or not query.strip():
            return []
        
        scores = self.bm25.get_scores(
            self.tokenize(query)
        )

        order = np.argsort(scores)[::-1]

        results = []

        for idx in order[:min(top_k, len(self.catalog))]:
            results.append(
                {
                    "assessment": self.catalog[idx],
                    "score": float(scores[idx]),
                    "source": "bm25",
                }
            )

        return results

    # ======================================================
    # Graph Expansion
    # ======================================================
    def graph_expand(
        self,
        assessment_id,
    ):
        if assessment_id not in self.graph:
            return []

        expanded = set()

        try:
            for metadata_node in self.graph.neighbors(assessment_id):
                for node in self.graph.neighbors(metadata_node):
                    if node == assessment_id:
                        continue
                    if node not in self.catalog_lookup:
                        continue
                    expanded.add(node)
        except:
            pass

        return list(expanded)

    # ======================================================
    # Retrieval
    # ======================================================
    def retrieve(
        self,
        intent,
        top_k=30,
    ):
        # --------------------------------------------------
        # Query Expansion
        # --------------------------------------------------
        query = self.expander.expand(intent)

        if not query or not query.strip():
            query = "assessment"

        candidate_size = max(
            top_k * 10,
            100,
        )
        candidate_size = min(candidate_size, len(self.catalog))

        # --------------------------------------------------
        # Hybrid Retrieval
        # --------------------------------------------------
        semantic = self.semantic_search(
            query,
            candidate_size,
        )

        keyword = self.keyword_search(
            query,
            candidate_size,
        )

        metadata = self.metadata.retrieve(
            intent,
            candidate_size,
        )

        # --------------------------------------------------
        # Metadata Boost
        # --------------------------------------------------
        for item in metadata:
            assessment = item["assessment"]

            text = (
                assessment.get("name", "")
                + " "
                + assessment.get("description", "")
            ).lower()

            matched = []
            boost = 0.0

            for skill in intent.get("skills", []):
                if skill.lower() in text:
                    matched.append(skill)
                    boost += 1.5

            keys = " ".join(assessment.get("keys", [])).lower()

            for domain in intent.get("domains", []):
                if domain.lower() in keys:
                    boost += 0.5

            levels = " ".join(assessment.get("job_levels", [])).lower()

            if (
                intent.get("job_level")
                and intent["job_level"].lower() in levels
            ):
                boost += 0.5

            assessment["matched_skills"] = matched
            item["score"] += boost

        # --------------------------------------------------
        # Graph Expansion
        # --------------------------------------------------
        graph_results = []
        visited = set()

        for result in semantic + keyword + metadata:
            aid = result["assessment"].get("entity_id", result["assessment"].get("assessment_id", ""))
            if not aid:
                continue

            for neighbor in self.graph_expand(aid):
                if neighbor in visited:
                    continue

                assessment = self.catalog_lookup.get(neighbor)
                if assessment is None:
                    continue

                visited.add(neighbor)

                graph_results.append(
                    {
                        "assessment": assessment,
                        "score": 2.0,
                        "source": "graph",
                    }
                )

        # --------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------
        results = self.fusion.fuse(
            [
                semantic,
                keyword,
                metadata,
                graph_results,
            ]
        )

        # --------------------------------------------------
        # Skill-aware Retrieval Boost
        # --------------------------------------------------
        for result in results:
            assessment = result["assessment"]

            text = (
                assessment.get("name", "")
                + " "
                + assessment.get("description", "")
            ).lower()

            retrieval_score = result.get("score", 0.0)

            matched_skills = []

            for skill in intent.get("skills", []):
                if skill.lower() in text:
                    retrieval_score += 2.0
                    matched_skills.append(skill)

            assessment["matched_skills"] = matched_skills

            keys = " ".join(assessment.get("keys", [])).lower()

            for domain in intent.get("domains", []):
                if domain.lower() in keys:
                    retrieval_score += 0.5

            levels = " ".join(assessment.get("job_levels", [])).lower()

            if (
                intent.get("job_level")
                and intent["job_level"].lower() in levels
            ):
                retrieval_score += 0.5

            if (
                intent.get("adaptive")
                and assessment.get("adaptive") == "yes"
            ):
                retrieval_score += 0.3

            if (
                intent.get("remote")
                and assessment.get("remote") == "yes"
            ):
                retrieval_score += 0.3

            result["retrieval_score"] = retrieval_score

        # --------------------------------------------------
        # Sort
        # --------------------------------------------------
        results.sort(
            key=lambda x: x["retrieval_score"],
            reverse=True,
        )

        # --------------------------------------------------
        # Duration Filter
        # --------------------------------------------------
        filtered = []

        for result in results:
            assessment = result["assessment"]
            duration_str = assessment.get("duration", "")

            duration_minutes = None
            if duration_str:
                import re
                minutes_match = re.search(r"(\d+)\s*(?:minutes?|mins?|min)", duration_str)
                if minutes_match:
                    duration_minutes = int(minutes_match.group(1))

            if (
                intent.get("duration") is not None
                and duration_minutes is not None
                and duration_minutes > intent["duration"]
            ):
                continue

            filtered.append(result)

        # --------------------------------------------------
        # Rule-Based Boosting
        # --------------------------------------------------
        filtered = self.rules.apply(
            intent,
            filtered,
        )

        # --------------------------------------------------
        # Return top_k
        # --------------------------------------------------
        return filtered[:top_k]
