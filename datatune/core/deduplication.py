import faiss
import ast
from litellm import embedding
import numpy as np
from datatune.logger import get_logger
from typing import List, Callable, Optional
from functools import partial
import pandas as pd
import dask.dataframe as dd

logger = get_logger(__name__)
def input_as_string(
    serialized_input_column: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts each row in the DataFrame to a string representation and stores it in a new column.

    Args:
        serialized_input_column (str): Name of the column to store the serialized row data.
        df (pd.DataFrame): Input DataFrame to process.
        input_fields (Optional[List], optional): List of input fields to include. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the added serialized input column.
    """
    df[serialized_input_column] = [
        str(row.to_dict()) for _, row in df.iterrows()
    ]
    return df
class SemanticDeduplicator:
    
    def __init__(
    self,
    embedding_model: str = "text-embedding-3-small",
    sim_threshold: float = 0.90,
    top_k: int = 50,
    hnsw_m: int = 32,
    ef_search: int = 64,
):
        self.embedding_model = embedding_model
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search

    def _embed_and_cluster(self, input_rows: List[str]):
        dicts = [ast.literal_eval(row) for row in input_rows]

        def dict_to_text(d):
            return ", ".join(f"{k}: {v}" for k, v in d.items())

        texts = [dict_to_text(d) for d in dicts]

        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        logger.info("Obtained embeddings for %d records", len(texts))
        embeddings = [item["embedding"] for item in response["data"]]
        X = np.array(embeddings).astype("float32")
        faiss.normalize_L2(X)

        dim = X.shape[1]
        index = faiss.IndexHNSWFlat(
            dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
        )
        index.add(X)
        index.hnsw.efSearch = self.ef_search

        clusters = []
        visited = set()

        for i in range(len(X)):
            if i in visited:
                continue

            D, I = index.search(X[i:i+1], self.top_k)

            group = []
            for score, j in zip(D[0], I[0]):
                if score >= self.sim_threshold:
                    group.append(j)
                    visited.add(j)

            clusters.append(group)
        cluster_candidates = [c for c in clusters if len(c) > 1]
        print(cluster_candidates)
        return cluster_candidates
    
    def _llm_evaluation(self, clusters, input_rows, llm):
        batch_prefix = (
            "You are given multiple independent CLUSTERS of records."
            "For EACH cluster: "
            "- Consider ONLY the records inside that cluster."
            "- Decide whether any records are duplicates."
            "- If duplicates exist, choose ONE canonical record."
            """- If no duplicates exist, say "NO_DUPLICATES". must be enclosed in double quotes"""
            "Output exactly ONE JSON object PER CLUSTER."
            "Do not reference other clusters."
        )

        batch_suffix = """For EACH cluster above:
- Produce exactly ONE output.
- Output MUST correspond to the cluster in the same order.
- If duplicates exist, output valid JSON:
{
"canonical_id": <int>,
"duplicate_ids": [<int>, ...]
}
- If no duplicates exist, output exactly:
"NO_DUPLICATES"

Do not add explanations.
Do not reference other clusters.
"""

        def cluster_to_string(cluster_id, cluster):
            lines = [f"CLUSTER {cluster_id} START"]
            for idx in cluster:
                lines.append(f"ID {idx}: {input_rows[idx]}")
            lines.append(f"CLUSTER {cluster_id} END")
            return "\n".join(lines)

        llm_inputs = [
            cluster_to_string(i, cluster)
            for i, cluster in enumerate(clusters)
        ]
        logger.info("Sending %d clusters to LLM for evaluation...", len(llm_inputs))
        llm_outputs = llm(
            llm_inputs,
            batch_prefix=batch_prefix,
            prompt_per_row="",
            batch_suffix=batch_suffix,
            max_retries=3,
            optimized=True
        )
        logger.info("LLM evaluation within clusters completed.")
        return llm_outputs
    def _parse_llm_outputs(self, llm_outputs, candidate_clusters):
        merges = []

        for i, output in enumerate(llm_outputs):
            output = output.strip()

            if output == "NO_DUPLICATES":
                continue

            try:
                data = ast.literal_eval(output)
            except (ValueError, SyntaxError):
                logger.warning("Invalid JSON for cluster %d: %s", i, output)
                continue

            canonical_id = data.get("canonical_id")
            duplicate_ids = data.get("duplicate_ids")

            if not isinstance(canonical_id, int):
                continue

            if not isinstance(duplicate_ids, list) or \
            not all(isinstance(x, int) for x in duplicate_ids):
                continue

            cluster_set = set(candidate_clusters[i])
            if canonical_id not in cluster_set or \
            not set(duplicate_ids).issubset(cluster_set):
                continue

            merges.append({
                "canonical_id": canonical_id,
                "duplicate_ids": duplicate_ids
            })

        return merges
    
    def __call__(self, df:dd.DataFrame, llm: Callable):
        df = df.map_partitions(
            partial(
                input_as_string,
                "serialized_input_column",
            )
        )
        input_rows = df["serialized_input_column"].compute().tolist()
        candidate_clusters = self._embed_and_cluster(input_rows)
        if not candidate_clusters:
            return []

        llm_outputs = self._llm_evaluation(
            candidate_clusters, input_rows, llm
        )

        merges = self._parse_llm_outputs(
            llm_outputs, candidate_clusters
        )

        return merges


