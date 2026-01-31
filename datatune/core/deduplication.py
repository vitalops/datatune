import glob
import faiss
import ast
from litellm import embedding
import numpy as np
from datatune.logger import get_logger
from typing import List, Callable, Optional
from functools import partial
import pandas as pd
import dask.dataframe as dd
import dask
import os
from .reduce import register_action

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
    df = df.astype(object).where(df.notna(), None)

    df[serialized_input_column] = [
        str(row.to_dict()) for _, row in df.iterrows()
    ]
    return df
    
@register_action("dedup")   
class SemanticDeduplicator():
    def __init__(
    self,
    llm:Callable,
    embedding_model: str = "text-embedding-3-small",
    sim_threshold: float = 0.90,
    top_k: int = 50,
    hnsw_m: int = 32,
    ef_search: int = 64,
    return_df: bool = False,
    
):
        self.embedding_model = embedding_model
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.llm = llm
        self.return_df = return_df


    def _embed_and_write_partition(
        self,
        part,                  # lazy Dask Series
        partition_id: int,
        output_dir: str,
    ):
        pdf = part

        if pdf.empty:
            print("empty partition")
            return 0

        emb_path = f"{output_dir}/embeddings_part_{partition_id}.npy"
        idx_path = f"{output_dir}/index_part_{partition_id}.npy"

        if os.path.exists(emb_path) and os.path.exists(idx_path):
            return 0

        row_index = pdf.index.to_numpy()

        def safe_parse(row):
            try:
                return ast.literal_eval(row)
            except Exception:
                return {}

        dicts = [safe_parse(row) for row in pdf]

        texts = [
            ", ".join(f"{k}: {v}" for k, v in d.items())
            for d in dicts
        ]

        embeddings = []

        for i in range(0, len(texts), 256):
            batch = texts[i:i+256]
            resp = embedding(model=self.embedding_model, input=batch)
            embeddings.extend([item["embedding"] for item in resp["data"]])

        X = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(X)

        np.save(
            f"{output_dir}/embeddings_part_{partition_id}.npy",
            X
        )

        np.save(
            f"{output_dir}/index_part_{partition_id}.npy",
            row_index
        )

        logger.info("Written embeddings for partition %d to disk." % partition_id)

        return len(pdf)


    def embed_column_to_disk(self, df, column, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        tasks = []

        for pid in range(df.npartitions):
            part = df[column].get_partition(pid)

            task = dask.delayed(self._embed_and_write_partition)(
                part,
                pid,
                output_dir,
            )
            tasks.append(task)

        dask.compute(*tasks)



    def build_faiss_index(
        self,
        embedding_dir: str,
        dim: int,
        hnsw_m: int,
        ef_search: int,
    ):
        index = faiss.IndexHNSWFlat(
            dim,
            hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = ef_search

        id_to_row_index = []

        emb_files = sorted(
            glob.glob(os.path.join(embedding_dir, "embeddings_part_*.npy"))
        )
        if not emb_files:
            raise RuntimeError("No embedding files found")

        for emb_path in emb_files:
            pid = emb_path.split("_part_")[-1].split(".")[0]
            idx_path = os.path.join(
                embedding_dir, f"index_part_{pid}.npy"
            )

            X = np.load(emb_path, mmap_mode="r")
            row_idx = np.load(idx_path, mmap_mode="r")

            if X.shape[1] != dim:
                raise ValueError("Embedding dimension mismatch")

            index.add(X)
            id_to_row_index.extend(row_idx.tolist())

        return index, np.asarray(id_to_row_index)

    
    def stream_cluster(
        self,
        index,
        id_to_row_index: np.ndarray,
        embedding_dir: str,
        top_k: int,
        sim_threshold: float,
    ):
        visited = set()
        clusters = []

        emb_files = sorted(
            glob.glob(os.path.join(embedding_dir, "embeddings_part_*.npy"))
        )

        faiss_id_cursor = 0 

        for emb_path in emb_files:
            X = np.load(emb_path, mmap_mode="r")

            D, I = index.search(X, top_k)

            for local_i in range(len(X)):
                anchor_faiss_id = faiss_id_cursor + local_i
                anchor_row = id_to_row_index[anchor_faiss_id]

                if anchor_row in visited:
                    continue

                group = []

                for score, nbr_faiss_id in zip(D[local_i], I[local_i]):
                    if score < sim_threshold:
                        continue

                    nbr_row = id_to_row_index[nbr_faiss_id]

                    if nbr_row == anchor_row:
                        continue

                    if nbr_row not in visited:
                        group.append(nbr_row)
                        visited.add(nbr_row)

                if group:
                    group.append(anchor_row)
                    visited.add(anchor_row)
                    clusters.append(group)

            faiss_id_cursor += len(X)

        return clusters

    def _llm_evaluation(self, clusters, df_column):
        """
        clusters: List[List[int]] of row indices
        df_column: Dask Series (the serialized_input_column)
        llm: your LLM function (unchanged)
        """
        batch_prefix = (
            "You are given multiple independent CLUSTERS of records."
            "For EACH cluster: "
            "- Consider ONLY the records inside that cluster."
            "- Decide whether any records are duplicates."
            "- If duplicates exist, choose ONE canonical record."
            """- If no duplicates exist, output a STRING "NO_DUPLICATES" enclosed in double quotes.""" 
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
            - If no duplicates exist, output exactly a string "NO_DUPLICATES" enclosed in double quotes.

            Do not add explanations.
            Do not reference other clusters.
        """
        from collections import defaultdict
        all_needed_indices = set(idx for cluster in clusters for idx in cluster)
        cluster_rows = defaultdict(list) 
        for pid in range(df_column.npartitions):
            part = df_column.get_partition(pid).compute()
            needed_in_partition = all_needed_indices.intersection(part.index)
            for row_idx in needed_in_partition:
                for cluster_id, cluster in enumerate(clusters):
                    if row_idx in cluster:
                        cluster_rows[cluster_id].append((row_idx, part.loc[row_idx]))
                        break 
        llm_inputs = []
        for cluster_id, cluster in enumerate(clusters):
            lines = [f"CLUSTER {cluster_id} START"]
            row_map = dict(cluster_rows[cluster_id])
            for idx in cluster:
                if idx in row_map:
                    lines.append(f"ID {idx}: {row_map[idx]}")
            lines.append(f"CLUSTER {cluster_id} END")
            llm_inputs.append("\n".join(lines))

        logger.info("Sending %d clusters to LLM for evaluation...", len(llm_inputs))
        llm_outputs = self.llm(
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
            elif isinstance(output, set):
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
    def _dedup_df(self, df, clusters):

        # build drop-set ONCE
        to_drop = set()
        for c in clusters:
            to_drop.update(c["duplicate_ids"])

        def _filter(pdf):
            return pdf.loc[~pdf.index.isin(to_drop)]

        return df.map_partitions(
            _filter,
            meta=df._meta
        )
    def __call__(self, df:dd.DataFrame):
        df = df.reset_index(drop=True)

        df = df.map_partitions(
            partial(
                input_as_string,
                "serialized_input_column",
            )
        )
        #input_rows = df["serialized_input_column"].compute().tolist()
        self.embed_column_to_disk(
            df,
            "serialized_input_column",
            "embeddings_output_dir"
        )
        index, id_to_row_index = self.build_faiss_index(
            embedding_dir="embeddings_output_dir",
            dim=1536,
            hnsw_m=32,
            ef_search=128
        )
        clusters = self.stream_cluster(
        index=index,
        id_to_row_index=id_to_row_index,
        embedding_dir="embeddings_output_dir",
        top_k=20,
        sim_threshold=0.92,
    )
        llm_outputs = self._llm_evaluation(
            clusters, df["serialized_input_column"]
        )

        merges = self._parse_llm_outputs(
            llm_outputs, clusters
        )
        if self.return_df:
            
            deduped_df = self._dedup_df(df, merges)
            deduped_df = deduped_df.drop(columns=["serialized_input_column"])
            return deduped_df
        
        return merges


