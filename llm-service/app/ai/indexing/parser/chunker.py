"""
The script is based on https://github.com/brandonstarxel/chunking_evaluation/
Original code can be found at https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py

This has minor changes to the original code to make it work with the LlamaIndex.
"""

from typing import List, Optional, Any
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TextSplitter,
    MetadataAwareTextSplitter,
    TokenTextSplitter,
    LangchainNodeParser,
)
from pydantic import BaseModel, Field

import numpy as np


class ClusterSemanticChunker(TextSplitter, BaseModel):
    """
    Adapted from the chunking_evaluation repository.

    A semantic chunker that clusters split sentences based on their semantic
    similarity and groups them together as chunks. The chunker uses a similarity
    matrix to calculate the similarity between sentences and then uses a dynamic
    programming algorithm to find the optimal segmentation of the sentences into
    chunks. The chunker uses an embedding function to convert sentences into embeddings.

    Args:
        embedding_function (Any): The embedding function to convert sentences into embeddings.
        max_chunk_size (int): The maximum size of the chunk.
        min_chunk_size (int): The minimum size of the chunk.
    """

    splitter: SentenceSplitter = Field(
        default_factory=lambda: SentenceSplitter(chunk_size=50, chunk_overlap=0)
    )
    _chunk_size: int = 400
    max_cluster: int = Field(default_factory=lambda: 400 // 50)
    embedding_function: Optional[Any] = None

    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50):
        super().__init__()
        self.splitter = SentenceSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
        )
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        self.embedding_function = embedding_function

    def _get_similarity_matrix(self, embedding_function, sentences):
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i : i + BATCH_SIZE]
            embeddings = embedding_function(batch_sentences)

            # Convert embeddings list of lists to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate(
                    (embedding_matrix, batch_embedding_matrix), axis=0
                )

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start : end + 1, start : end + 1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 1 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    # local_density = calculate_local_density(matrix, i, window_size)
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters

    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)

        similarity_matrix = self._get_similarity_matrix(
            self.embedding_function, sentences
        )

        clusters = self._optimal_segmentation(
            similarity_matrix, max_cluster_size=self.max_cluster
        )

        docs = [" ".join(sentences[start : end + 1]) for start, end in clusters]

        return docs
