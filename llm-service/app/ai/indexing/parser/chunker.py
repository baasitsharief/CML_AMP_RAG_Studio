"""
The script is based on https://github.com/brandonstarxel/chunking_evaluation/
Original code can be found at https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py

This has minor changes to the original code to make it work with the LlamaIndex.
"""

import re
from typing import Callable, List, Optional, Any
from tqdm import tqdm

from pydantic import BaseModel, Field
import numpy as np
from llama_index.core.node_parser import (
    SentenceSplitter,
    TextSplitter,
)
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.utils import get_tokenizer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from ....services.models import LLM


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
    embedding_function: Optional[Any] = Field(
        default_factory=lambda: BedrockEmbedding(
            model_name="cohere.embed-english-v3"
        ).get_text_embedding_batch
    )

    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50):
        super().__init__()
        self.splitter = SentenceSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
        )
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        self.embedding_function = (
            embedding_function
            or BedrockEmbedding(
                model_name="cohere.embed-english-v3"
            ).get_text_embedding_batch
        )

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


class LLMSemanticChunker(TextSplitter, BaseModel):
    """
    LLMSemanticChunker is a class designed to split text into thematically consistent sections based on suggestions from a Language Model (LLM).

    Args:
        splitter (SentenceSplitter, optional): The sentence splitter to split the text into sentences. Defaults to SentenceSplitter(chunk_size=50, chunk_overlap=0).
        llm (LLM, optional): The LLM model to use for the chunking. Defaults to BedrockConverse(model="meta.llama3-1-70b-instruct-v1:0").
        tokenizer (Optional[Callable], optional): The tokenizer to use for tokenizing the text. Defaults to get_tokenizer().

    Attributes:
        splitter (SentenceSplitter): The sentence splitter to split the text into sentences.
        llm (LLM): The LLM model to use for the chunking.
        tokenizer (Optional[Callable]): The tokenizer to use for tokenizing the text.
    """

    splitter: SentenceSplitter = Field(
        default_factory=lambda: SentenceSplitter(chunk_size=50, chunk_overlap=0)
    )
    llm: LLM = Field(
        default_factory=lambda: BedrockConverse(model="meta.llama3-1-70b-instruct-v1:0")
    )
    tokenizer: Optional[Callable] = Field(default_factory=get_tokenizer)

    def __init__(
        self,
        splitter: SentenceSplitter = None,
        llm: LLM = None,
        tokenizer: Optional[Callable] = None,
    ):
        super().__init__()
        self.splitter = splitter or SentenceSplitter(chunk_size=50, chunk_overlap=0)
        self.llm = llm or BedrockConverse(model="meta.llama3-1-70b-instruct-v1:0")
        self.tokenizer = tokenizer or get_tokenizer()

    def get_prompt(
        self, chunked_input, current_chunk=0, invalid_response=None
    ) -> List[ChatMessage]:
        """
        Get the prompt for the user to split the text into chunks.

        Args:
            chunked_input (str): The chunked input text.
            current_chunk (int): The current chunk number.
            invalid_response (List[int]): The invalid response from the user.

        Returns:
            List[ChatMessage]: The list of chat messages.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: "
                    + str(current_chunk)
                    + "."
                    + (
                        f"\n\nThe previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again."
                        if invalid_response
                        else ""
                    )
                ),
            ),
        ]
        return messages

    def split_text(self, text) -> List[str]:

        chunks = self.splitter.split_text(text)

        split_indices = []

        short_cut = len(split_indices) > 0

        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0

                chunked_input = ""

                for i in range(current_chunk, len(chunks)):
                    token_count += len(self.tokenizer(chunks[i]))
                    chunked_input += (
                        f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    )
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.llm.chat(messages=messages).message.content
                    # Use regular expression to find all numbers in the string
                    split_after_line = [
                        line
                        for line in result_string.split("\n")
                        if "split_after:" in line
                    ][0]
                    numbers = re.findall(r"\d+", split_after_line)
                    # Convert the found numbers to integers
                    numbers = list(map(int, numbers))

                    # print(numbers)

                    # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                    if not (
                        numbers != sorted(numbers)
                        or any(number < current_chunk for number in numbers)
                    ):
                        break
                    else:
                        messages = self.get_prompt(
                            chunked_input, current_chunk, numbers
                        )
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)

                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk = ""
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + " "
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs
