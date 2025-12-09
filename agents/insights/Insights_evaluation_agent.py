#!/usr/bin/env python3
import asyncio
import uuid

import anthropic
import json
import logging
import os
import re
import csv
import io
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import click
import mcp.types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from qdrant_client import QdrantClient, models
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================== Insight Evaluation (Updated) =========================================
@dataclass
class Claim:
    text: str
    truth_value: Optional[float] = None
    reasoning: str = ""


@dataclass
class InsightfulnessScores:
    actionability: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    novelty: float = 0.0

    def weighted_average(self, weights: Optional[Dict[str, float]] = None) -> float:
        if weights is None:
            weights = {'actionability': 1.0, 'relevance': 1.0, 'clarity': 1.0, 'novelty': 1.0}
        total_weight = sum(weights.values())
        weighted_sum = sum(getattr(self, dim) * weights[dim] for dim in weights)
        return weighted_sum / total_weight


@dataclass
class EvaluationResult:
    insight: str
    claims: List[Claim]
    correctness_score: Optional[float]
    insightfulness_scores: InsightfulnessScores
    combined_score: Optional[float]
    alpha: float = 0.5
    llm_explanation: str = ""
    data_summary: str = ""
    column_matches: Dict[str, str] = None
    cached: bool = False  # NEW: Track if result was from cache

    def to_dict(self) -> dict:
        return {
            'insight': self.insight,
            'claims': [asdict(c) for c in self.claims],
            'correctness_score': self.correctness_score,
            'insightfulness_scores': asdict(self.insightfulness_scores),
            'combined_score': self.combined_score,
            'alpha': self.alpha,
            'llm_explanation': self.llm_explanation,
            'data_summary': self.data_summary,
            'column_matches': self.column_matches,
            'cached': self.cached
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationResult':
        """Reconstruct EvaluationResult from dictionary"""
        claims = [Claim(**c) for c in data.get('claims', [])]
        insightfulness_data = data.get('insightfulness_scores', {})
        insightfulness_scores = InsightfulnessScores(**insightfulness_data)

        return cls(
            insight=data.get('insight', ''),
            claims=claims,
            correctness_score=data.get('correctness_score'),
            insightfulness_scores=insightfulness_scores,
            combined_score=data.get('combined_score'),
            alpha=data.get('alpha', 0.5),
            llm_explanation=data.get('llm_explanation', ''),
            data_summary=data.get('data_summary', ''),
            column_matches=data.get('column_matches', {}),
            cached=data.get('cached', False)
        )


@dataclass
class BatchEvaluationResult:
    results: List[EvaluationResult]
    data_summary: str
    total_insights: int
    successful_evaluations: int
    failed_evaluations: int
    average_combined_score: Optional[float]
    cache_hits: int = 0  # NEW: Track cache hits

    def to_dict(self) -> dict:
        return {
            'results': [r.to_dict() for r in self.results],
            'data_summary': self.data_summary,
            'total_insights': self.total_insights,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'average_combined_score': self.average_combined_score,
            'cache_hits': self.cache_hits,
            'summary': self._generate_summary()
        }

    def _generate_summary(self) -> str:
        lines = [
            f"Evaluated {self.total_insights} insights",
            f"Successful: {self.successful_evaluations}, Failed: {self.failed_evaluations}",
            f"Cache hits: {self.cache_hits}"
        ]
        if self.average_combined_score is not None:
            lines.append(f"Average combined score: {self.average_combined_score:.3f}")

        # Sort by combined score
        scored_results = [r for r in self.results if r.combined_score is not None]
        if scored_results:
            scored_results.sort(key=lambda x: x.combined_score, reverse=True)
            lines.append("\nTop insights:")
            for i, result in enumerate(scored_results[:3], 1):
                cache_indicator = " (cached)" if result.cached else ""
                lines.append(f"  {i}. Score {result.combined_score:.3f}: {result.insight[:80]}...{cache_indicator}")

        return "\n".join(lines)


class InsightEvaluatorServer:
    def __init__(self, anthropic_api_key: Optional[str] = None, model: str = "claude-3-5-haiku-latest",
                 use_qdrant_memory: bool = False, cache_similarity_threshold: float = 0.95):
        # Anthropic setup
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.model = model
        logger.info(f"Claude client initialized with model: {model}")

        # Qdrant setup
        try:
            self.qdrant_client = QdrantClient(":memory:" if use_qdrant_memory else "http://localhost:6333")
            logger.info(f"Qdrant client initialized {'in-memory' if use_qdrant_memory else 'at localhost:6333'}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        self.qdrant_client.set_model("BAAI/bge-small-en-v1.5")
        self.column_collection = "table_columns"
        self.insights_cache_collection = "evaluated_insights_cache"  # Cache collection
        self.cache_similarity_threshold = cache_similarity_threshold
        self._setup_qdrant_collections()

        # Sentence transformer for embedding column names
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        logger.info("Sentence transformer initialized")

    def _setup_qdrant_collections(self):
        """Setup both column and cache collections"""
        try:
            if not self.qdrant_client.collection_exists(self.column_collection):
                self.qdrant_client.create_collection(
                    collection_name=self.column_collection,
                    vectors_config=models.VectorParams(
                        size=384,  # bge-small-en-v1.5 dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.column_collection}")

            # NEW: Insights cache collection
            if not self.qdrant_client.collection_exists(self.insights_cache_collection):
                self.qdrant_client.create_collection(
                    collection_name=self.insights_cache_collection,
                    vectors_config=models.VectorParams(
                        size=384,  # bge-small-en-v1.5 dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.insights_cache_collection}")
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collections: {e}")
            raise

    async def _check_cache(self, insight: str, datasource_hash: str) -> Optional[EvaluationResult]:

        try:
            # Embed the insight
            insight_embedding = self.embedder.encode(insight, convert_to_numpy=True, show_progress_bar=False)

            # Search for similar insights
            search_results = self.qdrant_client.query_points(
                collection_name=self.insights_cache_collection,
                query=insight_embedding.tolist(),
                limit=1,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="datasource_hash",
                            match=models.MatchValue(value=datasource_hash)
                        )
                    ]
                )
            ).points

            if search_results and search_results[0].score >= self.cache_similarity_threshold:
                result = search_results[0]
                logger.info(f"Cache HIT: Found similar insight (similarity: {result.score:.4f})")

                # Reconstruct EvaluationResult from payload
                cached_result = EvaluationResult.from_dict(result.payload['evaluation_result'])
                cached_result.cached = True
                return cached_result
            else:
                logger.info("Cache MISS: No similar insight found")
                return None

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def _store_in_cache(self, insight: str, result: EvaluationResult, datasource_hash: str):
        """Store an evaluated insight in the cache"""
        try:
            # Embed the insight
            insight_embedding = self.embedder.encode(insight, convert_to_numpy=True, show_progress_bar=False)

            # Create point with full evaluation result
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=insight_embedding.tolist(),
                payload={
                    "insight": insight,
                    "datasource_hash": datasource_hash,
                    "evaluation_result": result.to_dict(),
                    "timestamp": asyncio.get_event_loop().time()
                }
            )

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.insights_cache_collection,
                points=[point]
            )
            logger.info(f"Stored insight in cache: {insight[:60]}...")

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")

    def _generate_datasource_hash(self, datasource: str) -> str:
        """Generate a hash to identify the datasource"""
        import hashlib
        # If it's a file path, use the path; otherwise use content hash
        if len(datasource) < 500 and '\n' not in datasource and os.path.isfile(datasource):
            return hashlib.md5(datasource.encode()).hexdigest()
        else:
            return hashlib.md5(datasource.encode()).hexdigest()

    def _parse_csv(self, csv_content: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            csv_file = io.StringIO(csv_content)
            reader = csv.DictReader(csv_file)
            headers = [h.replace('\ufeff', '').strip() for h in reader.fieldnames or []]
            rows = [{k.replace('\ufeff', '').strip(): v for k, v in row.items()} for row in reader]
            logger.info(f"Parsed CSV: {len(rows)} rows, {len(headers)} columns, headers: {headers}")
            return headers, rows
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise ValueError(f"Failed to parse CSV: {e}")

    def _index_csv_schema(self, headers: List[str], csv_path: Optional[str] = None):
        self.qdrant_client.delete_collection(self.column_collection)
        self._setup_qdrant_collections()
        try:
            # Generate embeddings for column names
            column_texts = [f"Column: {col}" for col in headers]  # Simple format
            embeddings = self.embedder.encode(column_texts, convert_to_numpy=True)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "column_name": col,
                        "source": csv_path or "inline_csv",
                        "type": "csv_column"
                    }
                )
                for col, embedding in zip(headers, embeddings)
            ]
            self.qdrant_client.upsert(
                collection_name=self.column_collection,
                points=points
            )
            logger.info(f"Indexed {len(headers)} columns in Qdrant")
        except Exception as e:
            logger.error(f"Failed to index CSV schema in Qdrant: {e}")
            raise

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        if "```" in text:
            match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
        array_pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
        array_match = re.search(array_pattern, text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return text[start_idx:i + 1]
        return text.strip()

    async def _identify_required_columns(self, insight: str, available_columns: List[str]) -> Tuple[
        List[str], Dict[str, str]]:
        try:
            column_matches = {}
            required_columns = []
            prompt = f"""Identify the MINIMUM set of columns needed to verify this insight: {insight}
            Available columns: {', '.join(available_columns)}
            Your task:
            1. Analyze the insight to determine which columns are necessary to verify all factual claims.
            2. Return a JSON array of column names (case-sensitive, exact matches from available columns).
            3. Do NOT include any explanatory text, comments, or additional content outside the JSON array.
            Example: ["WorkLifeBalance", "Gender", "OverTime"]
            Ensure the output is valid JSON."""
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = message.content[0].text
            try:
                json_str = self._extract_json(raw_response)
                claude_columns = json.loads(json_str)
                for col in claude_columns:
                    if col in available_columns and col not in required_columns:
                        required_columns.append(col)
                        column_matches[col] = "claude_selection"
            except json.JSONDecodeError:
                logger.error(f"Claude returned invalid JSON: {raw_response}")
                column_matches["error"] = "Claude JSON parsing failed"

            # Fallback to Qdrant vector search if insufficient columns
            if len(required_columns) < 1:
                logger.info("Falling back to Qdrant vector search due to insufficient columns")
                query_embedding = self.embedder.encode(insight, convert_to_numpy=True, show_progress_bar=False)
                logger.info(f"Encoded full insight for vector search: {insight[:50]}...")
                search_result = self.qdrant_client.query_points(
                    collection_name=self.column_collection,
                    query=query_embedding.tolist(),
                    limit=5,
                    with_payload=True
                ).points
                for result in search_result:
                    if result.score > 0.7:
                        matched_col = result.payload["column_name"]
                        logger.info(f"Matched insight to column '{matched_col}' (score: {result.score})")
                        column_matches[matched_col] = f"score: {result.score}"
                        if matched_col not in required_columns and matched_col in available_columns:
                            required_columns.append(matched_col)
                    else:
                        logger.info(f"Skipped column '{result.payload['column_name']}' (score: {result.score} < 0.7)")

            # Validate columns
            validated_columns = [col for col in required_columns if col in available_columns]
            if not validated_columns:
                logger.warning("No valid columns identified, using first 5 as fallback")
                validated_columns = available_columns[:5]
                column_matches["error"] = "No valid columns matched"
            return validated_columns, column_matches
        except Exception as e:
            logger.error(f"Error identifying columns: {e}")
            return available_columns[:5], {"error": str(e)}

    def _filter_data(self, all_rows: List[Dict[str, Any]], required_columns: List[str]) -> List[Dict[str, Any]]:
        filtered = []
        for row in all_rows:
            filtered_row = {col: row.get(col) for col in required_columns if col in row}
            if filtered_row:
                filtered.append(filtered_row)
        return filtered

    async def evaluate_insight(self, insight: str, datasource: Optional[str] = None,
                               alpha: float = 0.5, use_cache: bool = True) -> EvaluationResult:
        #Evaluate an insight, optionally using cache.


        try:
            logger.info(f"Evaluating insight with alpha={alpha}, use_cache={use_cache}")
            if not 0.0 <= alpha <= 1.0:
                alpha = 0.5
            if not datasource:
                return EvaluationResult(
                    insight=insight,
                    claims=[],
                    correctness_score=None,
                    insightfulness_scores=InsightfulnessScores(),
                    combined_score=None,
                    alpha=alpha,
                    llm_explanation="No datasource provided",
                    data_summary="No datasource provided",
                    column_matches={}
                )

            # Handle CSV input
            csv_content = datasource
            if len(datasource) < 500 and '\n' not in datasource and os.path.isfile(datasource):
                logger.info(f"Reading file: {datasource}")
                with open(datasource, 'r', encoding='utf-8') as f:
                    csv_content = f.read()

            # Check cache first
            if use_cache:
                datasource_hash = self._generate_datasource_hash(csv_content)
                cached_result = await self._check_cache(insight, datasource_hash)
                if cached_result:
                    logger.info("Returning cached result")
                    return cached_result

            headers, all_rows = self._parse_csv(csv_content)
            data_summary = f"Dataset: {len(all_rows)} rows, {len(headers)} columns, headers: {headers}"
            logger.info(data_summary)
            self._index_csv_schema(headers, datasource if os.path.isfile(datasource) else None)

            required_columns, column_matches = await self._identify_required_columns(insight, headers)
            logger.info(f"Required columns: {required_columns}, matches: {column_matches}")

            # Filter data
            filtered_data = self._filter_data(all_rows, required_columns)
            logger.info(f"Filtered data: {len(filtered_data)} rows, {len(required_columns)} columns")

            # Verify claims
            claims, raw_response = await self._verify_claims_with_filtered_data(
                insight, required_columns, filtered_data, data_summary
            )
            correctness_score = self._calculate_correctness(claims)

            # Evaluate insightfulness
            insightfulness_scores, explanation, _ = await self._evaluate_insightfulness(insight, data_summary)

            # Calculate combined score
            combined_score = self._calculate_combined_score(correctness_score, insightfulness_scores.weighted_average(),
                                                            alpha)

            result = EvaluationResult(
                insight=insight,
                claims=claims,
                correctness_score=correctness_score,
                insightfulness_scores=insightfulness_scores,
                combined_score=combined_score,
                alpha=alpha,
                llm_explanation=explanation,
                data_summary=data_summary,
                column_matches=column_matches,
                cached=False
            )

            if use_cache:
                datasource_hash = self._generate_datasource_hash(csv_content)
                await self._store_in_cache(insight, result, datasource_hash)

            logger.info(f"Evaluation complete. Combined score: {combined_score}")
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate insight: {e}", exc_info=True)
            raise

    async def evaluate_insights_batch(self, insights: List[str], datasource: Optional[str] = None,
                                      alpha: float = 0.5, delay_between_insights: float = 0,
                                      use_cache: bool = True) -> BatchEvaluationResult:

        try:
            logger.info(
                f"Batch evaluating {len(insights)} insights with alpha={alpha}, delay={delay_between_insights}s, use_cache={use_cache}")
            if not 0.0 <= alpha <= 1.0:
                alpha = 0.5

            if not datasource:
                logger.error("No datasource provided for batch evaluation")
                return BatchEvaluationResult(
                    results=[],
                    data_summary="No datasource provided",
                    total_insights=len(insights),
                    successful_evaluations=0,
                    failed_evaluations=len(insights),
                    average_combined_score=None,
                    cache_hits=0
                )

            csv_content = datasource
            if len(datasource) < 500 and '\n' not in datasource and os.path.isfile(datasource):
                logger.info(f"Reading file: {datasource}")
                with open(datasource, 'r', encoding='utf-8') as f:
                    csv_content = f.read()

            datasource_hash = self._generate_datasource_hash(csv_content)

            headers, all_rows = self._parse_csv(csv_content)
            data_summary = f"Dataset: {len(all_rows)} rows, {len(headers)} columns"
            logger.info(data_summary)
            self._index_csv_schema(headers, datasource if os.path.isfile(datasource) else None)

            results = []
            successful = 0
            failed = 0
            cache_hits = 0

            for i, insight in enumerate(insights, 1):
                logger.info(f"Evaluating insight {i}/{len(insights)}: {insight[:60]}...")
                try:
                    cached_result = None
                    if use_cache:
                        cached_result = await self._check_cache(insight, datasource_hash)

                    if cached_result:
                        results.append(cached_result)
                        successful += 1
                        cache_hits += 1
                        logger.info(f"Insight {i} returned from cache. Score: {cached_result.combined_score}")
                    else:
                        required_columns, column_matches = await self._identify_required_columns(insight, headers)
                        logger.info(f"Required columns for insight {i}: {required_columns}")
                        filtered_data = self._filter_data(all_rows, required_columns)
                        claims, _ = await self._verify_claims_with_filtered_data(
                            insight, required_columns, filtered_data, data_summary
                        )
                        correctness_score = self._calculate_correctness(claims)

                        # Evaluate insightfulness
                        insightfulness_scores, explanation, _ = await self._evaluate_insightfulness(insight,
                                                                                                    data_summary)

                        # Calculate combined score
                        combined_score = self._calculate_combined_score(
                            correctness_score,
                            insightfulness_scores.weighted_average(),
                            alpha
                        )

                        result = EvaluationResult(
                            insight=insight,
                            claims=claims,
                            correctness_score=correctness_score,
                            insightfulness_scores=insightfulness_scores,
                            combined_score=combined_score,
                            alpha=alpha,
                            llm_explanation=explanation,
                            data_summary=data_summary,
                            column_matches=column_matches,
                            cached=False
                        )
                        results.append(result)
                        successful += 1

                        #  Store in cache
                        if use_cache:
                            await self._store_in_cache(insight, result, datasource_hash)

                        logger.info(f"Insight {i} evaluated successfully. Score: {combined_score}")

                    # Add delay between insights to avoid rate limits
                    if i < len(insights):
                        logger.info(f"Waiting {delay_between_insights}s before next insight...")
                        await asyncio.sleep(delay_between_insights)

                except Exception as e:
                    logger.error(f"Failed to evaluate insight {i}: {e}", exc_info=True)
                    results.append(EvaluationResult(
                        insight=insight,
                        claims=[],
                        correctness_score=None,
                        insightfulness_scores=InsightfulnessScores(),
                        combined_score=None,
                        alpha=alpha,
                        llm_explanation=f"Error: {str(e)}",
                        data_summary=data_summary,
                        column_matches={"error": str(e)}
                    ))
                    failed += 1
                    if i < len(insights):
                        await asyncio.sleep(delay_between_insights)

            # Calculate average score
            scored_results = [r.combined_score for r in results if r.combined_score is not None]
            avg_score = sum(scored_results) / len(scored_results) if scored_results else None

            batch_result = BatchEvaluationResult(
                results=results,
                data_summary=data_summary,
                total_insights=len(insights),
                successful_evaluations=successful,
                failed_evaluations=failed,
                average_combined_score=avg_score,
                cache_hits=cache_hits
            )

            logger.info(
                f"Batch evaluation complete. {successful} successful, {failed} failed, {cache_hits} cache hits, avg score: {avg_score}")
            return batch_result

        except Exception as e:
            logger.error(f"Failed batch evaluation: {e}", exc_info=True)
            raise
    # ===================================== Correctness Evaluation =========================================

    async def _verify_claims_with_filtered_data(self, insight: str, columns: List[str],
                                                filtered_data: List[Dict[str, Any]], data_summary: str) -> Tuple[
        List[Claim], str]:
        data_json = json.dumps(filtered_data, separators=(',', ':'), default=str)
        if len(data_json) > 150000:
            sample_size = int(len(filtered_data) * 150000 / len(data_json))
            filtered_data = filtered_data[:sample_size]
            data_json = json.dumps(filtered_data, separators=(',', ':'), default=str)
            logger.warning(f"Data too large, sampled to {sample_size} rows")

        prompt = f"""You are verifying claims in a data insight by calculating statistics from the provided data.
Insight to verify: {insight}
Dataset info: {data_summary}
Columns provided: {', '.join(columns)}
Filtered data (JSON): {data_json}

Your task:
1. Extract ALL factual claims from the insight
2. For EACH claim, calculate the actual statistics from this data
3. Compare calculated values to claimed values
4. Assign truth values based on accuracy

CRITICAL - You MUST perform calculations:
- Count rows, calculate percentages, means, etc.
- For attrition: Count where column="Yes" / total in group × 100
- For averages: Sum values / count
- Show calculation steps in reasoning

Truth Value Scale:
- 1.0 = Perfect match (within 2%)
- 0.8-0.9 = Very close (within 5-10%)
- 0.5-0.7 = Partially correct (within 10-20%)
- 0.2-0.4 = Significantly off (>20% difference)
- 0.0 = Contradicted
- null = Cannot verify (missing columns)

CRITICAL: Return ONLY a valid JSON array, nothing else. No markdown, no explanations.
[
  {{
    "text": "exact claim from insight",
    "truth_value": 1.0,
    "reasoning": "Calculated: [your calculation]. Claimed: [value]. Match: [assessment]"
  }}
]
"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = message.content[0].text
            logger.info("Claim verification completed")
            logger.debug(f"Raw response: {raw_response[:500]}...")

            json_str = self._extract_json(raw_response)
            logger.debug(f"Extracted JSON: {json_str[:500]}...")

            if not json_str or json_str.strip() == "":
                logger.error("Empty JSON string after extraction")
                return [Claim(text=insight, truth_value=None,
                              reasoning="Failed to extract JSON from response")], raw_response

            claims_data = json.loads(json_str)

            # Handle both dict and list responses
            if isinstance(claims_data, dict):
                # If single claim returned as dict, wrap in list
                claims_data = [claims_data]
            elif not isinstance(claims_data, list):
                logger.error(f"Unexpected claims_data type: {type(claims_data)}")
                return [Claim(text=insight, truth_value=None, reasoning="Invalid response format")], raw_response

            claims = []
            for claim_dict in claims_data:
                if isinstance(claim_dict, dict):
                    claims.append(Claim(**claim_dict))
                else:
                    logger.warning(f"Skipping non-dict claim: {claim_dict}")

            if not claims:
                logger.warning("No valid claims extracted, creating default claim")
                return [
                    Claim(text=insight, truth_value=None, reasoning="No claims extracted from response")], raw_response

            for claim in claims:
                logger.info(f"Claim: {claim.text[:60]}... | Truth: {claim.truth_value}")
            return claims, raw_response

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Attempted to parse: {json_str[:500] if 'json_str' in locals() else 'N/A'}...")
            return [Claim(text=insight, truth_value=None,
                          reasoning=f"JSON parsing error: {e}")], raw_response if 'raw_response' in locals() else str(e)
        except Exception as e:
            logger.error(f"Error verifying claims: {e}", exc_info=True)
            return [Claim(text=insight, truth_value=None, reasoning=f"Error: {e}")], str(e)

    # ===================================== Insightfulness Evaluation =========================================

    async def _evaluate_insightfulness(self, insight: str, data_summary: str) -> Tuple[InsightfulnessScores, str, str]:
        prompt = f"""Rate this data insight on four dimensions (0.0-1.0 each):
            1. Actionability: Can it drive decisions/actions?
            2. Relevance: Is it valuable for business?
            3. Clarity: Is it clear and concise?
            4. Novelty: Does it reveal non-obvious patterns?
            Insight: {insight}
            Data: {data_summary}
            Return ONLY JSON:
            {{
              "actionability": 0.85,
              "relevance": 0.90,
              "clarity": 0.88,
              "novelty": 0.75,
              "explanation": "Brief explanation..."
            }}
            """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = message.content[0].text
            json_str = self._extract_json(raw_response)
            result = json.loads(json_str)
            scores = InsightfulnessScores(
                actionability=result.get('actionability', 0.5),
                relevance=result.get('relevance', 0.5),
                clarity=result.get('clarity', 0.5),
                novelty=result.get('novelty', 0.5)
            )
            explanation = result.get('explanation', '')
            return scores, explanation, raw_response
        except Exception as e:
            logger.error(f"Error evaluating insightfulness: {e}")
            return InsightfulnessScores(), "Error", str(e)

    def _calculate_correctness(self, claims: List[Claim]) -> Optional[float]:
        valid_claims = [c for c in claims if c.truth_value is not None]
        if not valid_claims:
            return None
        return sum(c.truth_value for c in valid_claims) / len(valid_claims)

    def _calculate_combined_score(self, correctness: Optional[float], insightfulness: float, alpha: float) -> Optional[
        float]:
        if correctness is None or correctness == 0 or insightfulness == 0:
            return None
        return 1 / ((alpha / insightfulness) + ((1 - alpha) / correctness))


evaluator_server = None


@click.command()
@click.option("--port", default=8000, help="Port to run the server on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--anthropic-api-key", help="Anthropic API key")
@click.option("--model", default="claude-3-5-haiku-latest", help="Claude model to use")
@click.option("--use-qdrant-memory", is_flag=True, help="Use in-memory Qdrant")
@click.option("--cache-threshold", default=0.95, help="Similarity threshold for cache matching (0.0-1.0)")
def main(port: int, host: str, anthropic_api_key: str, model: str, use_qdrant_memory: bool,
         cache_threshold: float) -> int:
    global evaluator_server
    app = Server("insight-evaluator")
    evaluator_server = InsightEvaluatorServer(
        anthropic_api_key=anthropic_api_key,
        model=model,
        use_qdrant_memory=use_qdrant_memory,
        cache_similarity_threshold=cache_threshold
    )

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="evaluate_insight",
                description="Evaluate a data insight for correctness and insightfulness. Uses cache to avoid re-evaluating similar insights.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "insight": {"type": "string", "description": "The insight text to evaluate"},
                        "datasource": {"type": "string",
                                       "description": "CSV file "},
                        "alpha": {"type": "number",
                                  "description": "Weight for correctness vs insightfulness (0.0-1.0, default 0.5)",
                                  "default": 0.5},
                        "use_cache": {"type": "boolean",
                                      "description": "Whether to use cache for this evaluation (default: true)",
                                      "default": True}
                    },
                    "required": ["insight", "datasource"]
                }
            ),
            types.Tool(
                name="evaluate_insights_batch",
                description="Evaluate multiple data insights from the same datasource. Uses cache to skip previously evaluated insights.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "insights": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of insight texts to evaluate"
                        },
                        "datasource": {
                            "type": "string",
                            "description": "CSV file"
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Weight for correctness vs insightfulness (0.0-1.0, default 0.5)",
                            "default": 0.5
                        },
                        "use_cache": {
                            "type": "boolean",
                            "description": "Whether to use cache for this batch (default: true)",
                            "default": True
                        }
                    },
                    "required": ["insights", "datasource"]
                }
            )
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "evaluate_insight":
                result = await evaluator_server.evaluate_insight(
                    insight=arguments["insight"],
                    datasource=arguments.get("datasource"),
                    alpha=arguments.get("alpha", 0.5),
                    use_cache=arguments.get("use_cache", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]
            elif name == "evaluate_insights_batch":
                result = await evaluator_server.evaluate_insights_batch(
                    insights=arguments["insights"],
                    datasource=arguments.get("datasource"),
                    alpha=arguments.get("alpha", 0.5),
                    use_cache=arguments.get("use_cache", True)
                )
                return [types.TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())
        return Response()

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    logger.info(f"Starting Insight Evaluator MCP server on {host}:{port}")
    uvicorn.run(starlette_app, host=host, port=port)
    return 0


if __name__ == "__main__":
    main()