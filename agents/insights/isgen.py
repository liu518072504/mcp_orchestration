import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import random

from transformers.testing_utils import backend_reset_max_memory_allocated

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict
import re

import anthropic
import click
import mcp.types as types
import numpy as np
import torch
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
import aiosqlite
import pandas as pd
from sentence_transformers import SentenceTransformer, util

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================== Data Models =====================================================
# ============================================================================================================

@dataclass
class InsightCard:
    question: str
    reason: str
    breakdown: str  # Column for grouping insql
    measure: str  # e.g., MEAN(Performance)
    iteration: int = 0
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reason": self.reason,
            "breakdown": self.breakdown,
            "measure": self.measure,
            "iteration": self.iteration
        }

    def format_for_prompt(self) -> str:
        return f"""[INSIGHT]
REASON: {self.reason}
QUESTION: {self.question}
BREAKDOWN: {self.breakdown}
MEASURE: {self.measure}
[/INSIGHT]"""


# Memory Storage Local
import json
from pathlib import Path


# ========================================== Memory Manager =====================================================
# ===============================================================================================================

class EpisodicMemory:
    def __init__(self, storage_path: str = "episodic_memory.json"):
        self.memories: List[Dict[str, Any]] = []
        self.storage_path = Path(storage_path)
        self._load_memories()

    def _load_memories(self):
        """Load memories from disk if they exist"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                self.memories = json.load(f)

    def _save_memories(self):
        """Save memories to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memories, indent=2, fp=f)

    def add_memory(self, query: str, sql: str, success: bool,
                   reflection: str, error: Optional[str] = None):
        memory = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "sql": sql,
            "success": success,
            "reflection": reflection,
            "error": error
        }
        self.memories.append(memory)
        self._save_memories()  # Save after each addition


# ===================================== Semantic Similarity============================================
# =====================================================================================================

class SemanticSimilarityModel:

    # Uses model all-MiniLM-L6-v2
    def __init__(self):
        logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model loaded successfully")

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # Compute cosine similarity
        return float(util.cos_sim(embedding1, embedding2))

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode([text1])[0]
        emb2 = self.encode([text2])[0]
        return self.compute_similarity(emb1, emb2)


# ======================================== Insight Card Filter ===============================================
# ============================================================================================================

class InsightCardFilter:

    def __init__(self, similarity_model: SemanticSimilarityModel, reflexion_sql_agent):
        self.similarity_model = similarity_model
        self.reflexion_sql_agent = reflexion_sql_agent
        logger.info("InsightCardFilter initialized")

    async def filter(self, new_cards: List[InsightCard], existing_cards: List[InsightCard], schema: Dict[str, Any]) -> \
    List[InsightCard]:
        logger.info(f"Starting filtering: {len(new_cards)} new cards")
        # Stage 1: Schema relevance
        # Keep questions that are semantically similar to table schema
        cards = self.filter_schema_relevance(new_cards, schema, threshold=0.1)
        logger.info(f"After schema relevance filter: {len(cards)} cards")
        # Stage 2: Duplicates (remove duplicated questions based on semantic similarity)
        cards = self.filter_duplicates(cards, existing_cards, threshold=0.85)
        logger.info(f"After duplicate filter: {len(cards)} cards")
        # Stage 3: Triviality check
        # Discard question if it only returns one row for sql query

        return cards

    def filter_schema_relevance(self, cards: List[InsightCard], schema: Dict[str, Any], threshold: float = 0.1) -> List[
        InsightCard]:
        # Create schema text representation
        schema_text = self._format_schema_text(schema)
        schema_embedding = self.similarity_model.encode([schema_text])[0]

        filtered = []
        for card in cards:
            # Compute embedding for question if not computed previously
            if card.embedding is None:
                card.embedding = self.similarity_model.encode([card.question])[0]
            # Check similarity to schema
            similarity = self.similarity_model.compute_similarity(
                card.embedding,
                schema_embedding
            )
            if similarity > threshold:
                filtered.append(card)
            else:
                logger.debug(f"Filtered out (low schema relevance {similarity:.3f}): {card.question}")

        return filtered

    def filter_duplicates(self, new_cards: List[InsightCard], existing_cards: List[InsightCard],
                          threshold: float = 0.85) -> List[InsightCard]:
        if not existing_cards:
            return new_cards

        # Ensure all cards have embeddings
        for card in new_cards:
            if card.embedding is None:
                card.embedding = self.similarity_model.encode([card.question])[0]

        for card in existing_cards:
            if card.embedding is None:
                card.embedding = self.similarity_model.encode([card.question])[0]
        # Get existing embeddings
        existing_embeddings = np.array([card.embedding for card in existing_cards])
        filtered = []
        for card in new_cards:
            # Compute similarity with all existing cards
            similarities = [
                self.similarity_model.compute_similarity(card.embedding, existing_emb)
                for existing_emb in existing_embeddings
            ]
            max_similarity = max(similarities) if similarities else 0.0
            if max_similarity < threshold:
                filtered.append(card)
            else:
                logger.debug(f"Filtered out (duplicate {max_similarity:.3f}): {card.question}")

        return filtered

    def _format_schema_text(self, schema: Dict[str, Any]) -> str:
        parts = [f"File: {schema['file_name']}"]
        parts.append("Columns: " + ", ".join(schema['columns']))
        return " ".join(parts)


# ======================================= Prompt Builder =====================================================
# ============================================================================================================

# Builds prompts with in-context examples from previous iterations
class PromptBuilder:

    def __init__(self):
        self.base_template = ""

    def build_prompt(self, schema: Dict[str, Any], stats: str, previous_cards: List[InsightCard], num_questions: int,
                     num_examples: int = 6) -> str:
        schema_text = self._format_schema(schema)
        examples_text = self._format_examples(previous_cards, num_examples)
        prompt = f"""You are a data analyst generating Insight Cards for exploratory data analysis.

Dataset Schema:
{schema_text}

Basic Statistics:
{stats}

An Insight Card consists of:
1. REASON: Why this question is insightful
2. QUESTION: The analytical question
3. BREAKDOWN: Column to group by (e.g., Year, Department, Category)
4. MEASURE: Aggregation to analyze (e.g., MEAN(price), COUNT(*), SUM(revenue))

Generate {num_questions} unique Insight Cards. Focus on:
- Trends over time or categories
- Comparisons across groups
- Distributions and patterns
- Outliers and anomalies
- Relationships between variables

IMPORTANT: Wrap each Insight Card in [INSIGHT] and [/INSIGHT] tags.

Format:
[INSIGHT]
REASON: <explanation of why this is insightful>
QUESTION: <natural language question>
BREAKDOWN: <column name>
MEASURE: <aggregation function(column)>
[/INSIGHT]

{examples_text}

Now generate {num_questions} NEW and UNIQUE Insight Cards that are different from the examples above:"""

        return prompt

    def _format_schema(self, schema: Dict[str, Any]) -> str:

        lines = [f"File: {schema['file_name']}"]
        lines.append("Columns:")
        for col in schema['columns']:
            col_type = schema['data_types'].get(col, 'unknown')
            lines.append(f"  - {col} ({col_type})")
        return "\n".join(lines)

    def _format_examples(self, cards: List[InsightCard], num_examples: int) -> str:
        # Sample and format previous cards as examples, provide initial example card if there's no previous cards
        if not cards:
            return """Examples:

[INSIGHT]
REASON: To analyze whether there are trends in average performance over time
QUESTION: How has employee performance varied over the years?
BREAKDOWN: Year
MEASURE: MEAN(Performance)
[/INSIGHT]

[INSIGHT]
REASON: To identify which department has the highest sales contribution
QUESTION: Which department generates the most revenue?
BREAKDOWN: Department
MEASURE: SUM(Revenue)
[/INSIGHT]"""

        # Sample cards from previous iterations
        sampled = random.sample(cards, min(num_examples, len(cards)))
        examples = ["Examples from previous analysis:"]
        for card in sampled:
            examples.append(card.format_for_prompt())

        return "\n\n".join(examples)


# ===================================== Question Generator (Updated) =========================================
# ============================================================================================================

class QuestionGenerator:

    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            logger.error("No Anthropic API key found!")
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        logger.info("Claude client initialized successfully")

        # Initialize class components
        self.sql_agent = SQLAgent(self.claude_client)
        self.reflexion_sql_agent = ReflexionSQLAgent(self.claude_client)
        self.similarity_model = SemanticSimilarityModel()
        self.card_filter = InsightCardFilter(self.similarity_model, self.reflexion_sql_agent)
        self.prompt_builder = PromptBuilder()

        logger.info("QuestionGenerator initialized")

    async def _extract_file_info(self, file_path: Path) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path, nrows=5)
            info = {
                "file_name": file_path.stem,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict()
            }
            return info
        except Exception as e:
            logger.error(f"Could not read csv file {file_path}: {e}")
            return {"error": str(e)}

    def _format_table_info_for_claude(self, table_info: Dict) -> str:
        if "error" in table_info:
            return f"Error reading file: {table_info['error']}"

        columns_with_types = [
            f"  - {col} ({table_info['data_types'].get(col, 'unknown')})"
            for col in table_info['columns']
        ]

        return f"""File: {table_info['file_name']}
Columns and Types:
{chr(10).join(columns_with_types)}""".strip()

    def _parse_insight_cards(self, response: str, iteration: int) -> List[InsightCard]:
        # Parse LLM response into InsightCard objects
        # Extracts [INSIGHT]...[/INSIGHT] blocks
        pattern = r'\[INSIGHT\](.*?)\[/INSIGHT\]'
        matches = re.findall(pattern, response, re.DOTALL)

        cards = []
        for match in matches:
            try:
                card = self._parse_single_card(match, iteration)
                if card:
                    cards.append(card)
            except Exception as e:
                logger.warning(f"Failed to parse card: {e}")
                continue

        return cards

    def _parse_single_card(self, text: str, iteration: int) -> Optional[InsightCard]:
        # Parse a single [INSIGHT]...[/INSIGHT] block into InsightCard
        reason_match = re.search(r'REASON:\s*(.+?)(?=\n\s*QUESTION:|\Z)', text, re.DOTALL)
        question_match = re.search(r'QUESTION:\s*(.+?)(?=\n\s*BREAKDOWN:|\Z)', text, re.DOTALL)
        breakdown_match = re.search(r'BREAKDOWN:\s*(.+?)(?=\n\s*MEASURE:|\Z)', text, re.DOTALL)
        measure_match = re.search(r'MEASURE:\s*(.+?)(?=\Z)', text, re.DOTALL)

        if not all([reason_match, question_match, breakdown_match, measure_match]):
            logger.warning("Missing required fields in insight card")
            return None

        card = InsightCard(
            reason=reason_match.group(1).strip(),
            question=question_match.group(1).strip(),
            breakdown=breakdown_match.group(1).strip(),
            measure=measure_match.group(1).strip(),
            iteration=iteration
        )

        return card

    async def generate_insight_cards_iterative(
            self,
            file_path: Path,
            num_iterations: int = 5,
            samples_per_iteration: int = 2,  # num of sample per iter
            temperature: float = 1.0,
            cards_per_sample: int = 5,  # of insight cards per sample
            num_examples: int = 6  # of few-shot examples as in-context examples for LLM
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Starting Question generation: {num_iterations} iterations")

            # Extract schema
            schema = await self._extract_file_info(file_path)
            if "error" in schema:
                return {
                    "success": False,
                    "error": f"Failed to read file: {schema['error']}"
                }
            # basic statistics generation
            stats = await self._generate_basic_stats(file_path, schema)
            # Create temp database for filtering
            db_path = await self.reflexion_sql_agent._create_temp_database(file_path)

            all_cards = []
            for iteration in range(num_iterations):
                logger.info(f"\n{'=' * 50}")
                logger.info(f"ITERATION {iteration + 1}/{num_iterations}")
                logger.info(f"{'=' * 50}")

                # Generate cards for the iteration
                iteration_cards = await self._generate_iteration(
                    schema=schema,
                    stats=stats,
                    previous_cards=all_cards,
                    iteration=iteration,
                    samples=samples_per_iteration,
                    temperature=temperature,
                    cards_per_sample=cards_per_sample,
                    num_examples=num_examples
                )
                # Filtering:
                filtered_cards = await self.card_filter.filter(
                    new_cards=iteration_cards,
                    existing_cards=all_cards,
                    schema=schema
                )
                logger.info(
                    f"Iteration {iteration + 1}: Generated {len(iteration_cards)}, Filtered to {len(filtered_cards)}")

                # Add to collection
                all_cards.extend(filtered_cards)

            # Cleanup
            await self.reflexion_sql_agent._cleanup_temp_database(db_path)

            logger.info(f"{len(all_cards)} total insight cards")

            return {
                "success": True,
                "file_name": file_path.stem,
                "total_iterations": num_iterations,
                "total_cards": len(all_cards),
                "insight_cards": [card.to_dict() for card in all_cards],
                "cards_by_iteration": self._group_by_iteration(all_cards)
            }

        except Exception as e:
            logger.error(f"Error in iterative generation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_iteration(
            self,
            schema: Dict[str, Any],
            stats: str,
            previous_cards: List[InsightCard],
            iteration: int,
            samples: int,
            temperature: float,
            cards_per_sample: int,
            num_examples: int
    ) -> List[InsightCard]:

        prompt = self.prompt_builder.build_prompt(
            schema=schema,
            stats=stats,
            previous_cards=previous_cards,
            num_questions=cards_per_sample,
            num_examples=num_examples
        )

        all_cards = []
        # Sample LLM
        for sample_idx in range(samples):
            logger.info(f" Sample {sample_idx + 1}/{samples}")

            try:
                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=3000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text.strip()

                # Parse insight cards
                cards = self._parse_insight_cards(response_text, iteration)
                logger.info(f"Parsed {len(cards)} cards from sample")

                all_cards.extend(cards)

            except Exception as e:
                logger.error(f"Error in sample {sample_idx + 1}: {e}")
                continue

        return all_cards

    async def _generate_basic_stats(self, file_path: Path, schema: Dict[str, Any]) -> str:
        # return only table schema
        # Enhance this with actual statistics if needed
        return self._format_table_info_for_claude(schema)

    def _group_by_iteration(self, cards: List[InsightCard]) -> Dict[int, int]:
        from collections import defaultdict
        groups = defaultdict(int)
        for card in cards:
            groups[card.iteration] += 1
        return dict(groups)

    # to directly generate (unevaluated) insights from insight cards
    async def generate_insights(
            self,
            file_path: Path,  # Changed from info: Dict
            num_iterations: int = 5,  # Add configuration params
            samples_per_iteration: int = 2,
            temperature: float = 1.0,
            cards_per_sample: int = 5
    ) -> Dict[str, Any]:
        try:
            # Generate Insight Cards (iterative)
            cards_result = await self.generate_insight_cards_iterative(
                file_path=file_path,
                num_iterations=num_iterations,
                samples_per_iteration=samples_per_iteration,
                temperature=temperature,
                cards_per_sample=cards_per_sample
            )
            if not cards_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to generate insight cards: {cards_result.get('error', 'Unknown error')}"
                }
            insight_cards = cards_result["insight_cards"]

            if not insight_cards:
                return {
                    "success": False,
                    "error": "No insight cards were generated"
                }
            logger.info(f"Generated {len(insight_cards)} insight cards")

            # Extract questions, breakdowns and measures from insight cards
            questions = [card["question"] for card in insight_cards]
            breakdowns = [card["breakdown"] for card in insight_cards]
            measures = [card["measure"] for card in insight_cards]
            logger.info(f"Extracted {len(questions)} questions")

            # get table info
            table_info = await self._extract_file_info(file_path)
            if "error" in table_info:
                return {
                    "success": False,
                    "error": f"Failed to read file: {table_info['error']}"
                }

            # Process questions with SQLAgent to generate natural language insights
            sql_result = await self.reflexion_sql_agent.process_questions(
                questions=questions,
                breakdowns=breakdowns,
                measures=measures,
                file_path=file_path,
                table_info=table_info
            )
            if not sql_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to process questions: {sql_result.get('error', 'Unknown error')}"
                }

            # Match each card with its corresponding SQL result (not necessary)
            combined_insights = []
            for i, card in enumerate(insight_cards):
                if i < len(sql_result["details"]):
                    sql_detail = sql_result["details"][i]
                    combined_insights.append({
                        "insight_card": card,
                        "sql_analysis": {
                            "sql_query": sql_detail["sql"],
                            "raw_result": sql_detail["raw_result"],
                            "natural_language": sql_detail["text"]
                        }
                    })
                else:
                    # Card without SQL result
                    combined_insights.append({
                        "insight_card": card,
                        "sql_analysis": {
                            "error": "No SQL result available"
                        }
                    })

            # Return comprehensive result
            return {
                "success": True,
                "file_name": file_path.stem,
                "summary": {
                    "total_insight_cards": len(insight_cards),
                    "total_questions_processed": sql_result["processed_count"],
                    "iterations_used": num_iterations
                },
                "natural_language_insights": sql_result["statistics"]  # unevaluated insights in natural language
            }

        except Exception as e:
            logger.error(f"Error in generate_insights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }


# ============================================== SQLAgent (Update 2025.10.19) ========================================
# Update 2025.10.28
class SQLAgent:
    def __init__(self, claude_client: anthropic.Anthropic, max_retries: int = 5):
        self.claude_client = claude_client
        self.max_retries = max_retries
        logger.info("SQLAgent initialized")

    async def process_questions(
            self,
            questions: List[str],
            breakdowns: List[str],
            measures: List[str],
            file_path: Path,
            table_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Create temporary database and load csv file
            db_path = await self._create_temp_database(file_path)
            table_name = "data_table"
            results = []
            nl_statistics = []
            # parse insight cards:
            async with aiosqlite.connect(db_path) as db:
                for i, (question, breakdown, measure) in enumerate(zip(questions, breakdowns, measures), 1):
                    logger.info(f"Processing question {i}/{len(questions)}")
                    try:
                        # Convert question to SQL query
                        sql_result = await self._question_to_sql(question, breakdown, measure, table_info, table_name,
                                                                 db)
                        if not sql_result["success"]:
                            logger.warning(f"Failed to generate SQL for: {question}")
                            continue
                        sql_query = sql_result["sql"]
                        # Execute SQL
                        execution_result = await self._execute_sql(sql_query, db)
                        if not execution_result["success"]:
                            logger.warning(f"Failed to execute SQL: {execution_result['error']}")
                            continue
                        # SQL result to natural language
                        nl_result = await self._result_to_text(question=question, sql=sql_query,
                                                               result=execution_result["result"])
                        # Store results
                        results.append({
                            "question": question,
                            "sql": sql_query,
                            "raw_result": execution_result["result"],
                            "text": nl_result
                        })
                        nl_statistics.append(nl_result)

                    except Exception as e:
                        logger.error(f"Error processing question '{question}': {e}")
                        continue

            await self._cleanup_temp_database(db_path)  # cleanup temp database
            return {
                "success": True,
                "statistics": nl_statistics,
                "details": results,
                "processed_count": len(results),
                "total_questions": len(questions)
            }
        except Exception as e:
            logger.error(f"Error in process_questions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _create_temp_database(self, csv_path: Path) -> str:
        try:
            temp_dir = tempfile.gettempdir()
            db_path = os.path.join(temp_dir, f"temp_db_{uuid.uuid4().hex}.db")
            logger.info(f"Creating temporary database at: {db_path}")
            df = pd.read_csv(csv_path)
            conn = sqlite3.connect(db_path)
            df.to_sql('data_table', conn, index=False, if_exists='replace')
            conn.close()
            logger.info(f"Loaded {len(df)} rows into temporary database")
            return db_path
        except Exception as e:
            logger.error(f"Error creating temp database: {e}")
            raise

    async def _cleanup_temp_database(self, db_path: str):
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Cleaned up temporary database: {db_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp database: {e}")

    def _format_schema_for_sql(self, table_info: Dict[str, Any], table_name: str) -> str:
        columns_info = []
        for col in table_info["columns"]:
            col_type = table_info['data_types'].get(col, 'unknown')
            columns_info.append(f"  - {col} ({col_type})")

        return f"""Table: {table_name}
                Columns: {chr(10).join(columns_info)}"""

    def _build_sql_prompt(self, question: str, breakdown: str, measure: str, table_info: Dict[str, Any],
                          table_name: str, error_history: List[Dict]) -> str:
        schema_text = self._format_schema_for_sql(table_info, table_name)

        error_feedback = ""
        if error_history:
            error_feedback = "\n\n## PREVIOUS ATTEMPTS AND ERRORS:\n"
            for i, error_info in enumerate(error_history, 1):
                error_feedback += f"\nAttempt {i}:\n"
                error_feedback += f"SQL: {error_info['sql']}\n"
                error_feedback += f"ERROR: {error_info['error']}\n"
            error_feedback += "\n## INSTRUCTIONS:\n"
            error_feedback += "Analyze the errors above and generate corrected SQL that fixes these issues.\n"

        base_prompt = f"""You are a SQL expert. Generate a SQLite query to answer the analytical question.

        {schema_text}
        Question: {question}
        Breakdown by: {breakdown}
        Measure: {measure}

                    IMPORTANT RULES:
                    1. Use standard SQLite syntax
                    2. The table name is '{table_name}'
                    3. Return ONLY the SQL query, no explanations or markdown
                    4. Use standard SQL functions: COUNT, AVG, SUM, MIN, MAX, GROUP BY, ORDER BY, LIMIT, etc.
                    5. For percentages, use: (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}))
                    6. For date ranges, use MIN and MAX
                    7. Always use proper aggregation with GROUP BY when needed

                    Examples:
                    Question: How many rows are in the dataset?
                    SQL: SELECT COUNT(*) as count FROM {table_name}

                    Question: What is the average price?
                    SQL: SELECT AVG(price) as avg_price FROM {table_name}

                    Question: How many unique categories are there?
                    SQL: SELECT COUNT(DISTINCT category) as unique_count FROM {table_name}

                    Question: What is the most common payment method?
                    SQL: SELECT payment_method, COUNT(*) as count FROM {table_name} GROUP BY payment_method ORDER BY count DESC LIMIT 1

                    Question: What percentage of orders have missing delivery dates?
                    SQL: SELECT (COUNT(CASE WHEN delivery_date IS NULL THEN 1 END) * 100.0 / COUNT(*)) as percentage FROM {table_name}

                    Question: What is the date range?
                    SQL: SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM {table_name}

        Return ONLY the SQL query, no explanations or markdown.{error_feedback}"""

        return base_prompt

    def _clean_sql(self, sql: str) -> str:
        # Clean SQL string from markdown and formatting
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = sql.strip()
        if sql.endswith(';'):
            sql = sql[:-1]
        return sql

    async def _question_to_sql(self, question: str, breakdown: str, measure: str, table_info: Dict[str, Any],
                               table_name: str, db: aiosqlite.Connection) -> Dict[str, Any]:
        attempt = 0
        error_history = []

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"SQL generation attempt {attempt}/{self.max_retries}")

            try:
                # Build prompt with error feedback from previous attempts
                prompt = self._build_sql_prompt(
                    question=question,
                    breakdown=breakdown,
                    measure=measure,
                    table_info=table_info,
                    table_name=table_name,
                    error_history=error_history,
                )

                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )

                sql = response.content[0].text.strip()
                sql = self._clean_sql(sql)
                # Validate SQL syntax
                if not self._validate_sql(sql):
                    error_msg = "SQL validation failed: Invalid or dangerous SQL syntax"
                    logger.warning(f"Attempt {attempt}: {error_msg}")
                    error_history.append({"sql": sql, "error": error_msg})
                    continue
                logger.info(f"Attempt {attempt}: Generated SQL: {sql}")
                # Execute SQL generated from LLM
                exec_result = await self._execute_sql(sql, db)
                if exec_result["success"]:
                    return {
                        "success": True,
                        "sql": sql,
                        "result": exec_result["result"],
                        "attempts": attempt
                    }
                else:
                    # if execution failed, add to error history for next iteration
                    error_msg = exec_result["error"]
                    logger.warning(f"Attempt {attempt}: SQL execution failed: {error_msg}")
                    error_history.append({"sql": sql, "error": error_msg})
                    # Continue to next attempt with error feedback

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt}: Exception during SQL generation: {error_msg}")
                error_history.append({"sql": "N/A", "error": error_msg})

        # All retries exhausted
        return {
            "success": False,
            "error": f"Failed after {self.max_retries} attempts",
            "error_history": error_history,
            "attempts": self.max_retries
        }

    def _validate_sql(self, sql: str) -> bool:  # a basic SQL syntax validation function
        if not sql:
            return False

        sql_lower = sql.lower()
        # Must contain SELECT
        if 'select' not in sql_lower:
            return False
        # Reject data definition language
        dangerous = ['drop', 'delete', 'insert', 'update', 'alter', 'create table',
                     'truncate', 'pragma', 'attach', 'detach']
        if any(word in sql_lower for word in dangerous):
            return False

        return True

    async def _execute_sql(self, sql: str, db: aiosqlite.Connection) -> Dict[str, Any]:
        try:
            # Enable row factory to get dict_like results
            db.row_factory = aiosqlite.Row
            async with db.execute(sql) as cursor:
                rows = await cursor.fetchall()
                result = [dict(row) for row in rows]

                return {
                    "success": True,
                    "result": result
                }
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _result_to_text(self, question: str, sql: str, result: List[Dict]) -> str:
        try:
            if not result:
                result_text = "No result"
            result_lines = []
            for row in result:
                result_lines.append(", ".join(f"{k}={v}" for k, v in row.items()))
            result_text = f"Results:\n" + "\n".join(result_lines)
            prompt = f"""Convert this SQL query result into a natural language statement.
Question: {question}
SQL: {sql}
{result_text}
Provide a concise, clear natural language statement. Be specific and include actual values.
Use appropriate formatting for numbers:
- Use commas for thousands (e.g., 1,000)
- Use 2 decimal places for averages and percentages
- Format dates in readable format

Examples:
Question: How many rows are in the dataset?
Result: count=1000
Answer: There are 1,000 rows in the dataset.

Question: What is the most common payment method?
Result: payment_method=Credit Card, count=678
Answer: The most common payment method is Credit Card, appearing 678 times.

Question: What percentage of orders have missing delivery dates?
Result: percentage=8.5
Answer: 8.5% of orders have missing delivery dates.

Question: What is the date range?
Result: min_date=2023-01-01, max_date=2024-12-31
Answer: The date range spans from 2023-01-01 to 2024-12-31.

Now convert this result:
Question: {question}
{result_text}
Answer:"""
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            nl_text = response.content[0].text.strip()

            return nl_text
        except Exception as e:
            logger.error(f"Error in result_to_natural_language: {e}")
            # Fallback to simple template
            return "Error converting result to natural language"


# ============================================== Reflexion SQL Agent =============================================
class ReflexionSQLAgent(SQLAgent):
    def __init__(self, claude_client, max_retries=5, storage_path: str = "reflexion_memory.json"):
        super().__init__(claude_client, max_retries)
        self.episodic_memory = []  # long-term memory
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # looking for similar past errors
        self.storage_path = Path(storage_path)
        self._load_memory()

    def _load_memory(self):
        # Load episodic memory from disk if exists
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    loaded_data = json.load(f)
                    # Convert stored embeddings back to numpy arrays
                    for memory in loaded_data:
                        if "question_embd" in memory and memory["question_embd"] is not None:
                            memory["question_embd"] = np.array(memory["question_embd"])
                        if "timestamp" in memory and isinstance(memory["timestamp"], str):
                            memory["timestamp"] = datetime.fromisoformat(memory["timestamp"])
                    self.episodic_memory = loaded_data
                    logger.info(f"Loaded {len(self.episodic_memory)} memories from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading memory from {self.storage_path}: {e}")
                self.episodic_memory = []

    def _save_memory(self):
        # Save episodic memory to disk
        try:
            # store data in json format
            serializable_memory = []
            for memory in self.episodic_memory:
                mem_copy = memory.copy()
                if "question_embd" in mem_copy and isinstance(mem_copy["question_embd"], np.ndarray):
                    mem_copy["question_embd"] = mem_copy["question_embd"].tolist()
                if "timestamp" in mem_copy and isinstance(mem_copy["timestamp"], datetime):
                    mem_copy["timestamp"] = mem_copy["timestamp"].isoformat()
                serializable_memory.append(mem_copy)

            with open(self.storage_path, 'w') as f:
                json.dump(serializable_memory, f, indent=2)
            logger.info(f"Saved {len(self.episodic_memory)} memories to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving memory to {self.storage_path}: {e}")

    async def _question_to_sql(self, question: str, breakdown: str, measure: str, table_info: Dict[str, Any],
                               table_name: str, db: aiosqlite.Connection) -> Dict[str, Any]:
        relevant_memory = self._retrieve_relevant_mem(question)
        local_errors = []
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"SQL generation attempt {attempt}/{self.max_retries}")
            # Build prompt with error feedback from previous attempts
            prompt = self._build_reflexion_prompt(
                question=question,
                breakdown=breakdown,
                measure=measure,
                table_info=table_info,
                table_name=table_name,
                local_errors=local_errors,
                past_learnings=relevant_memory
            )

            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            sql = response.content[0].text.strip()
            sql = self._clean_sql(sql)
            # Execute SQL generated from LLM
            exec_result = await self._execute_sql(sql, db)
            if exec_result["success"]:
                return {
                    "success": True,
                    "sql": sql,
                    "result": exec_result["result"]
                }
            else:
                error_msg = exec_result["error"]
                local_errors.append({"success": False, "sql": sql, "error": error_msg})
        if local_errors:
            reflection = await self._generate_reflection(question, local_errors)
            self.episodic_memory.append({
                "question": question,
                "question_embd": self.similarity_model([question])[0],
                "errors": local_errors,
                "reflection": reflection,
                "timestamp": datetime.now()
            })
            self._save_memory()  # Save to disk after adding new memory
        # store to local dir:
        # ===================

        return {"success": False, "error": "Failed after retries"}

    def _retrieve_relevant_mem(self, question: str, top_k: int = 3):
        if not self.episodic_memory:
            return []
        question_embd = self.similarity_model([question])[0]
        similarities = []
        for memory in self.episodic_memory:
            sim = util.cos_sim(question_embd, memory['question_embd'])
            similarities.append((sim, memory))
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [mem["reflection"] for _, mem in similarities[:top_k]]

    def _build_reflexion_prompt(
            self,
            question: str,
            breakdown: str,
            measure: str,
            table_info: Dict[str, Any],
            table_name: str,
            local_errors: List[Dict],
            past_learnings: List[Dict]
    ) -> str:
        schema_text = self._format_schema_for_sql(table_info, table_name)
        base_prompt = f"""
You are a SQL expert. Generate a SQLite query to answer the analytical question.
IMPORTANT RULES:
1. Use standard SQLite syntax
2. The table name is '{table_name}'
3. Return ONLY the SQL query, no explanations or markdown

Examples:
Question: How many rows are in the dataset?
SQL: SELECT COUNT(*) as count FROM {table_name}

Question: How many unique categories are there?
SQL: SELECT COUNT(DISTINCT category) as unique_count FROM {table_name}

Question: What is the most common payment method?
SQL: SELECT payment_method, COUNT(*) as count FROM {table_name} GROUP BY payment_method ORDER BY count DESC LIMIT 1

Question: What percentage of orders have missing delivery dates?
SQL: SELECT (COUNT(CASE WHEN delivery_date IS NULL THEN 1 END) * 100.0 / COUNT(*)) as percentage FROM {table_name}

Question: What is the date range?
SQL: SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM {table_name}

Return ONLY the SQL query, no explanations or markdown.
"""

        base_prompt += f"\n\nTable Schema: {schema_text}\nQuestion: {question}\nBreakdown: {breakdown}\nMeasure:{measure}\n"

        if local_errors:
            base_prompt += "\n\nPREVIOUS ATTEMPTS TO THIS QUESTION: \n"
            for i, err in enumerate(local_errors):
                base_prompt += f"\nAttempt {i + 1}: \nSQL: {err['sql']}\nError: {err['error']}\n"
        if past_learnings:
            base_prompt += "\n\nINFORMATION LEARNED FROM PAST SIMILAR QUESTIONS:\n"
            for i, learning in enumerate(past_learnings):
                base_prompt += f"\nAttempt {i + 1}: Learning: {learning}\n"
        base_prompt += "\n\nGenerate correct SQL:"
        return base_prompt

    async def _generate_reflection(self, question: str, errors: List[Dict]) -> str:
        error_summary = "\n".join([
            f"Attempt {i + 1}: SQL: {e['sql']}\nError: {e['error']}"
            for i, e in enumerate(errors)
        ])
        prompt = f"""Analyze these SQL generation failures and extract key lessons:
Question: {question}

Failures:
{error_summary}

Provide a concise reflection on:
1. what pattern of errors occurred?
2. what is the root cause?
3. what specific lesson should be remembered for future similar qustions?

Reflection:
"""
        response = self.claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


# ============================================== Main Function =============================================
# ==========================================================================================================

question_gen = None


@click.command()
@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
@click.option("--anthropic-api-key", help="Anthropic API key")
def main(port: int, host: str, anthropic_api_key: str) -> int:
    global question_gen

    app = Server("qugen-pipeline")
    question_gen = QuestionGenerator(anthropic_api_key=anthropic_api_key)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="generate_insights",  # NEW TOOL
                description="Complete pipeline: Generate insight cards AND execute SQL to produce natural language insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to CSV file"
                        },
                        "num_iterations": {
                            "type": "integer",
                            "description": "Number of QUGEN iterations",
                            "default": 5
                        },
                        "samples_per_iteration": {
                            "type": "integer",
                            "description": "LLM samples per iteration",
                            "default": 2
                        }
                    },
                    "required": ["file_path"]
                }
            )
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if name == "generate_insights":  # NEW TOOL HANDLER
                file_path = Path(arguments["file_path"])
                num_iterations = arguments.get("num_iterations", 5)
                samples_per_iteration = arguments.get("samples_per_iteration", 2)

                result = await question_gen.generate_insights(
                    file_path=file_path,
                    num_iterations=num_iterations,
                    samples_per_iteration=samples_per_iteration
                )

                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [types.TextContent(type="text", text=json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2))]

    # Run server
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
    uvicorn.run(starlette_app, host=host, port=port)
    return 0


if __name__ == "__main__":
    main()