#!/usr/bin/env python3
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict, Coroutine

import anthropic
import click
import mcp.types as types
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from pdf2image import convert_from_path
from qdrant_client import QdrantClient, models
import asyncpg
import aiosqlite

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Rag_server:
    def __init__(self, use_memory: bool = False, anthropic_api_key: Optional[str] = None):
        # Qdrant setup
        if use_memory:
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client")
        else:
            try:
                self.qdrant_client = QdrantClient("http://localhost:6333")
                logger.info("Connected to local Qdrant at localhost:6333")
            except Exception as e:
                logger.error(f"Failed to connect to local Qdrant: {e}")
                raise
        self.qdrant_client.set_model("BAAI/bge-small-en-v1.5")
        self.table_collection = "database_tables"

        self.colpali_model = None
        self.colpali_processor = None
        self.collection_name = "rag_server"

        # Claude Vision setup
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            logger.error(
                "No Anthropic API key found!")
            raise ValueError("ANTHROPIC_API_KEY not found ")

        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        logger.info("Claude client initialized successfully")

        self.pdf_images = {}

        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.set_per_process_memory_fraction(0.7)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("GPU not available, using CPU")

    def _load_pdf_images(self, pdf_path: str, doc_id: str) -> Dict[int, Image.Image]:
        if doc_id in self.pdf_images:
            logger.debug(f"Images for {doc_id} already in memory cache")
            return self.pdf_images[doc_id]

        logger.info(f"Loading PDF: {pdf_path}")
        try:
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return {}

            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Store in memory cache
            self.pdf_images[doc_id] = {
                i + 1: img for i, img in enumerate(images)
            }

            logger.info(f"Successfully loaded {len(images)} images for doc_id: {doc_id}")
            return self.pdf_images[doc_id]

        except Exception as e:
            logger.error(f"Failed to load images from {pdf_path}: {e}")
            return {}

    async def initialize_colpali(self, model_name: str = "vidore/colpali-v1.3"):
        try:
            logger.info(f"Loading ColPali model: {model_name}")

            self.colpali_model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()

            self.colpali_processor = ColPaliProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            logger.info("ColPali model loaded successfully")

            await self._setup_collection()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to initialize ColPali: {e}")
            if "out of memory" in str(e).lower():
                logger.info("GPU out of memory, falling back to CPU")
                self.device = "cpu"
                self.colpali_model = ColPali.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).eval()
                self.colpali_processor = ColPaliProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                await self._setup_collection()
            else:
                raise

    async def _setup_collection(self):
        try:
            if self.qdrant_client.collection_exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def _embed_images_colpali(self, images: List[Image.Image]):
        batch_size = 1 if self.device == "cpu" else min(2, len(images))
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_features = self.colpali_processor.process_images(batch_images)
                batch_features = {k: v.to(self.colpali_model.device) for k, v in batch_features.items()}

                embeddings = self.colpali_model(**batch_features)
                embeddings_np = embeddings.cpu().float().numpy()
                all_embeddings.extend(embeddings_np)

                del batch_features, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return all_embeddings

    def _embed_query_colpali(self, query: str):
        with torch.no_grad():
            batch_queries = self.colpali_processor.process_queries([query])
            batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}
            query_embedding = self.colpali_model(**batch_queries)
            return query_embedding.cpu().float().numpy()

    def _image_to_base64(self, image: Image.Image, max_size: tuple = (1024, 1024)) -> str:
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')

    async def _extract_with_claude(self, images: List[Image.Image], query: str) -> Dict:
        try:
            image_contents = []
            for i, img in enumerate(images):
                base64_image = self._image_to_base64(img)
                image_contents.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image
                    }
                })

            messages = [
                {
                    "role": "user",
                    "content": image_contents + [
                        {
                            "type": "text",
                            "text": f"""Based on the PDF pages shown, please analyze the database schema and answer this query: {query}

                        Provide a comprehensive summary that covers:
                        1. Key tables and their purposes
                        2. Important columns and relationships
                        3. How the data can be used to answer the query

                        Respond with just a clear, detailed paragraph summary """
                        }
                    ]
                }
            ]
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=messages
            )

            response_text = response.content[0].text.strip()

            return {
                "success": True,
                "summary": response_text,
                "raw_response": response_text
            }

        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            return {
                "success": False,
                "error": f"Claude extraction failed: {str(e)}"
            }

    async def index_pdf(self, pdf_path: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        try:
            if doc_id is None:
                doc_id = Path(pdf_path).stem

            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(pdf_path)

            self.pdf_images[doc_id] = {
                i + 1: img for i, img in enumerate(images)
            }

            logger.info("Generating ColPali embeddings...")
            embeddings = self._embed_images_colpali(images)

            points = []
            for i, (embedding, image) in enumerate(zip(embeddings, images)):
                payload = {
                    "pdf_path": pdf_path,
                    "doc_id": doc_id,
                    "page_number": i + 1,
                    "total_pages": len(images),
                    **(metadata or {})
                }

                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                ))

            logger.info("Uploading to Qdrant")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Successfully indexed {len(images)} pages from {pdf_path}")
            return f"Indexed {len(images)} pages from {Path(pdf_path).name} (doc_id: {doc_id})"

        except Exception as e:
            logger.error(f"Failed to index PDF: {e}")
            raise

    async def search_file(self, query: str, limit: int = 5, extract_from_top: int = 3) -> Dict:

        try:
            logger.info(f"Searching for: {query}")
            query_embedding = self._embed_query_colpali(query)

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=np.mean(query_embedding[0], axis=0).tolist(),
                limit=limit,
                with_payload=True
            )

            if not search_result:
                return {"error": "No relevant pages found"}

            top_results = search_result[:extract_from_top]
            grouped_images = []
            page_info = []

            for result in top_results:
                doc_id = result.payload.get("doc_id")
                page_num = result.payload.get("page_number")
                pdf_path = result.payload.get("pdf_path")

                # Load images if not in cache
                if doc_id not in self.pdf_images:
                    logger.info(f"Images not in cache for {doc_id}, loading from {pdf_path}")
                    doc_images = self._load_pdf_images(pdf_path, doc_id)
                else:
                    doc_images = self.pdf_images[doc_id]

                # Get the specific page image
                if page_num in doc_images:
                    grouped_images.append(doc_images[page_num])
                    page_info.append({
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "score": result.score,
                        "pdf_path": pdf_path
                    })
                else:
                    logger.warning(f"Page {page_num} not found for doc_id={doc_id}")

            if not grouped_images:
                return {
                    "error": "Images not available for extraction",
                    "search_results": [
                        {

                            "score": r.score,
                            "page_number": r.payload.get("page_number"),
                            "doc_id": r.payload.get("doc_id"),
                            "pdf_path": r.payload.get("pdf_path")
                        } for r in search_result
                    ]
                }

            logger.info(f"Extracting information with Claude from {len(grouped_images)} pages")

            claude_result = await self._extract_with_claude(grouped_images, query)
            if claude_result.get("success") and "summary" in claude_result:
                summary = claude_result["summary"]
                logger.info("Summary extracted successfully")
                return {"summary": summary}
            else:
                logger.warning("Failed to extract summary")
                return {
                    "error": "Failed to extract summary",
                    "claude_success": claude_result.get("success", False),
                    "claude_error": claude_result.get("error", "Unknown error"),
                    "search_results": page_info,
                    "total_pages_searched": len(search_result)
                }

        except Exception as e:
            logger.error(f"Failed in search and extract: {e}")
            return {"error": str(e)}

    async def database_connection(self, query: str, url: str) -> str | list[Any]:
        try:
            db_type = self._get_database_type(url)
            logger.info(f"Connecting to {db_type} database")

            tables = await self.get_table_names(url)

            if not tables["success"]:
                return f"Failed to get tables: {tables['error']}"

            documents = []
            metadata = []
            for table_name in tables["table_names"]:
                document_text = f"Database table: {table_name}"
                documents.append(document_text)

                metadata.append({
                    "table_name": table_name,
                    "database_url": url,
                    "database_type": db_type,
                    "type": "database_table"
                })

            if self.qdrant_client.collection_exists(self.table_collection):
                try:
                    self.qdrant_client.delete(
                        collection_name=self.table_collection,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="database_url",
                                        match=models.MatchValue(value=url)
                                    )
                                ]
                            )
                        )
                    )
                    logger.info("Cleared existing table data for this URL")
                except Exception as e:
                    logger.warning(f"Could not clear existing data: {e}")

            self.qdrant_client.add(
                collection_name=self.table_collection,
                documents=documents,
                metadata=metadata
            )

            results = self.qdrant_client.query(
                collection_name=self.table_collection,
                query_text=query,
                limit=1,
            )

            table_names = [
                result.metadata["table_name"]
                for result in results
            ]

            return table_names

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return f"Error: {str(e)}"

    def _get_database_type(self, url: str) -> str:
        if url.startswith('postgresql://') or url.startswith('postgres://'):
            return 'postgresql'
        elif url.startswith('sqlite://') or url.endswith('.db') or url.endswith('.sqlite') or url.endswith('.sqlite3'):
            return 'sqlite'
        else:
            raise ValueError(f"Unsupported database type in URL: {url}")

    async def find_table(self, query: str, data_source: str) -> Dict[str, Any]:
        try:
            db_type = self._get_database_type(data_source)
            logger.info(f"Searching for table matching query: '{query}' in {db_type} database")

            table_names = await self.database_connection(query, data_source)

            if isinstance(table_names, str):
                return {
                    "success": False,
                    "error": table_names
                }

            if not table_names:
                return {
                    "success": False,
                    "error": "No matching tables found"
                }

            target_table = table_names[0]

            schema_info = await self.get_table_schema(target_table, data_source)

            if not schema_info["success"]:
                return {
                    "success": False,
                    "error": f"Failed to get schema for table '{target_table}': {schema_info['error']}"
                }

            db_info = self.parse_database_url(data_source)

            return {
                "success": True,
                "database_type": db_type,
                "table_name": target_table,
                "table_path": {
                    "database": db_info["database"],
                    "schema": db_info.get("schema", "public" if db_type == "postgresql" else "main"),
                    "table": target_table,
                    "full_path": f"{db_info['database']}.{db_info.get('schema', 'public' if db_type == 'postgresql' else 'main')}.{target_table}"
                },
                "connection_info": db_info,
                "schema": schema_info["schema"],
                "column_count": len(schema_info["schema"]),
                "query_match": query
            }

        except Exception as e:
            logger.error(f"Error in find_table: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_table_schema(self, table_name: str, url: str) -> Dict[str, Any]:
        try:
            db_type = self._get_database_type(url)

            if db_type == "postgresql":
                return await self._get_postgresql_schema(table_name, url)
            elif db_type == "sqlite":
                return await self._get_sqlite_schema(table_name, url)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported database type: {db_type}"
                }

        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_postgresql_schema(self, table_name: str, url: str) -> Dict[str, Any]:
        query = """
                SELECT column_name,
                       data_type,
                       is_nullable,
                       column_default,
                       character_maximum_length,
                       numeric_precision,
                       numeric_scale,
                       ordinal_position
                FROM information_schema.columns
                WHERE table_name = $1
                  AND table_schema = 'public'
                ORDER BY ordinal_position;
                """

        try:
            conn = await asyncpg.connect(url)
            results = await conn.fetch(query, table_name)
            await conn.close()

            if not results:
                return {
                    "success": False,
                    "error": f"Table '{table_name}' not found or no columns available"
                }

            schema = []
            for row in results:
                column_info = {
                    "column_name": row['column_name'],
                    "data_type": row['data_type'],
                    "is_nullable": row['is_nullable'] == 'YES',
                    "column_default": row['column_default'],
                    "ordinal_position": row['ordinal_position']
                }

                if row['character_maximum_length']:
                    column_info["max_length"] = row['character_maximum_length']
                if row['numeric_precision']:
                    column_info["precision"] = row['numeric_precision']
                if row['numeric_scale']:
                    column_info["scale"] = row['numeric_scale']

                schema.append(column_info)

            return {
                "success": True,
                "schema": schema,
                "table_name": table_name
            }

        except Exception as e:
            logger.error(f"Error getting PostgreSQL schema: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_sqlite_schema(self, table_name: str, url: str) -> Dict[str, Any]:
        try:
            if url.startswith('sqlite://'):
                db_path = url[9:]
            else:
                db_path = url

            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(f"PRAGMA table_info({table_name})")
                results = await cursor.fetchall()

                if not results:
                    return {
                        "success": False,
                        "error": f"Table '{table_name}' not found or no columns available"
                    }

                schema = []
                for row in results:
                    column_info = {
                        "column_name": row[1],
                        "data_type": row[2],
                        "is_nullable": not bool(row[3]),
                        "column_default": row[4],
                        "ordinal_position": row[0],
                        "is_primary_key": bool(row[5])
                    }
                    schema.append(column_info)

                return {
                    "success": True,
                    "schema": schema,
                    "table_name": table_name
                }

        except Exception as e:
            logger.error(f"Error getting SQLite schema: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def parse_database_url(self, url: str) -> Dict[str, Any]:
        try:
            db_type = self._get_database_type(url)

            if db_type == "postgresql":
                return self._parse_postgresql_url(url)
            elif db_type == "sqlite":
                return self._parse_sqlite_url(url)
            else:
                return {"database_type": "unknown", "error": "Unsupported database type"}

        except Exception as e:
            logger.error(f"Error parsing database URL: {e}")
            return {"database_type": "unknown", "error": str(e)}

    def _parse_postgresql_url(self, url: str) -> Dict[str, Any]:
        try:
            import re
            pattern = r'postgresql://(?:([^:]+):([^@]+)@)?([^:]+):(\d+)/(.+)'
            match = re.match(pattern, url)

            if match:
                user, password, host, port, database = match.groups()
                return {
                    "database_type": "postgresql",
                    "user": user,
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "schema": "public"
                }
            else:
                parts = url.split('@')
                if len(parts) > 1:
                    host_db = parts[1]
                    host_port, database = host_db.split('/')
                    host, port = host_port.split(':')
                    return {
                        "database_type": "postgresql",
                        "host": host,
                        "port": int(port),
                        "database": database,
                        "schema": "public"
                    }
                else:
                    return {
                        "database_type": "postgresql",
                        "host": "unknown",
                        "port": 5432,
                        "database": "unknown",
                        "schema": "public"
                    }

        except Exception as e:
            logger.error(f"Error parsing PostgreSQL URL: {e}")
            return {
                "database_type": "postgresql",
                "host": "unknown",
                "port": 5432,
                "database": "unknown",
                "schema": "public",
                "error": str(e)
            }

    def _parse_sqlite_url(self, url: str) -> Dict[str, Any]:
        try:
            if url.startswith('sqlite://'):
                db_path = url[9:]
            else:
                db_path = url

            return {
                "database_type": "sqlite",
                "database": Path(db_path).name,
                "path": db_path,
                "schema": "main"
            }

        except Exception as e:
            logger.error(f"Error parsing SQLite URL: {e}")
            return {
                "database_type": "sqlite",
                "database": "unknown",
                "path": url,
                "schema": "main",
                "error": str(e)
            }

    async def get_table_names(self, url: str) -> Dict[str, Any]:
        try:
            db_type = self._get_database_type(url)

            if db_type == "postgresql":
                return await self._get_postgresql_tables(url)
            elif db_type == "sqlite":
                return await self._get_sqlite_tables(url)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported database type: {db_type}"
                }

        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_postgresql_tables(self, url: str) -> Dict[str, Any]:
        query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """

        try:
            conn = await asyncpg.connect(url)
            results = await conn.fetch(query)
            await conn.close()

            table_names = [row['table_name'] for row in results]

            return {
                "success": True,
                "table_names": table_names,
                "table_count": len(table_names)
            }

        except Exception as e:
            logger.error(f"Error getting PostgreSQL tables: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_sqlite_tables(self, url: str) -> Dict[str, Any]:
        try:
            if url.startswith('sqlite://'):
                db_path = url[9:]
            else:
                db_path = url

            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                results = await cursor.fetchall()

                table_names = [row[0] for row in results]

                return {
                    "success": True,
                    "table_names": table_names,
                    "table_count": len(table_names)
                }

        except Exception as e:
            logger.error(f"Error getting SQLite tables: {e}")
            return {
                "success": False,
                "error": str(e)
            }


rag_server = None


@click.command()
@click.option("--use-memory", is_flag=True, help="Use in-memory Qdrant")
@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
@click.option("--anthropic-api-key", help="Anthropic API key ")
def main(use_memory: bool, port: int, host: str, anthropic_api_key: str) -> int:
    global rag_server

    app = Server("rag")
    rag_server = Rag_server(use_memory=use_memory, anthropic_api_key=anthropic_api_key)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="index_pdf",
                description="Index a PDF using ColPali",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {"type": "string", "description": "Path to the PDF file"},
                    },
                    "required": ["pdf_path"]
                }
            ),
            types.Tool(
                name="search_file",
                description=" Use ColPali to find relevant pages",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search query"},
                        "extract_page": {"type": "integer", "description": "Number of top results",
                                         "default": 3}
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="search_data_sources",
                description=" Search relevant data in database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Table name"},
                        "data_source": {"type": "string",
                                        "description": "data source url"}
                    },
                    "required": ["query", "data_source"]
                }
            ),
            types.Tool(
                name="find_table",
                description=" find the path and schema of the table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Table name"},
                        "data_source": {"type": "string",
                                        "description": "data source url"}
                    },
                    "required": ["query", "data_source"]
                }
            )

        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:

            if name == "index_pdf":
                if rag_server.colpali_model is None:
                    await rag_server.initialize_colpali()
                pdf_path = arguments["pdf_path"]
                result = await rag_server.index_pdf(pdf_path)
                return [types.TextContent(type="text", text=result)]

            elif name == "search_file":
                if rag_server.colpali_model is None:
                    await rag_server.initialize_colpali()
                query = arguments["query"]
                limit = arguments.get("limit", 5)
                extract_from_top = arguments.get("extract_from_top", 3)
                result = await rag_server.search_file(query, limit, extract_from_top)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            elif name == "search_data_sources":
                query = arguments["query"]
                url = arguments["data_source"]
                result = await rag_server.database_connection(query, url)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "find_table":
                query = arguments["query"]
                url = arguments["data_source"]
                result = await rag_server.find_table(query, url)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]



            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

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