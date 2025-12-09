import os
import hashlib
from typing import Tuple, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from Backend.config import settings
from langchain.embeddings import SentenceTransformerEmbeddings

class QueryPreprocessor:
    """Handle query preprocessing using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def clean_query(self, user_query: str) -> str:
        """Clean and enhance user query using LLM"""
        
        if not user_query or len(user_query.strip()) < 2:
            return user_query
        
        prompt = f"""Clean and improve this query for an Indian law search system:

Rules:
1. Fix spelling mistakes (e.g., "secton" → "section", "waht" → "what")
2. Remove filler words (e.g., "um", "like", "you know")
3. Keep it concise but complete (under 25 words)
4. Preserve all legal terms, section numbers, article numbers
5. If query is already clear, return it as-is
6. Don't change the intent or add new information

Examples:
Input: "waht is secton 420 ipc?"
Output: What is Section 420 IPC?

Input: "um like tell me abt cheeting laws"
Output: Tell me about cheating laws

Input: "What is Article 21?"
Output: What is Article 21?

User query: {user_query}

Cleaned query (just the query, nothing else):"""
        
        try:
            response = self.llm.invoke(prompt)
            cleaned = response.content.strip().strip('"\'')
            
            # Validation: don't use if too different
            if len(cleaned) > len(user_query) * 2.5:
                print(f"⚠️ LLM output too long, using original")
                return user_query.strip()
            
            if not cleaned or len(cleaned) < 2:
                print(f"⚠️ LLM output too short, using original")
                return user_query.strip()
            
            # Log if changed
            if cleaned.lower() != user_query.lower().strip():
                print(f"🔧 Query preprocessed:")
                print(f"   Original: '{user_query}'")
                print(f"   Cleaned:  '{cleaned}'")
            
            return cleaned
            
        except Exception as e:
            print(f"⚠️ Query preprocessing failed: {e}")
            return user_query.strip()


class LangChainRAG:
    def __init__(self):
        """Initialize LangChain RAG components"""
        
        print("🚀 Initializing LangChain RAG System...")
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        print(f"✓ LLM initialized: {settings.LLM_MODEL}")
        
        # Initialize query preprocessor
        self.query_preprocessor = QueryPreprocessor(self.llm)
        print(f"✓ Query preprocessor initialized")
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        print(f"✓ Embeddings initialized: {settings.EMBEDDING_MODEL}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        print(f"✓ Text splitter initialized (chunk_size={settings.CHUNK_SIZE})")
        
        # Track processed documents to avoid duplicates
        self.processed_files = self._load_processed_files_registry()
        
        # Initialize or load vector store
        self.vectorstore = self._initialize_vectorstore()
        
        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        print("✅ LangChain RAG System Ready!\n")
    
    def _load_processed_files_registry(self) -> Dict[str, str]:
        """Load registry of processed files to avoid duplicates"""
        registry_path = os.path.join(settings.CHROMA_DB_PATH, "processed_files.txt")
        processed = {}
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    for line in f:
                        if '|' in line:
                            filepath, file_hash = line.strip().split('|')
                            processed[filepath] = file_hash
                print(f"✓ Loaded registry: {len(processed)} files tracked")
            except Exception as e:
                print(f"⚠️ Error loading registry: {e}")
        
        return processed
    
    def _save_processed_files_registry(self):
        """Save registry of processed files"""
        registry_path = os.path.join(settings.CHROMA_DB_PATH, "processed_files.txt")
        
        try:
            os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
            with open(registry_path, 'w') as f:
                for filepath, file_hash in self.processed_files.items():
                    f.write(f"{filepath}|{file_hash}\n")
        except Exception as e:
            print(f"⚠️ Error saving registry: {e}")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"⚠️ Error hashing file {filepath}: {e}")
            return ""
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            if os.path.exists(settings.CHROMA_DB_PATH):
                # Load existing vector store
                vectorstore = Chroma(
                    persist_directory=settings.CHROMA_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name=settings.COLLECTION_NAME
                )
                count = vectorstore._collection.count()
                print(f"✓ Loaded existing vector store ({count} documents)")
            else:
                # Create new vector store
                vectorstore = Chroma(
                    persist_directory=settings.CHROMA_DB_PATH,
                    embedding_function=self.embeddings,
                    collection_name=settings.COLLECTION_NAME
                )
                print("✓ Created new vector store")
            
            return vectorstore
            
        except Exception as e:
            print(f"⚠️ Error initializing vector store: {e}")
            return Chroma(
                persist_directory=settings.CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name=settings.COLLECTION_NAME
            )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for Indian law queries"""
        
        template = """You are an expert assistant on Indian law, trained specifically on the Indian Constitution and legal documents.

Use the context below to answer the user's question accurately and concisely.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Cite specific Articles, Sections, or Parts when mentioned
3. If the context doesn't fully answer the question, state what information IS available
4. If the context is insufficient, clearly say: "The provided documents do not contain sufficient information to answer this question."
5. Never fabricate or assume information not in the context
6. Be clear, concise, and structured in your response

CONTEXT FROM LEGAL DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def auto_ingest_documents(self) -> Tuple[str, int, int, int]:
        """
        Automatically ingest all PDFs from DATA_PATH
        Skips already processed files, updates modified files
        
        Returns:
            Tuple of (message, new_files, updated_files, skipped_files)
        """
        
        try:
            print("\n📥 Auto-ingestion: Scanning for documents...")
            
            # Find all PDF files
            pdf_files = []
            for root, dirs, files in os.walk(settings.DATA_PATH):
                for file in files:
                    if file.endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            if not pdf_files:
                print("⚠️ No PDF files found")
                return "No PDF files found in data directory", 0, 0, 0
            
            print(f"✓ Found {len(pdf_files)} PDF files")
            
            new_count = 0
            updated_count = 0
            skipped_count = 0
            total_chunks = 0
            
            for pdf_path in pdf_files:
                file_hash = self._get_file_hash(pdf_path)
                relative_path = os.path.relpath(pdf_path, settings.DATA_PATH)
                
                # Check if file is already processed
                if relative_path in self.processed_files:
                    if self.processed_files[relative_path] == file_hash:
                        # File unchanged, skip
                        print(f"⏭️  Skipping (unchanged): {relative_path}")
                        skipped_count += 1
                        continue
                    else:
                        # File modified, remove old chunks and re-process
                        print(f"🔄 Updating (modified): {relative_path}")
                        self._remove_document_by_source(relative_path)
                        updated_count += 1
                else:
                    # New file
                    print(f"➕ Adding (new): {relative_path}")
                    new_count += 1
                
                # Load and process document
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    # Update metadata with relative path
                    for doc in documents:
                        doc.metadata['source'] = relative_path
                    
                    # Split into chunks
                    splits = self.text_splitter.split_documents(documents)
                    
                    # Add to vector store
                    self.vectorstore.add_documents(splits)
                    total_chunks += len(splits)
                    
                    # Update registry
                    self.processed_files[relative_path] = file_hash
                    
                    print(f"   ✓ Processed: {len(splits)} chunks")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {relative_path}: {e}")
                    continue
            
            # Persist vector store and registry
            self.vectorstore.persist()
            self._save_processed_files_registry()
            
            message = f"Auto-ingestion complete: {new_count} new, {updated_count} updated, {skipped_count} skipped ({total_chunks} total chunks)"
            print(f"✅ {message}\n")
            
            return message, new_count, updated_count, skipped_count
            
        except Exception as e:
            error_msg = f"Error during auto-ingestion: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, 0, 0, 0
    
    def _remove_document_by_source(self, source: str):
        """Remove all chunks from a specific source document"""
        try:
            # Get all document IDs with this source
            results = self.vectorstore._collection.get(
                where={"source": source}
            )
            
            if results and results['ids']:
                self.vectorstore._collection.delete(ids=results['ids'])
                print(f"   🗑️  Removed {len(results['ids'])} old chunks")
        except Exception as e:
            print(f"   ⚠️ Error removing old chunks: {e}")
    
    def query(self, user_query: str) -> Tuple[str, List[Dict], bool]:
        """
        Query the RAG system with automatic query preprocessing
        
        Args:
            user_query: User's question (may contain typos/errors)
        
        Returns:
            Tuple of (answer, sources, is_relevant)
        """
        
        print(f"\n{'='*70}")
        print(f"Original Query: {user_query}")
        
        # STEP 1: Preprocess query using LLM
        cleaned_query = self.query_preprocessor.clean_query(user_query)
        print(f"Processed Query: {cleaned_query}")
        print(f"{'='*70}")
        
        try:
            # Check if vector store has documents
            if self.vectorstore._collection.count() == 0:
                return (
                    "No documents have been loaded yet. Please add PDF files to the data directory and restart the server.",
                    [],
                    False
                )
            
            # STEP 2: Create retriever and search with cleaned query
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.TOP_K_RESULTS}
            )
            
            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(cleaned_query)
            
            if not relevant_docs:
                print("❌ No relevant documents found")
                return self._out_of_scope_response(), [], False
            
            # Check relevance
            is_relevant = self._check_relevance(cleaned_query, relevant_docs)
            
            if not is_relevant:
                print("❌ Query not relevant to document content")
                return self._out_of_scope_response(), [], False
            
            print(f"✓ Found {len(relevant_docs)} relevant documents")
            
            # STEP 3: Create QA chain and generate answer
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            
            print("🤖 Generating answer...")
            result = qa_chain({"query": cleaned_query})
            
            # Extract answer and sources
            answer = result["result"]
            source_documents = result["source_documents"]
            
            # Add note if query was modified
            if cleaned_query.lower() != user_query.lower().strip():
                answer += f"\n\n*(Note: I interpreted your query as: \"{cleaned_query}\")*"
            
            # Format sources
            sources = []
            for doc in source_documents:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", None)
                }
                sources.append(source_info)
            
            print(f"✅ Answer generated with {len(sources)} sources\n")
            
            return answer, sources, True
            
        except Exception as e:
            error_msg = f"Error during query: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, [], False
    
    def _check_relevance(self, query: str, documents: List[Document]) -> bool:
        """Check if retrieved documents are relevant to the query"""
        
        query_words = set(query.lower().split())
        stop_words = {'what', 'is', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        query_keywords = query_words - stop_words
        
        # Check for obvious non-legal queries
        non_legal_keywords = {
            'recipe', 'cook', 'bake', 'restaurant', 'food', 'weather', 
            'movie', 'song', 'game', 'sport', 'travel'
        }
        
        if query_keywords.intersection(non_legal_keywords):
            return False
        
        # Check if any document has reasonable keyword overlap
        for doc in documents[:3]:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_keywords.intersection(doc_words))
            
            if overlap >= len(query_keywords) * 0.3:
                return True
        
        return len(documents) > 0
    
    def _out_of_scope_response(self) -> str:
        """Return out of scope message"""
        return (
            "I apologize, but this question appears to be outside the scope of "
            "Indian law that I have knowledge of. Please ask questions related to "
            "Indian legal matters, acts, sections, or constitutional provisions."
        )
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            count = self.vectorstore._collection.count()
            
            # Get sample to check sources
            sample = self.vectorstore._collection.get(limit=100)
            sources = set()
            
            if sample and sample.get('metadatas'):
                for metadata in sample['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
            
            return {
                "total_documents": count,
                "unique_sources": len(sources),
                "sources": list(sources),
                "processed_files": len(self.processed_files),
                "collection_name": settings.COLLECTION_NAME,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.LLM_MODEL
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_documents": 0, "unique_sources": 0, "sources": [], "processed_files": 0}

# Global instance
langchain_rag = None

def get_langchain_rag() -> LangChainRAG:
    global langchain_rag
    if langchain_rag is None:
        langchain_rag = LangChainRAG()
    return langchain_rag