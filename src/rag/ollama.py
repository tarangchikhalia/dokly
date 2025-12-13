from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.rag.vector_store import VectorStore
from src.utils.config import settings


class OllamaGenerator:
    def __init__(self):
        self.llm_model = settings.LLM_MODEL
        self.llm_url = settings.LLM_URL

        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=self.llm_model,
            base_url=self.llm_url
        )

        # Initialize vector store for retrieval
        self.vector_store = VectorStore()

        # Create retriever that gets top 3 chunks
        self.retriever = self.vector_store.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        # Define the RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use the following context to answer the question.
If you cannot answer the question based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Build the RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def generate(self, prompt: str) -> str:
        """Generate a response using RAG.

        Args:
            prompt: The user's question/prompt

        Returns:
            Generated response from the LLM based on retrieved context
        """
        try:
            # Invoke the RAG chain
            response = self.rag_chain.invoke(prompt)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
