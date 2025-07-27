import pytesseract
from pdf2image import convert_from_path
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

class BanglaRAG():
    
    def __init__(self, the_pdf_path, data_path='data/', llm="ollama"):
        print("Loading LLM")
        self.llm = OllamaLLM(
                    model="kaizu/bn_chat",
                    temperature=0.45, #less creativity, more factual context based answered
                    system="""Always respond in Bengali. Use the context provided. 
                           If unsure, say "আমি জানি না\""""
                )
        # self.llm = self.load_ollama_model if llm == "ollama" else load_hg_model if llm == "hg" else lambda: print("invalid model loaded")
        print("Initializing RAG Pipeline")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}  # or "cuda" for GPU
        )

        self.pdf_path = the_pdf_path
        self.data_path = data_path
        
        print("Feeding PDF into OCR")
        unclean_text = self.ocr_with_tesseract(self.pdf_path)
        print("Text retrieved")
        self.clean_text = self.precise_bengali_cleaner(unclean_text)
        self.clean_text = self.ultra_precise_cleaner(self.clean_text)
        self.final_text_file_path = self.data_path + "final_cleaned.txt"
        self.save_text_to_file(self.clean_text, self.final_text_file_path) #probably unnecesary
        
        self.text_chunkz_data = self.text_splitter(self.final_text_file_path)
        self.vector_db = Chroma.from_documents(
            documents=self.text_chunkz_data,
            embedding=self.embeddings,
            persist_directory="./bengali_chroma_db"  # Local storage
        )
        print("Vector database populated, you may now submit your queries")
        
    # def load_ollama_model(self, model_name="kaizu/bn_chat"):
    #     llm = OllamaLLM(
    #         model="kaizu/bn_chat",
    #         temperature=0.3,  # Control creativity (0-1)
    #         system="""Always respond in Bengali. Use the context provided. 
    #                If unsure, say "আমি জানি না\""""
    #     )

    #     return llm

    # def load_hg_model(self,):
    #     pass
        
    def ocr_with_tesseract(self, pdf_path):
        images = convert_from_path(pdf_path, dpi=300)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='ben') + "\n"
        return text
    
    def precise_bengali_cleaner(self, text: str) -> str:
        text = re.sub(r'\x0c\[[^\]]+\]', '', text)
        text = re.sub(r'(?<!\S)\d[\d ১২৩৪৫৬৭৮৯০?]+\b', '', text)
        text = re.sub(r'[€£][\d]+', '', text)    
        text = text.replace('\x0c', '').strip()
        return text
    
    def ultra_precise_cleaner(self, text: str) -> str:
        text = re.sub(r'^\[লুল\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'জআললাইন ব্যাচ”\n?', '', text)
        text = re.sub(r'^\?$\n', '', text, flags=re.MULTILINE)
        
        return text

    def save_text_to_file(self, text: str, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def text_splitter(self, txt_path : str):
        with open(txt_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
        
        bengali_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=1000,  
            chunk_overlap=200,  
            length_function=len,
            is_separator_regex=False
        )
        
        # Split the documents
        text_chunks = bengali_splitter.create_documents([cleaned_text])
        return text_chunks

    def rag_pipeline(self, query):
        print("Query received, generating answer")
        docs = self.vector_db.similarity_search(query, k=2)  # Top 2 chunks
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""
                নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন:
                {context}
            
                প্রশ্ন: {query}
                উত্তর: 
                """
        return self.llm.invoke(prompt)
    

if __name__ == "__main__":
    pdf_path = "book/HSC26-Bangla1st-Paper.pdf"
    data_path = "data/"

    rag_model = BanglaRAG(the_pdf_path=pdf_path)
    rag_model.rag_pipeline("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
