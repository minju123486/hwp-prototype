from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()
import os
# HyDERetriever 정의 (RAG용, 실제 사용 X)
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from pydantic import Field
from typing import Any

class HyDERetriever(BaseRetriever):
    llm: Any = Field()
    embeddings: Any = Field()
    vectorstore: Any = Field()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
        prompt = f"""
아래의 질문에 대해, 기업명과 사업자 등록번호가 포함된 가상의 사업계획서 요약을 작성하세요.

질문: {query}

예시:
기업명: ㈜예시컴퍼니
사업자 등록번호: 123-45-67890
지원 과제 요약: AI 기반 문서 자동화 솔루션 개발
아이템 핵심 사항: 자연어처리, 문서 파싱, 자동화

작성:
"""
        hypothetical_doc = self.llm.predict(prompt)
        hyde_emb = self.embeddings.embed_query(hypothetical_doc)
        docs = self.vectorstore.similarity_search_by_vector(hyde_emb, k=5)
        return docs

def semantic_chunking(text):
    # 시멘틱 청킹: 의미 단위로 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 필요에 따라 조정
        chunk_overlap=50
    )
    return splitter.create_documents([text])

def build_vectorstore(docs, embeddings):
    # FAISS 벡터스토어 생성
    return FAISS.from_documents(docs, embeddings)

def get_ensemble_retriever(llm, embeddings, vectorstore):
    # HyDE 리트리버
    hyde_retriever = HyDERetriever(llm=llm, embeddings=embeddings, vectorstore=vectorstore)
    # FAISS 기본 리트리버
    faiss_retriever = vectorstore.as_retriever()
    # 앙상블 리트리버 (가중치는 필요에 따라 조정)
    ensemble = EnsembleRetriever(
        retrievers=[hyde_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble

def ParseFirstFile(file_path):
    # 1. HWP 파일에서 텍스트 추출
    loader = HWPLoader(file_path)
    docs = loader.load()
    text = docs[0].page_content

    # 2. 시멘틱 청킹
    chunked_docs = semantic_chunking(text)

    # 3. 임베딩 및 벡터스토어 생성
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = build_vectorstore(chunked_docs, embeddings)

    # 4. LLM 및 앙상블 리트리버 준비
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    retriever = get_ensemble_retriever(llm, embeddings, vectorstore)

    # 5. 예시 쿼리로 리트리버 사용 (실제 사용에 맞게 수정)
    query = "이 문서의 기관명, 담당자명, 연락처, 기업명, 사업자번호, 대표자명, 연락처1, 연락처2, 지원과제명, 아이템, 추천사유 알려줘"
    retrieved_docs = retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # 6. LLM 프롬프트 및 체인 실행 (기존과 동일)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
아래의 텍스트에서 다음 항목을 찾아서 JSON 형태로 반환해 주세요.
- 기관명
- 담당자명
- 연락처
- 기업명
- 사업자번호
- 대표자명
- 연락처1
- 연락처2
- 지원과제명
- 아이템
- 추천사유
텍스트:
{text}

반환 예시:
{{
    "기관명": "...",
    "담당자명": "...",
    "연락처": "...",
    "기업명": "...",
    "사업자번호": "..."
    "대표자명": "..."
    "연락처1": "..."
    "연락처2": "..."
    "지원과제명": "..."
    "아이템": "..."
    "추천사유": "..."
}}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(text=combined_text)

    # 7. 결과 반환 (문자열을 딕셔너리로 변환)
    import json
    try:
        result_dict = json.loads(result)
    except Exception:
        result_dict = {"raw_output": result}

    return result_dict

def main(path):
    import sys

    # 파일 경로를 명령행 인자로 받거나, 직접 입력받을 수 있음

    result = ParseFirstFile(path)
    print("전체 결과:ㅇㅁㄴㅁㄴㅇㅁㄴㅇㅇㄴㅁ", result)

    # 각 항목을 변수로 추출
    company_name = result.get("기관명")
    business_number = result.get("담당자명")
    project_summary = result.get("연락처")
    item_core = result.get("기업명")

    # 각 변수 출력
    print("기업명:", company_name)
    print("사업자 등록번호:", business_number)
    print("지원 과제 요약:", project_summary)
    print("아이템 핵심 사항:", item_core)
    return result