from langchain_teddynote.document_loaders import HWPLoader
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import time
import os
import hashlib
load_dotenv()

# HyDERetriever 정의 (RAG용, 실제 사용 X)
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from pydantic import Field
from typing import Any
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

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
        # ChatOpenAI는 .content가 있지만, Ollama는 문자열을 직접 반환
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            hypothetical_doc = response.content
        else:
            hypothetical_doc = response
            
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

def get_vectorstore_path(file_path):
    """파일 경로를 기반으로 벡터스토어 저장 경로 생성"""
    # 파일 경로의 해시값 생성
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    
    # 벡터스토어 저장 디렉토리 구조
    vectorstore_dir = "vectorstores"
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    
    # 파일명에서 확장자 제거하여 폴더명 생성
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # 한글 및 특수문자 제거, 영문/숫자만 허용
    safe_name = "".join(c for c in file_name if c.isascii() and c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')  # 공백을 언더스코어로 변경
    safe_name = safe_name[:20]  # 길이 제한을 더 짧게
    
    # 최종 경로: vectorstores/파일명_해시/
    final_path = os.path.join(vectorstore_dir, f"{safe_name}_{file_hash[:8]}")
    return final_path

def load_or_create_vectorstore(chunked_docs, embeddings, file_path):
    """벡터스토어를 로드하거나 새로 생성"""
    vectorstore_path = get_vectorstore_path(file_path)
    
    # file_hash 생성 추가
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    
    # 벡터스토어가 이미 존재하는지 확인
    if os.path.exists(vectorstore_path):
        print(f"[{time.strftime('%H:%M:%S')}]   - 기존 벡터스토어 발견: {os.path.basename(vectorstore_path)}")
        try:
            # allow_dangerous_deserialization=True 추가
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print(f"[{time.strftime('%H:%M:%S')}]   - 기존 벡터스토어 로딩 완료")
            return vectorstore
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}]   - 기존 벡터스토어 로딩 실패: {e}")
            print(f"[{time.strftime('%H:%M:%S')}]   - 새로 생성합니다...")
            # 실패한 폴더 삭제
            try:
                import shutil
                shutil.rmtree(vectorstore_path)
                print(f"[{time.strftime('%H:%M:%S')}]   - 손상된 벡터스토어 폴더 삭제 완료")
            except:
                pass
    
    # 새로 생성
    print(f"[{time.strftime('%H:%M:%S')}]   - 새 벡터스토어 생성 중...")
    vectorstore = build_vectorstore(chunked_docs, embeddings)
    
    # 로컬에 저장
    try:
        vectorstore.save_local(vectorstore_path)
        print(f"[{time.strftime('%H:%M:%S')}]   - 벡터스토어 저장 완료: {os.path.basename(vectorstore_path)}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   - 벡터스토어 저장 실패: {e}")
        # 저장 실패 시 임시 경로로 저장 시도
        try:
            temp_path = os.path.join("vectorstores", f"temp_{file_hash[:8]}")
            vectorstore.save_local(temp_path)
            print(f"[{time.strftime('%H:%M:%S')}]   - 임시 경로로 벡터스토어 저장 완료: {os.path.basename(temp_path)}")
        except Exception as e2:
            print(f"[{time.strftime('%H:%M:%S')}]   - 임시 저장도 실패: {e2}")
    
    return vectorstore

def cleanup_old_vectorstores(max_age_days=30):
    """오래된 벡터스토어 정리 (선택사항)"""
    import shutil
    from datetime import datetime, timedelta
    
    vectorstore_dir = "vectorstores"
    if not os.path.exists(vectorstore_dir):
        return
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(days=max_age_days)
    
    for item in os.listdir(vectorstore_dir):
        item_path = os.path.join(vectorstore_dir, item)
        if os.path.isdir(item_path):
            # 폴더 생성 시간 확인
            try:
                creation_time = datetime.fromtimestamp(os.path.getctime(item_path))
                if creation_time < cutoff_time:
                    shutil.rmtree(item_path)
                    print(f"오래된 벡터스토어 삭제: {item}")
            except Exception as e:
                print(f"벡터스토어 정리 중 오류: {e}")

def ParseFirstFile(file_path):
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 분석 시작: {file_path}")
    
    # 1. HWP 파일에서 텍스트 추출
    print(f"[{time.strftime('%H:%M:%S')}] 1단계: HWP 파일 로딩 중...")
    loader_start = time.time()
    try:
        print(1)
        loader = HWPLoader(file_path)
        print(2)
        docs = loader.load()
        print(3)
        text = docs[0].page_content
        print(text)
        loader_time = time.time() - loader_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ HWP 파일 로딩 완료 ({loader_time:.2f}초)")
        print(f"   - 추출된 텍스트 길이: {len(text)} 문자")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ HWP 파일 로딩 실패: {e}")
        return {"error": f"HWP 파일 로딩 실패: {e}"}

    # 2. 시멘틱 청킹
    print(f"[{time.strftime('%H:%M:%S')}] 2단계: 텍스트 청킹 중...")
    chunk_start = time.time()
    try:
        chunked_docs = semantic_chunking(text)
        chunk_time = time.time() - chunk_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 텍스트 청킹 완료 ({chunk_time:.2f}초)")
        print(f"   - 생성된 청크 수: {len(chunked_docs)}개")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 텍스트 청킹 실패: {e}")
        return {"error": f"텍스트 청킹 실패: {e}"}

    # 3. 임베딩 및 벡터스토어 생성 (KURE-v1 모델 사용)
    print(f"[{time.strftime('%H:%M:%S')}] 3단계: 임베딩 모델 로딩 중... (처음 실행 시 다운로드 시간이 오래 걸릴 수 있습니다)")
    embed_start = time.time()
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nlpai-lab/KURE-v1",
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        embed_load_time = time.time() - embed_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 임베딩 모델 로딩 완료 ({embed_load_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 임베딩 모델 로딩 실패: {e}")
        return {"error": f"임베딩 모델 로딩 실패: {e}"}
    
    print(f"[{time.strftime('%H:%M:%S')}] 4단계: 벡터스토어 처리 중...")
    vector_start = time.time()
    try:
        vectorstore = load_or_create_vectorstore(chunked_docs, embeddings, file_path)
        vector_time = time.time() - vector_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 벡터스토어 처리 완료 ({vector_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 벡터스토어 처리 실패: {e}")
        return {"error": f"벡터스토어 처리 실패: {e}"}

    # 4. LLM 및 앙상블 리트리버 준비
    print(f"[{time.strftime('%H:%M:%S')}] 5단계: LLM 모델 준비 중...")
    llm_start = time.time()
    try:
        # HyDE 리트리버용 OpenAI LLM
        print(f"[{time.strftime('%H:%M:%S')}]   - OpenAI 모델 로딩 중...")
        hyde_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # 실제 LLM 체인용 Ollama LLM
        print(f"[{time.strftime('%H:%M:%S')}]   - Ollama 모델 로딩 중... (kanana3 모델 확인 중)")
        ollama_llm = Ollama(model="kanana3")
        
        retriever = get_ensemble_retriever(hyde_llm, embeddings, vectorstore)
        llm_time = time.time() - llm_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ LLM 모델 준비 완료 ({llm_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ LLM 모델 준비 실패: {e}")
        return {"error": f"LLM 모델 준비 실패: {e}"}

    # 5. 예시 쿼리로 리트리버 사용 (실제 사용에 맞게 수정)
    print(f"[{time.strftime('%H:%M:%S')}] 6단계: 문서 검색 중...")
    search_start = time.time()
    try:
        query = "이 문서의 기업명, 사업자 등록번호, 지원 과제 요약, 아이템 핵심 사항을 알려줘"
        retrieved_docs = retriever.invoke(query)
        combined_text = "\n".join([doc.page_content for doc in retrieved_docs])
        search_time = time.time() - search_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ 문서 검색 완료 ({search_time:.2f}초)")
        print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ 문서 검색 실패: {e}")
        return {"error": f"문서 검색 실패: {e}"}

    # 6. LLM 프롬프트 및 체인 실행 (Ollama LLM 사용)
    print(f"[{time.strftime('%H:%M:%S')}] 7단계: LLM 분석 중... (가장 오래 걸리는 단계)")
    analysis_start = time.time()
    try:
        # LLMChain 대신 직접 invoke 사용
        prompt_text = f"""
당신은 문서 분석 전문가입니다. 아래 텍스트에서 요청된 정보를 정확히 추출하여 JSON 형식으로만 응답해야 합니다.

중요: 반드시 JSON 형식으로만 응답하세요. 다른 설명이나 텍스트는 포함하지 마세요.

추출할 정보:
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

분석할 텍스트:
{combined_text}

응답 형식 (JSON만):
{{
    "기관명": "추출된 기관명 또는 빈 문자열",
    "담당자명": "추출된 담당자명 또는 빈 문자열",
    "연락처": "추출된 연락처 또는 빈 문자열",
    "기업명": "추출된 기업명 또는 빈 문자열",
    "사업자번호": "추출된 사업자번호 또는 빈 문자열",
    "대표자명": "추출된 대표자명 또는 빈 문자열",
    "연락처1": "추출된 연락처1 또는 빈 문자열",
    "연락처2": "추출된 연락처2 또는 빈 문자열",
    "지원과제명": "추출된 지원과제명 또는 빈 문자열",
    "아이템": "추출된 아이템 또는 빈 문자열",
    "추천사유": "추출된 추천사유 또는 빈 문자열"
}}

# 주의사항:
# 1. JSON 형식만 출력하세요
# 2. 설명이나 추가 텍스트는 절대 포함하지 마세요
# 3. 정보를 찾을 수 없으면 빈 문자열("")로 설정하세요
# 4. JSON 구문이 정확해야 합니다
# 5. 따옴표와 쉼표를 정확히 사용하세요

"""


        # Ollama LLM은 문자열을 직접 반환하므로 .content 제거
        result = ollama_llm.invoke(prompt_text)
        analysis_time = time.time() - analysis_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ LLM 분석 완료 ({analysis_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ LLM 분석 실패: {e}")
        return {"error": f"LLM 분석 실패: {e}"}

    # 7. 결과 반환 (문자열을 딕셔너리로 변환)
    print(f"[{time.strftime('%H:%M:%S')}] 8단계: 결과 처리 중...")
    process_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ LLM 원본 응답:\n{result}")
    import json
    
    import re
    
    try:
        # JSON 부분만 추출하는 정규식
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            result_dict = json.loads(json_str)
            print(f"[{time.strftime('%H:%M:%S')}] ✓ JSON 파싱 성공")
        else:
            # JSON을 찾을 수 없는 경우 원본 텍스트에서 JSON 부분만 추출 시도
            print(f"[{time.strftime('%H:%M:%S')}] ⚠ JSON 패턴을 찾을 수 없어 원본에서 추출 시도")
            result_dict = {"raw_output": result}
            
    except json.JSONDecodeError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠ JSON 파싱 실패: {e}")
        print(f"[{time.strftime('%H:%M:%S')}] 원본 출력: {result[:200]}...")
        result_dict = {"raw_output": result}
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠ 예상치 못한 오류: {e}")
        result_dict = {"raw_output": result}
    
    process_time = time.time() - process_start
    
    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] ===== 전체 분석 완료 =====")
    print(f"총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
    print(f"시간 분포:")
    print(f"  - HWP 로딩: {loader_time:.2f}초 ({(loader_time/total_time)*100:.1f}%)")
    print(f"  - 텍스트 청킹: {chunk_time:.2f}초 ({(chunk_time/total_time)*100:.1f}%)")
    print(f"  - 임베딩 모델 로딩: {embed_load_time:.2f}초 ({(embed_load_time/total_time)*100:.1f}%)")
    print(f"  - 벡터스토어 처리: {vector_time:.2f}초 ({(vector_time/total_time)*100:.1f}%)")
    print(f"  - LLM 모델 준비: {llm_time:.2f}초 ({(llm_time/total_time)*100:.1f}%)")
    print(f"  - 문서 검색: {search_time:.2f}초 ({(search_time/total_time)*100:.1f}%)")
    print(f"  - LLM 분석: {analysis_time:.2f}초 ({(analysis_time/total_time)*100:.1f}%)")
    print(f"  - 결과 처리: {process_time:.2f}초 ({(process_time/total_time)*100:.1f}%)")

    return result_dict

def main(file_path):
    import sys
    print('파일 경로', file_path)
    # 오래된 벡터스토어 정리 (선택사항)
    # cleanup_old_vectorstores()

    # 파일 경로를 명령행 인자로 받거나, 직접 입력받을 수 있음

    result = ParseFirstFile(file_path)
    print("전체 결과:", result)

    # 각 항목을 변수로 추출
    company_name = result.get("기업명")
    business_number = result.get("사업자 등록번호")
    project_summary = result.get("지원 과제 요약")
    item_core = result.get("아이템 핵심 사항")

    # 각 변수 출력
    print("기업명:", company_name)
    print("사업자 등록번호:", business_number)
    print("지원 과제 요약:", project_summary)
    print("아이템 핵심 사항:", item_core)
    return result