from langchain_teddynote.document_loaders import HWPLoader
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

class HyDERetriever(BaseRetriever):
    llm: Any = Field()
    embeddings: Any = Field()
    vectorstore: Any = Field()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None):
        prompt = f"""
당신은 정부 지원 사업계획서의 핵심 내용을 요약하는 AI 전문가입니다.
아래 '질문'에 가장 잘 부합하는 가상의 사업계획서 핵심 내용(Hypothetical Document)을 생성해 주세요. 이 내용은 실제 사업계획서의 일부처럼 보여야 합니다.

질문: {query}

---
**가상 사업계획서 생성 예시:**

**1. 신청 현황**
  - **기업명:** (주)테크이노베이션
  - **대표자명:** 김혁신
  - **사업자등록번호:** 123-45-67890
  - **담당자:** 이노아
  - **연락처:** 010-1234-5678

**2. 과제 개요**
  - **지원 과제명:** 빅데이터 기반 실시간 수요 예측 및 자동 발주 시스템 개발
  - **핵심 아이템:** AI 수요 예측 솔루션
  - **기술 요약:** 머신러닝(LSTM) 모델을 활용하여 판매 데이터를 분석하고, 재고 및 물류 최적화를 위한 자동 발주 알고리즘을 구현합니다. 이를 통해 중소 유통업체의 재고 관리 비용을 30% 절감하고 운영 효율성을 극대화하는 것을 목표로 합니다.
---

이제 위 예시와 같은 형식으로, 주어진 질문에 대한 가상의 사업계획서 핵심 내용을 작성하세요:
"""
        response = self.llm.invoke(prompt)
        hypothetical_doc = response.content
            
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
    # 한글 및 특수문자 제거하여 폴더명 생성
    safe_name = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).strip()
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
        loader = HWPLoader(file_path)
        docs = loader.load()
        text = docs[0].page_content
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

    # 3. 임베딩 및 벡터스토어 생성 (OpenAIEmbeddings 사용)
    print(f"[{time.strftime('%H:%M:%S')}] 3단계: 임베딩 모델 로딩 중...")
    embed_start = time.time()
    try:
        embeddings = OpenAIEmbeddings()
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
        # HyDE 리트리버 및 분석용 OpenAI LLM
        print(f"[{time.strftime('%H:%M:%S')}]   - OpenAI 모델 로딩 중...")
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        retriever = get_ensemble_retriever(llm, embeddings, vectorstore)
        llm_time = time.time() - llm_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ LLM 모델 준비 완료 ({llm_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ LLM 모델 준비 실패: {e}")
        return {"error": f"LLM 모델 준비 실패: {e}"}

    # 5. 예시 쿼리로 리트리버 사용 (실제 사용에 맞게 수정)
    print(f"[{time.strftime('%H:%M:%S')}] 6단계: 문서 검색 중...")
    search_start = time.time()
    # try:
    query = "이 문서의 기업명, 사업자 등록번호, 지원 과제 요약, 아이템 핵심 사항을 알려줘"
    print(3)
    retrieved_docs = retriever.invoke(query)
    print(2)
    combined_text = "\n".join([doc.page_content for doc in retrieved_docs])
    print(1)
    search_time = time.time() - search_start
    print(f"[{time.strftime('%H:%M:%S')}] ✓ 문서 검색 완료 ({search_time:.2f}초)")
    print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
    # except Exception as e:
    #     print(f"[{time.strftime('%H:%M:%S')}] ❌ 문서 검색 실패: {e}")
    #     return {"error": f"문서 검색 실패: {e}"}

    # 6. LLM 프롬프트 및 체인 실행 (OpenAI LLM 사용)
    print(f"[{time.strftime('%H:%M:%S')}] 7단계: LLM 분석 중... (가장 오래 걸리는 단계)")
    analysis_start = time.time()
    try:
        # LLMChain 대신 직접 invoke 사용
        prompt_text = f"""
당신은 대한민국 정부 지원 사업계획서 분석 전문가입니다. 제공된 사업계획서 텍스트에서 다음 항목들을 매우 신중하고 정확하게 추출하여 JSON 형식으로 응답해 주세요.

**지침:**
1.  **정확성:** 주어진 텍스트에 명시적으로 언급된 정보만 추출하세요. 추측하거나 정보를 만들어내지 마세요.
2.  **형식:** 반드시 지정된 JSON 형식으로만 응답해야 합니다. 다른 설명이나 추가 텍스트는 절대 포함하지 마세요.
3.  **빈 값 처리:** 만약 텍스트에서 특정 정보를 찾을 수 없다면, 해당 필드의 값으로 빈 문자열("")을 사용하세요.
4.  **세부 항목별 힌트:**
    *   **기관명:** '신청기관', '주관기관' 등의 키워드 주변을 확인하세요.
    *   **담당자명:** '담당자', '책임자' 등의 키워드와 함께 나오는 이름을 찾으세요.
    *   **연락처:** '연락처', '전화', '핸드폰', 'HP' 등의 키워드와 함께 나오는 전화번호를 찾으세요.
    *   **기업명:** '기업명', '회사명', '창업기업명' 등의 키워드를 찾아보세요. 보통 표의 형태로 제공될 수 있습니다.
    *   **사업자번호:** '사업자등록번호' 키워드와 함께 나오는 'XXX-XX-XXXXX' 형식의 번호를 찾으세요.
    *   **대표자명:** '대표자', '대표' 키워드와 함께 나오는 이름을 찾으세요.
    *   **연락처1 (대표자 연락처):** 대표자의 연락처를 찾아보세요. '연락처', '핸드폰' 등으로 표시될 수 있습니다.
    *   **연락처2 (담당자 연락처):** 담당자의 연락처를 찾아보세요. '연락처', '핸드폰' 등으로 표시될 수 있습니다.
    *   **지원과제명:** '과제명', '사업명', '아이템명' 등의 키워드로 시작하는 긴 제목을 찾으세요.
    *   **아이템:** '개발하고자 하는 아이템', '주요 아이템' 등 과제명을 요약한 핵심 기술 또는 제품명을 찾으세요.
    *   **추천사유:** 문서 전체 내용을 바탕으로, 이 과제가 왜 필요한지, 어떤 문제를 해결하는지, 그리고 어떤 기술적/사회적 가치가 있는지를 1~2문장으로 요약하여 추천 사유를 작성하세요.

**분석할 텍스트:**
---
{combined_text}
---

**응답 형식 (JSON만):**
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
    "추천사유": "추천 사유 요약"
}}
"""
        result = llm.invoke(prompt_text).content
        analysis_time = time.time() - analysis_start
        print(f"[{time.strftime('%H:%M:%S')}] ✓ LLM 분석 완료 ({analysis_time:.2f}초)")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ LLM 분석 실패: {e}")
        return {"error": f"LLM 분석 실패: {e}"}

    # 7. 결과 반환 (문자열을 딕셔너리로 변환)
    print(f"[{time.strftime('%H:%M:%S')}] 8단계: 결과 처리 중...")
    process_start = time.time()
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