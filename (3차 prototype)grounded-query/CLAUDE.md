# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드를 제공합니다.

## 프로젝트 개요

React/TypeScript 프론트엔드와 FastAPI 백엔드를 사용하는 RAG (Retrieval-Augmented Generation) QA 챗봇입니다. 사용자가 문서(PDF, DOCX, TXT, MD)를 업로드하면 임베딩을 통해 인덱싱되어 ChromaDB에 저장됩니다. 사용자가 질문하면 시스템이 관련 문서 청크를 검색하여 로컬에서 호스팅되는 vLLM 서버를 사용해 근거 있는 답변을 생성합니다.

## 아키텍처

### 3계층 서비스 아키텍처

애플리케이션은 함께 작동해야 하는 3개의 독립적인 서비스로 구성됩니다:

1. **vLLM 서버** (포트 9000): 파인튜닝된 Qwen3-4B 모델을 실행하는 OpenAI 호환 API 서버
2. **FastAPI 백엔드** (포트 8000): 문서 처리, 벡터 검색, LLM 오케스트레이션을 담당하는 Python 서비스
3. **Vite 프론트엔드** (포트 8080): 사용자 인터페이스를 제공하는 React/TypeScript SPA

### 데이터 흐름

```
문서 업로드 → FastAPI 텍스트 추출 → 텍스트 청킹 (600자, 120자 겹침)
→ 임베딩 생성 (multilingual-e5-large-instruct) → ChromaDB 저장

질문 입력 → 쿼리 임베딩 → ChromaDB 벡터 유사도 검색 (top_k=5)
→ 컨텍스트 + 질문을 vLLM에 전송 → 답변 생성 → 인용과 함께 반환
```

### 핵심 컴포넌트

**백엔드 (Python FastAPI)**
- `backend/main.py`: 모든 엔드포인트, 벡터 저장소 로직, LLM 통합을 포함하는 단일 파일 백엔드
- 문서 처리: PDF(pdfplumber), DOCX(python-docx), TXT/MD(평문) 지원
- 벡터 저장소: `backend/storage/chroma/`에 영구 저장되는 ChromaDB
- 임베딩: `intfloat/multilingual-e5-large-instruct`를 사용하는 sentence-transformers
- LLM: OpenAI 호환 `/v1/chat/completions` 엔드포인트를 통해 vLLM에 연결

**프론트엔드 (React + TypeScript)**
- `src/pages/Index.tsx`: 문서 패널과 채팅 인터페이스가 있는 메인 페이지 레이아웃
- `src/adapters/llm.ts`: 채팅 응답을 위한 FastAPI 백엔드 통합
- `src/adapters/vectorStore.ts`: 문서 업로드 및 삭제 처리
- `src/components/ChatInterface.tsx`: 메시지 히스토리가 있는 채팅 UI
- `src/components/DocumentUpload.tsx`: 드래그 앤 드롭을 지원하는 파일 업로드 컴포넌트
- `src/components/DocumentList.tsx`: 상태와 함께 업로드된 문서 표시

**어댑터 패턴**
- 프론트엔드는 백엔드와 통신하기 위해 어댑터 인터페이스(`LLMAdapter`, `VectorStoreAdapter`) 사용
- 이 추상화를 통해 UI 코드 변경 없이 백엔드 구현 교체 가능
- 모든 RAG 로직(벡터 검색, LLM 프롬프팅)은 백엔드에서 수행

## 주요 명령어

### 개발 환경

모든 서비스 한 번에 시작:
```bash
./start_all.sh
```

서비스 개별 시작:
```bash
# 터미널 1: vLLM 서버 (GPU 필요)
cd backend && ./start_vllm.sh

# 터미널 2: FastAPI 백엔드
cd backend && ./start_backend.sh

# 터미널 3: 프론트엔드 개발 서버
./start_frontend.sh
```

모든 서비스 중지:
```bash
./stop_all.sh
```

### 프론트엔드 명령어

```bash
# 의존성 설치
npm install

# 개발 서버 시작 (포트 8080)
npm run dev

# 프로덕션 빌드
npm run build

# 개발용 빌드 (소스맵 포함)
npm run build:dev

# 프로덕션 빌드 미리보기
npm run preview

# 코드 린트
npm run lint
```

### 백엔드 명령어

```bash
# Python 의존성 설치
cd backend
pip install -r requirements.txt

# Python으로 백엔드 직접 실행
cd backend
python3 main.py

# 또는 uvicorn으로 실행
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 테스트

현재 프로젝트에는 자동화된 테스트가 구성되어 있지 않습니다.

## 환경 설정

### 백엔드 환경 변수

`backend/.env` 생성 (예시: `backend/.env.example`):
- `VLLM_BASE_URL`: vLLM 서버 엔드포인트 (기본값: http://127.0.0.1:9000/v1)
- `VLLM_MODEL_NAME`: 요청할 모델 이름 (기본값: qwen3-4b-ft)
- `VLLM_API_KEY`: vLLM API 키 (기본값: EMPTY)
- `VLLM_TEMPERATURE`: LLM 온도 (기본값: 0.2)
- `VLLM_MAX_TOKENS`: 최대 응답 토큰 수 (기본값: 1024)
- `EMBEDDER_MODEL`: 임베딩 모델 이름 (기본값: intfloat/multilingual-e5-large-instruct)
- `CORS_ORIGINS`: 허용된 CORS 출처 (기본값: http://127.0.0.1:8080,http://localhost:8080)
- `FASTAPI_HOST`: FastAPI 바인딩 호스트 (기본값: 127.0.0.1)
- `FASTAPI_PORT`: FastAPI 바인딩 포트 (기본값: 8000)

### 프론트엔드 환경 변수

`.env.local` 생성 (예시: `.env.example`):
- `VITE_FASTAPI_URL`: 백엔드 API URL (기본값: http://127.0.0.1:8000)

### vLLM 시작 설정

`backend/start_vllm.sh`를 편집하여 vLLM 파라미터 수정:
- `VLLM_MODEL_PATH`: 모델 가중치 경로 (기본값: /home/lys/Desktop/qwen3_4b_ft/final_merged_model)
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU 메모리 사용률 (기본값: 0.75, OOM 발생 시 낮춤)
- `VLLM_MAX_NUM_SEQS`: 최대 동시 시퀀스 수 (기본값: 128, OOM 발생 시 낮춤)
- `PYTORCH_CUDA_ALLOC_CONF`: 메모리 최적화를 위해 `expandable_segments:True`로 설정

## 중요한 구현 세부사항

### 문서 청킹 전략

문서는 컨텍스트를 보존하기 위해 겹치는 윈도우로 청킹됩니다:
- 청크 크기: 600자
- 겹침: 120자
- 스텝: 480자 (chunk_size - overlap)
- 함수: `backend/main.py:109`의 `chunk_text()`

### 벡터 검색 및 유사도

- 임베딩 함수는 코사인 유사도 사용 (ChromaDB 설정: `hnsw:space: cosine`)
- ChromaDB는 거리를 반환하며, 유사도로 변환: `similarity = 1.0 - distance`
- Top-K 검색은 기본적으로 5개 청크 반환
- 유사도 임계값이 신뢰도 수준을 결정:
  - 높음(High): similarity > 0.85
  - 중간(Medium): similarity > 0.7
  - 낮음(Low): similarity ≤ 0.7

### LLM 프롬프팅 패턴

백엔드는 다음 구조로 한국어 프롬프트를 구성합니다 (main.py:190의 `build_prompt()` 참조):
```
[CONTEXT]
[1] chunk_text_1
[2] chunk_text_2
...

[QUESTION]
user_question

지침: 컨텍스트에 있는 정보만 사용해 한국어로 답변, 불확실하면 "모른다"고 말할 것.
```

시스템 프롬프트는 모델에게 컨텍스트만 사용하여 답변하는 한국어 어시스턴트가 되도록 지시합니다.

### 파일 업로드 흐름

1. 프론트엔드가 `status: 'pending'`인 임시 Document 객체 생성
2. 파일, document_id, original_filename을 포함한 `FormData`로 `POST /upload`에 전송
3. 백엔드가 텍스트 추출 → 청킹 → 임베딩 → ChromaDB 저장
4. 백엔드가 원본 파일을 `backend/storage/originals/{doc_id}/{filename}`에 저장
5. 프론트엔드가 상태를 `indexing`으로 업데이트하며 진행 상황 시뮬레이션
6. 성공 시 상태가 `completed`가 되고, 오류 시 오류 메시지 표시

### 질문 답변 흐름

1. 프론트엔드가 `{question, top_k}`로 `POST /ask` 전송
2. 백엔드가 질문 임베딩 → ChromaDB 검색 → top_k 청크 검색
3. 청크 + 질문을 프롬프트로 포맷팅 → vLLM에 전송
4. vLLM이 컨텍스트를 사용하여 답변 생성
5. 백엔드가 `{answer, matches, citations, used_web_fallback, source, top_similarity}` 반환
6. 프론트엔드가 소스 청크로의 인용 링크와 함께 답변 표시

### 기술 스택

**프론트엔드:**
- React 18 + TypeScript
- Vite 빌드 도구
- React Router (내비게이션)
- TanStack Query (데이터 페칭)
- Radix UI + Tailwind CSS (컴포넌트, shadcn/ui)
- Lucide React 아이콘

**백엔드:**
- FastAPI with uvicorn
- ChromaDB (영구 벡터 저장소)
- sentence-transformers (임베딩)
- pdfplumber, python-docx (문서 파싱)
- httpx (vLLM으로의 비동기 HTTP)

**LLM 추론:**
- vLLM으로 서빙되는 파인튜닝된 Qwen3-4B
- OpenAI 호환 API

## 프로젝트 구조 참고사항

- `src/adapters/`: 백엔드 통신을 위한 추상화 계층
- `src/components/ui/`: shadcn/ui의 재사용 가능한 UI 컴포넌트
- `src/types/`: TypeScript 타입 정의
- `backend/storage/originals/`: doc_id별로 구성된 업로드된 문서 파일
- `backend/storage/chroma/`: ChromaDB 영구 벡터 데이터베이스
- `logs/`: start_all.sh 사용 시 서비스 로그 (vllm.log, backend.log)

## Lovable 통합

이 프로젝트는 Lovable(구 GPT Engineer)로 생성되었습니다. 프론트엔드는 https://lovable.dev/projects/75b22da6-f508-4ab7-8515-4f7794ceeac9 의 Lovable 웹 인터페이스를 통해 편집할 수 있습니다. Lovable에서 변경한 사항은 자동으로 이 저장소에 커밋됩니다.

## 자주 발생하는 문제

**"FastAPI 연결 실패" 오류:**
- 백엔드가 포트 8000에서 실행 중인지 확인
- 프론트엔드 `.env.local`의 `VITE_FASTAPI_URL`이 백엔드 주소와 일치하는지 확인
- 백엔드 `.env`의 CORS_ORIGINS에 프론트엔드 URL이 포함되어 있는지 확인

**vLLM Out of Memory (OOM):**
- `backend/start_vllm.sh`의 `VLLM_GPU_MEMORY_UTILIZATION` 낮추기 (0.6 또는 0.5 시도)
- `VLLM_MAX_NUM_SEQS` 낮추기 (64 또는 32 시도)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 설정되어 있는지 확인

**ChromaDB 영속성 문제:**
- ChromaDB는 `backend/storage/chroma/`에 데이터를 저장
- 컬렉션 스키마가 변경되면 이 디렉토리를 삭제해야 할 수 있음
- 문서 삭제 시 ChromaDB 항목과 원본 파일이 모두 제거됨

**모델 경로를 찾을 수 없음:**
- `backend/start_vllm.sh`의 `VLLM_MODEL_PATH`를 모델 위치로 업데이트
- 기본값은 `/home/lys/Desktop/qwen3_4b_ft/final_merged_model`의 파인튜닝된 모델을 예상
