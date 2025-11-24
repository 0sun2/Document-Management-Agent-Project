# Document Management RAG System (Refactored)

> **Version 2.0** - Complete refactoring with modular architecture, improved security, and best practices

A production-ready RAG (Retrieval-Augmented Generation) system for document question answering with web search fallback. Built with FastAPI, React, and vLLM.

## What's New in 2.0

### Architecture Improvements
✅ **Modular Backend Structure** - 1,567-line monolith split into organized modules
✅ **Proper Logging System** - Replaced 20+ print statements with structured logging
✅ **Type Safety** - TypeScript strict mode enabled
✅ **Security Best Practices** - .gitignore, .env.example, no hardcoded secrets
✅ **Dependency Pinning** - Exact versions for reproducible builds

### Security Fixes
✅ **No Exposed API Keys** - Template files only, real keys in .env (gitignored)
✅ **Path Traversal Protection** - Filesystem tools limited to DOC_ROOT
✅ **Input Validation** - Pydantic schemas for all requests

### Code Quality
✅ **Separation of Concerns** - Services, models, routes clearly separated
✅ **Logging Framework** - File and console logging with levels
✅ **Error Handling** - Proper exception handling throughout
✅ **Documentation** - Docstrings and type hints everywhere

## Architecture

```
project/
├── backend/                    # Python FastAPI backend
│   ├── api/                    # API routes
│   │   └── routes/
│   │       ├── documents.py    # Document upload/list/delete
│   │       └── qa.py           # Question answering
│   ├── core/                   # Core utilities
│   │   ├── config.py           # Configuration management
│   │   └── logging.py          # Logging setup
│   ├── models/                 # Data models
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── database.py         # Supabase client
│   ├── services/               # Business logic
│   │   ├── document_service.py # Document processing
│   │   ├── embedding_service.py# Vector embeddings
│   │   ├── llm_service.py      # LLM communication
│   │   └── search_service.py   # Web search
│   ├── tools/                  # Agent tools
│   │   ├── http_fetch.py       # HTTP requests
│   │   ├── serper_search.py    # Web search
│   │   └── filesystem.py       # File operations
│   ├── storage/                # Data storage
│   │   ├── originals/          # Uploaded documents
│   │   └── vectors/            # Vector database
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Environment variables template
├── frontend/                   # React TypeScript frontend
│   ├── src/                    # Source code
│   ├── public/                 # Static assets
│   ├── tsconfig.json           # TypeScript config (strict mode)
│   ├── package.json            # Node dependencies
│   └── .env.example            # Frontend environment template
├── logs/                       # Application logs
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL with pgvector extension (via Supabase)
- GPU with CUDA (for vLLM server)

### Option 1: 한번에 모두 실행 (권장)

```bash
# 1. 환경 변수 설정
cp backend/.env.example backend/.env
nano backend/.env  # API 키 입력

cp frontend/.env.example frontend/.env.local

# 2. 의존성 설치
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..

cd frontend
npm install
cd ..

# 3. 모든 서비스 한번에 시작
./start_all.sh

# 종료하려면: Ctrl+C 또는
./stop_all.sh
```

이 명령어는 다음을 순서대로 실행합니다:
1. vLLM 서버 (포트 9000) - 백그라운드
2. FastAPI 백엔드 (포트 8000) - 백그라운드
3. React 프론트엔드 (포트 8080) - 포그라운드

### Option 2: 개별 실행

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your actual credentials

# Run backend
./start_backend.sh
```

#### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local if needed

# Run development server
./start_frontend.sh
```

#### 3. vLLM Server Setup

```bash
cd backend

# Edit start_vllm.sh with your model path if needed
# Default: /home/lys/Desktop/qwen3_4b_ft/final_merged_model

./start_vllm.sh
```

### Option 3: Docker로 배포 (프로덕션 권장)

Docker Compose를 사용하여 모든 서비스를 컨테이너로 실행합니다.

#### 사전 준비

1. **Docker & Docker Compose 설치**
   ```bash
   # Docker 설치 확인
   docker --version
   docker-compose --version
   ```

2. **NVIDIA Container Toolkit 설치** (GPU 사용을 위해 필수)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # 설치 확인
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

#### 배포 단계

1. **환경 변수 설정**
   ```bash
   # Docker용 환경 변수 파일 생성
   cp .env.docker .env

   # .env 파일 편집 (필수 항목 입력)
   nano .env
   ```

   필수 항목:
   - `SUPABASE_URL`: Supabase 프로젝트 URL
   - `SUPABASE_ANON_KEY`: Supabase Anon Key
   - `SERPER_API_KEY`: Serper API 키
   - `VLLM_MODEL_PATH`: 호스트 머신의 모델 경로 (절대 경로)

2. **Docker 이미지 빌드 및 실행**
   ```bash
   # 모든 서비스 시작
   docker-compose up -d

   # 로그 확인
   docker-compose logs -f

   # 특정 서비스 로그만 확인
   docker-compose logs -f backend
   docker-compose logs -f vllm
   ```

3. **서비스 접근**
   - 프론트엔드: `http://서버IP:8080`
   - 백엔드 API: `http://서버IP:8000`
   - API 문서: `http://서버IP:8000/docs`
   - vLLM 서버: `http://서버IP:9000`

4. **서비스 관리**
   ```bash
   # 서비스 중지
   docker-compose down

   # 서비스 중지 및 볼륨 삭제 (데이터 초기화)
   docker-compose down -v

   # 서비스 재시작
   docker-compose restart

   # 특정 서비스만 재시작
   docker-compose restart backend

   # 서비스 상태 확인
   docker-compose ps
   ```

#### 외부 접근 허용 (포트 포워딩)

내부 네트워크에서 외부 접근을 허용하려면:

1. **공유기 포트 포워딩 설정**
   - 공유기 관리자 페이지 접속
   - 포트 포워딩 규칙 추가:
     - 외부 포트 8080 → 서버 IP:8080 (프론트엔드)
     - 외부 포트 8000 → 서버 IP:8000 (백엔드)

2. **방화벽 설정**
   ```bash
   # UFW 사용 시
   sudo ufw allow 8080/tcp
   sudo ufw allow 8000/tcp
   ```

3. **외부 IP 확인**
   ```bash
   curl ifconfig.me
   ```

   외부에서 `http://외부IP:8080`으로 접속

#### 프로덕션 배포 시 주의사항

1. **HTTPS 설정 권장**
   - Nginx 리버스 프록시 + Let's Encrypt 사용
   - Cloudflare Tunnel 또는 ngrok 사용

2. **환경 변수 보안**
   - `.env` 파일을 git에 절대 커밋하지 말 것
   - 프로덕션 환경에서는 별도의 비밀 관리 도구 사용 권장

3. **데이터 백업**
   ```bash
   # Docker 볼륨 백업
   docker run --rm -v rag-backend-storage:/data -v $(pwd):/backup \
       ubuntu tar czf /backup/backend-storage-backup.tar.gz /data
   ```

4. **리소스 모니터링**
   ```bash
   # 컨테이너 리소스 사용량 확인
   docker stats
   ```

#### 트러블슈팅

**vLLM OOM 에러**
```bash
# .env 파일에서 GPU 메모리 사용률 낮추기
VLLM_GPU_MEMORY_UTILIZATION=0.6
VLLM_MAX_NUM_SEQS=64

# 재시작
docker-compose restart vllm
```

**프론트엔드에서 백엔드 연결 실패**
- 브라우저에서 접근 시 `http://localhost:8000`이 아닌 실제 서버 IP 사용
- `frontend/.env.production` 파일에서 `VITE_FASTAPI_URL` 확인

**Docker 빌드 실패**
```bash
# 캐시 없이 다시 빌드
docker-compose build --no-cache

# 개별 서비스만 빌드
docker-compose build backend
```

## Environment Variables

### Backend (.env)

```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SERPER_API_KEY=your_serper_api_key_here

# Optional (defaults provided)
VLLM_BASE_URL=http://127.0.0.1:9000/v1
VLLM_MODEL_NAME=qwen3-4b-ft
EMBEDDER_MODEL=intfloat/multilingual-e5-large-instruct
FASTAPI_PORT=8000
```

### Frontend (.env.local)

```env
VITE_FASTAPI_URL=http://127.0.0.1:8000
```

## Key Features

### 1. Document Processing
- **Supported formats**: PDF, DOCX, TXT, MD
- **Text extraction**: pdfplumber, python-docx
- **Chunking**: RecursiveCharacterTextSplitter (800 chars, 120 overlap)
- **Embeddings**: multilingual-e5-large-instruct (1024-dim)
- **Storage**: Supabase + pgvector

### 2. Question Answering
- **Vector search**: Cosine similarity in Supabase
- **Keyword boosting**: Filename + content token matching
- **Smart fallback**: Automatic web search when confidence is low
- **LLM**: Fine-tuned Qwen3-4B via vLLM

### 3. Web Search Integration
- **Provider**: Serper API
- **Triggers**: Low similarity, missing keywords, insufficient coverage
- **Hybrid mode**: Combines document context + web results

### 4. Security & Best Practices
- **No secrets in code**: All credentials via environment variables
- **Path traversal protection**: Filesystem tools restricted to DOC_ROOT
- **Input validation**: Pydantic schemas
- **CORS configuration**: Configurable allowed origins
- **Logging**: Structured logging to files and console

## API Endpoints

### Documents

- `GET /health` - Health check
- `GET /documents?q={query}` - List documents with optional keyword search
- `POST /upload` - Upload and index a document
- `DELETE /documents/{doc_id}` - Delete a document and its embeddings

### Question Answering

- `POST /ask` - Ask a question (RAG + optional web search)

See API documentation at `http://127.0.0.1:8000/docs` when server is running.

## Development

### Running Tests

```bash
# Backend tests (TODO: implement)
cd backend
pytest

# Frontend tests (TODO: implement)
cd frontend
npm test
```

### Code Quality

```bash
# Backend linting
cd backend
black .
flake8 .

# Frontend linting
cd frontend
npm run lint
```

### Logging

Logs are written to `logs/backend.log` with the following levels:
- **INFO**: Normal operations (server start, document upload, etc.)
- **WARNING**: Recoverable issues (Supabase fallback, search failure, etc.)
- **ERROR**: Serious problems (API failures, database errors, etc.)

## Troubleshooting

### "Supabase not configured"
- Check `backend/.env` has `SUPABASE_URL` and `SUPABASE_ANON_KEY`
- Verify Supabase project is running
- Ensure pgvector extension is enabled

### "vLLM request failed"
- Check vLLM server is running on port 9000: `curl http://127.0.0.1:9000/health`
- Verify `VLLM_BASE_URL` in `.env`
- Check vLLM logs for OOM errors (reduce `VLLM_GPU_MEMORY_UTILIZATION`)

### "SERPER_API_KEY not configured"
- Add your Serper API key to `backend/.env`
- Get one at https://serper.dev

### CORS errors
- Add your frontend URL to `CORS_ORIGINS` in `backend/.env`
- Example: `CORS_ORIGINS=http://127.0.0.1:8080,http://localhost:3000`

## Migration from v1.0

If you're upgrading from the old monolithic structure:

1. **Backup your data**:
   ```bash
   cp -r old_project/backend/storage new_project/backend/storage
   ```

2. **Update environment variables**:
   - Copy values from old `.env` to new `backend/.env`
   - **IMPORTANT**: Rotate any API keys that were committed to git

3. **Update imports** (if you customized code):
   - Old: `from main import embedder`
   - New: `from services.embedding_service import embedder`

4. **Database migration**: No schema changes needed, Supabase data is compatible

## Contributing

This is a refactored version focusing on best practices:
- Follow the modular structure
- Use the logging system (not print statements)
- Add type hints to all functions
- Write docstrings for public APIs
- Update tests when adding features

## License

[Your license here]

## Acknowledgments

- Original prototype by [Your name]
- Refactored with best practices by Claude Code
- Built with FastAPI, React, Supabase, and vLLM
