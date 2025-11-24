# Migration Guide: v1.0 → v2.0

## Overview of Changes

### Critical Security Improvements
1. ✅ **API Keys Secured**: No more exposed credentials in git
2. ✅ **.gitignore Added**: Prevents future accidental commits
3. ✅ **Environment Templates**: `.env.example` files guide proper setup

### Architecture Improvements
1. ✅ **Modular Backend**: 1,567-line monolith → organized modules
2. ✅ **Proper Logging**: 20+ print statements → structured logging system
3. ✅ **Type Safety**: TypeScript strict mode enabled
4. ✅ **Dependency Pinning**: Exact versions for reproducibility

## Before You Start

### ⚠️ IMPORTANT: Security Action Required

If your old `.env` file was committed to git, **your API keys are compromised**:

1. **Rotate Supabase Keys**:
   - Go to Supabase Dashboard → Settings → API
   - Click "Reset API Keys"
   - Update `.env` with new keys

2. **Rotate Serper API Key**:
   - Go to https://serper.dev/dashboard
   - Generate new API key
   - Revoke old key
   - Update `.env` with new key

3. **Clean Git History** (Optional but recommended):
   ```bash
   # Use BFG Repo-Cleaner to remove .env from history
   # https://rtyley.github.io/bfg-repo-cleaner/
   ```

## Migration Steps

### 1. Backup Your Data

```bash
# Backup document storage
cp -r old_project/backend/storage new_project/backend/storage

# Backup Supabase data (already in cloud, no action needed)
```

### 2. Environment Setup

#### Backend
```bash
cd new_project/backend

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or your preferred editor

# IMPORTANT: Use NEW API keys if old ones were exposed
```

#### Frontend
```bash
cd new_project/frontend

# Copy environment template
cp .env.example .env.local

# Edit if needed (defaults should work)
nano .env.local
```

### 3. Install Dependencies

#### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Frontend
```bash
cd frontend
npm install
```

### 4. Code Changes (If You Modified Original)

If you customized the v1.0 code, you'll need to update imports:

#### Import Path Changes

```python
# OLD (v1.0)
from main import embedder, extract_text_from_file

# NEW (v2.0)
from services.embedding_service import embedder
from services.document_service import extract_text_from_file
```

#### Common Import Mappings

| Old Import | New Import |
|-----------|-----------|
| `from main import embedder` | `from services.embedding_service import embedder` |
| `from main import sb` | `from models.database import get_supabase_client` |
| `from main import query_vllm` | `from services.llm_service import query_vllm` |
| `from main import search_web_serper` | `from services.search_service import search_web_serper` |

#### Configuration Access

```python
# OLD
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")

# NEW
from core.config import VLLM_BASE_URL
```

#### Logging

```python
# OLD
print(f"[DEBUG] Processing document...")

# NEW
from core.logging import setup_logger
logger = setup_logger(__name__)
logger.info("Processing document...")
logger.debug("Debug information")
logger.error("Error occurred")
```

### 5. Testing

#### Start Services

```bash
# Terminal 1: Backend
cd backend
./start_backend.sh

# Terminal 2: Frontend
cd frontend
./start_frontend.sh

# Terminal 3: vLLM (if needed)
cd backend
./start_vllm.sh
```

#### Verify Functionality

1. Open http://localhost:8080
2. Upload a test document
3. Ask a question
4. Verify answer is generated
5. Check logs in `logs/backend.log`

### 6. Database Migration

**Good news**: No database migration needed! The Supabase schema is identical.

Your existing documents and embeddings will work without changes.

## Feature Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Document upload | ✅ | ✅ |
| Vector search | ✅ | ✅ |
| Web search fallback | ✅ | ✅ |
| Logging | print() | Structured logging |
| Code structure | 1 file | Modular |
| Security | ⚠️ Keys exposed | ✅ Secured |
| Type safety | Partial | Strict |
| Dependencies | Unpinned | Pinned versions |
| Error handling | Basic | Comprehensive |
| Documentation | Minimal | Complete |

## What's Removed

1. **MCP (Model Context Protocol)** incomplete code - Removed unfinished implementation
2. **Agent endpoint** - Simplified for v2.0, can be re-added if needed
3. **Debug print statements** - Replaced with proper logging

## Rollback Plan

If you need to rollback to v1.0:

```bash
# Keep v1.0 code in archive
cp -r old_project archive/v1.0-backup-$(date +%Y%m%d)

# Continue using v1.0
cd old_project
```

## Getting Help

### Common Issues

**"ModuleNotFoundError"**
```bash
# Ensure you're in the virtual environment
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

**"Supabase connection failed"**
```bash
# Check .env file
cat backend/.env | grep SUPABASE

# Verify keys are not the same as old exposed keys
```

**"TypeScript errors"**
```bash
# v2.0 uses strict mode
# Fix type errors or temporarily disable strict mode in tsconfig.json
```

### Support

- GitHub Issues: [Your repo URL]
- Original code: `archive/` directory
- Documentation: `README.md`

## Recommendations

1. ✅ **Use v2.0 for new features** - Better foundation for growth
2. ✅ **Rotate exposed API keys** - Don't skip this step
3. ✅ **Review logs directory** - Monitor application health
4. ✅ **Keep v1.0 as backup** - Until fully confident in v2.0

## Next Steps After Migration

1. **Set up CI/CD** - Use GitHub Actions for testing
2. **Add monitoring** - Application performance monitoring
3. **Write tests** - Unit and integration tests
4. **Review security** - Run security audit
5. **Performance tuning** - Optimize based on usage patterns

Happy migrating! 🚀
