# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/75b22da6-f508-4ab7-8515-4f7794ceeac9

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/75b22da6-f508-4ab7-8515-4f7794ceeac9) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/75b22da6-f508-4ab7-8515-4f7794ceeac9) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)

---

## 로컬에서 한 번에 실행하기
모든 서비스를 개발 모드로 동시에 띄우려면 프로젝트 루트에서 `./start_all.sh`를 실행하세요. 이 스크립트는 순서대로 vLLM → FastAPI → Vite 개발 서버를 구동하고, 로그는 `logs/` 디렉터리에 기록됩니다.

```bash
chmod +x start_all.sh
./start_all.sh
```

- vLLM 설정은 `backend/start_vllm.sh`를 참고해 `VLLM_MODEL_PATH`, `VLLM_API_KEY` 등을 환경 변수로 넘겨줄 수 있습니다.
- FastAPI는 `backend/.env`(없으면 `.env.example` 복사)에서 Supabase, Serper, CORS 등 필요한 값을 읽습니다.

## Docker 기반 배포
동일한 구성을 컨테이너로 띄우고 싶다면 세 개의 Dockerfile과 `docker-compose.yml`을 사용합니다.

1. **환경 파일 준비**
   - `backend/.env` : FastAPI에서 사용하는 기존 환경 변수 (Supabase/Serper 키, `FASTAPI_HOST`, `FASTAPI_PORT` 등)를 이 파일에 채웁니다.
   - `deploy/.env.compose.example`을 복사해 `deploy/.env.compose`를 만들고, 모델 경로/포트/프론트엔드가 호출할 API URL 등을 수정합니다.

     ```bash
     cp deploy/.env.compose.example deploy/.env.compose
     ```

2. **(선택) 모델 디렉터리 공유**
   - 호스트에 있는 파인튜닝 모델 폴더를 `VLLM_MODEL_HOST_DIR`에 지정하면 Compose가 `vllm` 컨테이너 안의 `VLLM_MODEL_PATH`로 마운트합니다.
   - GPU 서버라면 NVIDIA Container Toolkit이 설치되어 있어야 하며, 최신 Docker/Compose(`docker compose` v2.6+)가 필요합니다.

3. **컨테이너 빌드 & 실행**

   ```bash
   docker compose --env-file deploy/.env.compose up -d --build
   ```

   - `frontend` 이미지는 `Dockerfile.frontend`에서 Vite 빌드를 진행한 뒤 Nginx(`deploy/frontend.nginx.conf`)로 정적 파일을 제공합니다.
   - `backend`는 `Dockerfile.backend`를 통해 FastAPI를 uvicorn 포그라운드 모드로 실행합니다.
   - `vllm`은 `Dockerfile.vllm` + `deploy/run_vllm.sh`로 OpenAI 호환 API를 띄웁니다. GPU가 여러 개인 경우 `VLLM_TENSOR_PARALLEL_SIZE` 등 추가 환경 변수를 넣어 조정할 수 있습니다.

4. **포트 및 도메인**
   - 기본적으로 프론트엔드 `8080`, FastAPI `8000`, vLLM `9000` 포트를 노출합니다. 필요하면 `deploy/.env.compose`에서 `FRONTEND_PORT`, `FASTAPI_PORT`, `VLLM_PORT`를 바꾸세요.
   - 운영에서는 프론트엔드 앞단에 Caddy/Nginx/Traefik을 두고 HTTPS를 적용한 뒤, `/api`는 FastAPI로 프록시하고 `/v1`은 vLLM으로 프록시하도록 구성하면 됩니다.

5. **중지 및 로그 확인**

   ```bash
   docker compose --env-file deploy/.env.compose logs -f
   docker compose --env-file deploy/.env.compose down
   ```

## 리버스 프록시 & HTTPS
- Nginx를 사용할 경우 `deploy/nginx/fullstack.conf` 예시를 `/etc/nginx/sites-available/grounded-query.conf`로 복사하고 도메인/포트를 수정한 뒤 `ln -s`로 `sites-enabled`에 연결합니다. `/`는 프런트엔드(8080), `/api/`는 FastAPI(8000), `/v1/`은 vLLM(9000)에 프록시되며, Certbot을 적용하면 자동으로 HTTPS 리다이렉트 구문을 추가할 수 있습니다.
- Caddy를 선호하면 `deploy/caddy/Caddyfile`을 `/etc/caddy/Caddyfile`로 복사하고 `grounded-query.example.com`을 실제 도메인으로 교체하십시오. Caddy는 자동으로 TLS 인증서를 발급하고 `/api`, `/v1` 경로를 내부 서비스로 포워딩합니다.
- 외부에 노출되는 서비스는 반드시 HTTPS를 사용하고, vLLM 엔드포인트에 API 키 검증을 추가하거나 리버스 프록시에서 `Authorization` 헤더를 강제로 넣어두면 안전합니다.

## systemd로 자동 시작
Docker Compose 스택을 부팅 시 자동으로 올리고 싶다면 `deploy/systemd/grounded-query-compose.service`를 사용하세요.

```bash
sudo cp deploy/systemd/grounded-query-compose.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now grounded-query-compose
```

- `WorkingDirectory` 경로와 `deploy/.env.compose` 파일 위치가 실제 서버와 다르면 서비스 파일을 수정해야 합니다.
- 멈추고 싶으면 `sudo systemctl stop grounded-query-compose`, 로그는 `journalctl -u grounded-query-compose -f`로 확인합니다.

## RunPod로 이전할 때
- 위 컨테이너 구성을 그대로 RunPod GPU 인스턴스에 복사하면 됩니다. 모델 디렉터리를 `/workspace/models` 등에 업로드하고 `deploy/.env.compose` 값만 pod 환경에 맞춰 변경하세요.
- 스토리지가 휘발성이라면 `backend/storage`를 S3/NFS 같은 영구 스토리지에 마운트하고, 업로드 데이터를 정기적으로 백업하세요.
