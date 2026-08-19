# Deployment Guide

## Quick Deploy Commands

### Streamlit Community Cloud (Free, Recommended)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select your repo
4. Main file: `app.py`
5. Add secrets (see below)

### Render.com
1. Create new **Web Service**
2. Connect GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false`

### Railway
1. New Project → Deploy from GitHub
2. Add `PORT` environment variable (Railway sets this automatically)
3. Start command from `Procfile` is used automatically

### Fly.io
```bash
fly launch  # Detects Streamlit, creates fly.toml
fly deploy
```

### Heroku
```bash
heroku create your-app-name
heroku buildpacks:set heroku/python
git push heroku main
```

## Required Environment Variables

Set these in your platform's dashboard (Settings → Environment Variables / Secrets):

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `GEMINI_MODEL` | No | Default: `gemini-3.6-flash` |
| `GEMINI_EMBEDDING_MODEL` | No | Default: `models/gemini-embedding-001` |
| `CHUNK_SIZE` | No | Default: `1000` |
| `CHUNK_OVERLAP` | No | Default: `200` |
| `TOP_K` | No | Default: `4` |

## Streamlit Cloud Specific

In Streamlit Cloud, add secrets in `.streamlit/secrets.toml` format:

```toml
GOOGLE_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
```

Or use the UI: Settings → Secrets → paste the TOML above.

## Common Issues

### "gunicorn app:app" Error
**You're deploying a Streamlit app, not a Flask/FastAPI app.**  
Use `streamlit run app.py` as the start command, NOT gunicorn.

### Port Binding
Always use `$PORT` environment variable:
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### CORS/XSRF
Disable for deployment:
```bash
--server.enableCORS=false --server.enableXsrfProtection=false
```

### Memory Issues (Large PDFs)
- Increase instance memory
- Or reduce `CHUNK_SIZE` in env vars

### API Quota Exceeded
- Free tier: 100 embeddings/minute
- Enable billing on Google Cloud for higher limits
- Or use smaller PDFs

## Files Created for Deployment

| File | Purpose |
|------|---------|
| `Procfile` | Start command for Heroku/Railway/Render |
| `runtime.txt` | Python version |
| `.streamlit/config.toml` | Streamlit server config |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Local Testing of Production Build

```bash
# Test with production-like settings
export PORT=8501
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
```

## Health Check Endpoint

Streamlit provides a health check at:
```
GET /_stcore/health
```
Returns `ok` when healthy.