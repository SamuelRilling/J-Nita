# Railway Deployment Guide - Fixed

## Quick Deploy to Railway

### Step 1: Remove runtime.txt (Already Done)
Railway doesn't need `runtime.txt` - it auto-detects Python from `requirements.txt`.

### Step 2: Deploy

1. **Go to Railway**: [railway.app](https://railway.app)
2. **Sign up**: Use GitHub to sign in
3. **New Project**: Click "New Project"
4. **Deploy from GitHub**: Click "Deploy from GitHub repo"
5. **Select Repository**: Choose your OCR5 repository
6. **Auto-Deploy**: Railway will:
   - Auto-detect Python
   - Install dependencies from `requirements.txt`
   - Start the server using `run.py`

### Step 3: Get Your URL

1. Railway will automatically assign a URL
2. Click on your service
3. Go to "Settings" → "Domains"
4. Copy the default domain (e.g., `https://ocr5-production.up.railway.app`)
5. Or add a custom domain if you want

### Step 4: Configure Frontend

1. Open your GitHub Pages site
2. You should see "Backend Configuration" section (if you've pushed the updated index.html)
3. Enter your Railway URL
4. Click "Test Connection"

## Troubleshooting

### Build Fails with Python Version Error

**Solution**: Railway auto-detects Python version. If you see errors:
1. Remove `runtime.txt` (already done)
2. Railway will use Python 3.11+ automatically
3. The `.python-version` file helps with local development only

### Port Configuration

Railway automatically sets the `PORT` environment variable. The `run.py` script reads it correctly.

### Dependencies Not Installing

Make sure `requirements.txt` is in the root directory and all dependencies are listed.

### Service Won't Start

Check logs in Railway dashboard. Common issues:
- Missing dependencies
- Import errors
- Port configuration issues

## Environment Variables (Optional)

Railway allows you to set environment variables in the dashboard:
- `FLASK_ENV`: Set to `production` (optional)
- `PORT`: Auto-set by Railway (don't override)

## Free Tier Limits

Railway's free tier includes:
- $5 credit/month
- Sufficient for light usage
- Upgrade if you need more

## Next Steps

After deployment:
1. Test the backend: `https://your-app.up.railway.app/api/health`
2. Configure frontend with the URL
3. Start using the app!
