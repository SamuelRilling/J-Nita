# Backend Deployment Guide

The OCR5 frontend requires a backend API to function. This guide will help you deploy the backend to various platforms.

## Quick Deploy Options

### Option 1: Railway (Easiest) ⭐ Recommended

1. **Sign up**: Go to [railway.app](https://railway.app) and sign up with GitHub
2. **New Project**: Click "New Project" → "Deploy from GitHub repo"
3. **Select Repository**: Choose your OCR5 repository
4. **Auto-detect**: Railway will auto-detect Python and install dependencies
5. **Deploy**: Railway will automatically deploy
6. **Get URL**: Copy the generated URL (e.g., `https://ocr5-production.up.railway.app`)
7. **Configure Frontend**: Add this URL to the "Backend Configuration" section in the web app

**No configuration needed!** Railway handles everything automatically.

---

### Option 2: Render

1. **Sign up**: Go to [render.com](https://render.com) and sign up
2. **New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Connect your GitHub repository
4. **Configure**:
   - **Name**: `ocr5-backend` (or any name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
5. **Deploy**: Click "Create Web Service"
6. **Get URL**: Copy the URL (e.g., `https://ocr5-backend.onrender.com`)
7. **Configure Frontend**: Add this URL to the "Backend Configuration" section

---

### Option 3: Heroku

1. **Install Heroku CLI**: Download from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)
2. **Login**: 
   ```bash
   heroku login
   ```
3. **Create App**:
   ```bash
   heroku create your-app-name
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```
5. **Get URL**: Your app will be at `https://your-app-name.herokuapp.com`
6. **Configure Frontend**: Add this URL to the "Backend Configuration" section

**Note**: Heroku free tier was discontinued, but paid plans are available.

---

### Option 4: PythonAnywhere

1. **Sign up**: Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. **Create Web App**: Dashboard → Web → "Add a new web app"
3. **Upload Files**: Upload all Python files via Files tab
4. **Configure**:
   - **Source code**: `/home/yourusername/ocr5/app.py`
   - **WSGI file**: Edit and point to your Flask app
5. **Install Dependencies**: Open Bash console and run:
   ```bash
   pip3.10 install --user -r requirements.txt
   ```
6. **Reload**: Click "Reload" button
7. **Get URL**: Your app will be at `https://yourusername.pythonanywhere.com`
8. **Configure Frontend**: Add this URL to the "Backend Configuration" section

---

## After Deployment

1. **Test the Backend**: Visit `https://your-backend-url.com/api/health` - it should return `{"status":"ok"}`

2. **Configure Frontend**:
   - Open your GitHub Pages site
   - Expand "Backend Configuration" section
   - Enter your backend URL
   - Click "Test Connection"
   - You should see "✓ Backend connected successfully!"

3. **Use the App**: Now you can upload images and process OCR!

---

## Troubleshooting

### CORS Errors
- Make sure `flask-cors` is installed: `pip install flask-cors`
- The backend should automatically allow all origins (configured in `app.py`)

### 404 Errors
- Ensure the backend URL is correct (no trailing slash)
- Check that the backend is actually running
- Verify the `/api/health` endpoint works

### Timeout Errors
- OCR processing can take 30-60 seconds
- Some platforms have request timeouts (Render: 30s, Railway: no limit)
- Consider upgrading if you need longer processing times

### Module Not Found
- Ensure all dependencies in `requirements.txt` are installed
- Some platforms require specific Python versions (check `runtime.txt`)

---

## Environment Variables

Some platforms allow you to set environment variables:

- `PORT`: Server port (usually auto-detected)
- `FLASK_ENV`: Set to `production` for production mode

---

## Cost Comparison

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| Railway | ❌ No | $5/month+ |
| Render | ✅ Yes (with limits) | $7/month+ |
| Heroku | ❌ No | $5/month+ |
| PythonAnywhere | ✅ Yes (limited) | $5/month+ |

**Recommendation**: Start with Render's free tier, upgrade if needed.

---

## Need Help?

1. Check the backend logs on your deployment platform
2. Test the `/api/health` endpoint directly
3. Verify all dependencies are installed
4. Check that Python version matches `runtime.txt` (3.11.0)
