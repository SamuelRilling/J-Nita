# Render Deployment Fix

## Problem

Render was trying to use `gunicorn` but it wasn't installed:
```
bash: line 1: gunicorn: command not found
```

## Solution

Two options:

### Option 1: Use Gunicorn (Recommended for Production)

**Add gunicorn to requirements.txt** (already done ✅):
```
gunicorn>=21.2.0
```

**Start Command in Render Dashboard:**
```
gunicorn app:app --bind 0.0.0.0:$PORT
```

### Option 2: Use Python Directly (Simpler)

**Start Command in Render Dashboard:**
```
python run.py
```

No need for gunicorn if using this option.

## Recommended: Use Gunicorn

Gunicorn is better for production because:
- ✅ More stable for production workloads
- ✅ Handles multiple requests better
- ✅ Recommended by Flask for production

Since `gunicorn` is now in `requirements.txt`, just use:
```
gunicorn app:app --bind 0.0.0.0:$PORT
```

## Finding Your Render Backend URL

1. Go to: https://dashboard.render.com/web/srv-d5l9613e5dus73cqtje0
2. Look at the top of the page - you'll see your service name
3. The URL format is: `https://[service-name].onrender.com`
4. Common format: `https://j-nita.onrender.com` or `https://j-nita-backend.onrender.com`

### How to Get the Exact URL:

1. Open your Render dashboard
2. Click on your service (the one you're deploying)
3. In the top section, you'll see "URL" or "Service URL"
4. Copy that URL - that's your backend API URL!

Example: If your service is named `j-nita`, your URL will be:
```
https://j-nita.onrender.com
```

## Configure Frontend

Once you have your Render URL:

1. Open: `https://samuelrilling.github.io/J-Nita/`
2. Go to "🔗 Backend Configuration" section
3. Enter: `https://[your-service-name].onrender.com`
4. Click "Test Connection"

## Test Your Backend

Visit: `https://[your-service-name].onrender.com/api/health`

Should return: `{"status":"ok"}`
