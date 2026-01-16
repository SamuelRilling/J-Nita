# Backend Setup Guide - Quick Fix

## Problem

You're seeing CORS errors because:
1. The frontend is trying to connect to a placeholder URL
2. You need to configure your actual backend URL

## Solution in 3 Steps

### Step 1: Deploy Backend (if not already done)

**Easiest - Railway:**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Select **J-Nita** repository
5. Wait for deployment (build should succeed with `opencv-python-headless`)
6. Copy the URL (e.g., `https://j-nita-production.up.railway.app`)

### Step 2: Configure Frontend

1. Open your GitHub Pages site: `https://samuelrilling.github.io/J-Nita/`
2. **Scroll down** to find "🔗 Backend Configuration" section (it's expanded by default)
3. **Clear the placeholder** in the input field (if any)
4. **Paste your Railway URL** (e.g., `https://j-nita-production.up.railway.app`)
5. **Click "Test Connection"** button

### Step 3: Verify

You should see:
- ✅ "✓ Backend connected successfully! You can now use the app."

If you see an error:
- ❌ Check that the backend URL is correct (no trailing slash)
- ❌ Check that the backend is running (visit the URL directly)
- ❌ Make sure the URL starts with `https://`

## Troubleshooting

### Still seeing "your-backend-url.herokuapp.com" errors?

1. **Clear browser cache** or use incognito mode
2. **Clear localStorage**: Open browser console and run:
   ```javascript
   localStorage.removeItem('jnita_backend_url');
   location.reload();
   ```
3. **Enter your backend URL** in the Backend Configuration section

### CORS errors persist?

1. **Backend must be deployed** and running
2. **URL must be correct** (check Railway dashboard)
3. **Backend must have CORS enabled** (already configured in `app.py`)

### Backend not responding?

1. Check Railway logs for errors
2. Visit `https://your-backend-url.railway.app/api/health` directly
   - Should return: `{"status":"ok"}`
3. If 404 or error, redeploy the backend

## Visual Guide

```
1. Open: https://samuelrilling.github.io/J-Nita/
                    ↓
2. Find: 🔗 Backend Configuration (below action buttons)
                    ↓
3. Enter: https://j-nita-production.up.railway.app
                    ↓
4. Click: Test Connection
                    ↓
5. See: ✓ Backend connected successfully!
```

## Still Need Help?

1. Check that your backend is deployed and running
2. Verify the backend URL in Railway dashboard
3. Make sure you're using the correct URL format (https://...)
