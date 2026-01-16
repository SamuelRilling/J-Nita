<<<<<<< HEAD
# Fixing CORS Errors - Solution

## Problem

When deploying to GitHub Pages, you were seeing CORS errors because:
1. The frontend was trying to connect to a placeholder backend URL (`your-backend-url.herokuapp.com`)
2. This URL doesn't exist, causing 404 and CORS errors

## Solution Applied

The frontend has been updated to:

1. **Backend Configuration Section**: Added a new "Backend Configuration" section at the top of the page
2. **Dynamic URL Input**: You can now enter your backend URL directly in the web interface
3. **Connection Testing**: Click "Test Connection" to verify your backend is working
4. **Local Storage**: Your backend URL is saved in browser localStorage
5. **Better Error Handling**: Clear error messages when backend is not configured

## How to Fix Your Deployment

### Step 1: Deploy the Backend

Choose one of these options (see [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md) for details):

**Easiest - Railway:**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Select your OCR5 repository
5. Railway auto-deploys
6. Copy the URL (e.g., `https://ocr5-production.up.railway.app`)

**Free Option - Render:**
1. Go to [render.com](https://render.com)
2. Sign up
3. New Web Service → Connect GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python run.py`
5. Deploy and copy the URL

### Step 2: Configure Frontend

1. Open your GitHub Pages site: `https://samuelrilling.github.io/OCR5/`
2. Scroll to "Backend Configuration" section (at the top)
3. Enter your backend URL (e.g., `https://ocr5-production.up.railway.app`)
4. Click "Test Connection"
5. You should see: "✓ Backend connected successfully!"

### Step 3: Use the App

Now you can:
- Upload images
- Condition images
- Run OCR processing

## Testing

After configuring:

1. **Health Check**: Visit `https://your-backend-url.com/api/health`
   - Should return: `{"status":"ok"}`

2. **Frontend Test**: Click "Test Connection" in the web app
   - Should show success message

3. **Full Test**: Upload an image and try conditioning
   - Should work without CORS errors

## Troubleshooting

### Still seeing CORS errors?

1. **Check backend URL**: Make sure it's correct (no trailing slash)
2. **Test backend directly**: Visit `https://your-backend-url.com/api/health`
3. **Check backend logs**: Look for errors in Railway/Render dashboard
4. **Verify flask-cors**: Make sure `flask-cors` is in requirements.txt (it is)

### Backend not responding?

1. **Check deployment logs**: Look for errors during build/deploy
2. **Verify dependencies**: All packages should install successfully
3. **Check Python version**: Should match runtime.txt (3.11.0)
4. **Test locally**: Run `python app.py` locally to verify it works

### Connection test fails?

1. **Backend URL format**: Should be `https://...` (not `http://`)
2. **No trailing slash**: Remove any trailing `/` from URL
3. **Backend is running**: Check deployment platform dashboard
4. **CORS enabled**: Backend should allow all origins (configured in app.py)

## Files Changed

- `index.html`: Added backend configuration UI and dynamic URL handling
- `DEPLOY_BACKEND.md`: Comprehensive backend deployment guide
- `render.yaml`: Render deployment configuration

## Next Steps

1. Deploy backend to Railway or Render
2. Configure frontend with backend URL
3. Test the full workflow
4. Share your working app! 🎉
=======
# Fixing CORS Errors - Solution

## Problem

When deploying to GitHub Pages, you were seeing CORS errors because:
1. The frontend was trying to connect to a placeholder backend URL (`your-backend-url.herokuapp.com`)
2. This URL doesn't exist, causing 404 and CORS errors

## Solution Applied

The frontend has been updated to:

1. **Backend Configuration Section**: Added a new "Backend Configuration" section at the top of the page
2. **Dynamic URL Input**: You can now enter your backend URL directly in the web interface
3. **Connection Testing**: Click "Test Connection" to verify your backend is working
4. **Local Storage**: Your backend URL is saved in browser localStorage
5. **Better Error Handling**: Clear error messages when backend is not configured

## How to Fix Your Deployment

### Step 1: Deploy the Backend

Choose one of these options (see [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md) for details):

**Easiest - Railway:**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Select your OCR5 repository
5. Railway auto-deploys
6. Copy the URL (e.g., `https://ocr5-production.up.railway.app`)

**Free Option - Render:**
1. Go to [render.com](https://render.com)
2. Sign up
3. New Web Service → Connect GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python run.py`
5. Deploy and copy the URL

### Step 2: Configure Frontend

1. Open your GitHub Pages site: `https://samuelrilling.github.io/OCR5/`
2. Scroll to "Backend Configuration" section (at the top)
3. Enter your backend URL (e.g., `https://ocr5-production.up.railway.app`)
4. Click "Test Connection"
5. You should see: "✓ Backend connected successfully!"

### Step 3: Use the App

Now you can:
- Upload images
- Condition images
- Run OCR processing

## Testing

After configuring:

1. **Health Check**: Visit `https://your-backend-url.com/api/health`
   - Should return: `{"status":"ok"}`

2. **Frontend Test**: Click "Test Connection" in the web app
   - Should show success message

3. **Full Test**: Upload an image and try conditioning
   - Should work without CORS errors

## Troubleshooting

### Still seeing CORS errors?

1. **Check backend URL**: Make sure it's correct (no trailing slash)
2. **Test backend directly**: Visit `https://your-backend-url.com/api/health`
3. **Check backend logs**: Look for errors in Railway/Render dashboard
4. **Verify flask-cors**: Make sure `flask-cors` is in requirements.txt (it is)

### Backend not responding?

1. **Check deployment logs**: Look for errors during build/deploy
2. **Verify dependencies**: All packages should install successfully
3. **Check Python version**: Should match runtime.txt (3.11.0)
4. **Test locally**: Run `python app.py` locally to verify it works

### Connection test fails?

1. **Backend URL format**: Should be `https://...` (not `http://`)
2. **No trailing slash**: Remove any trailing `/` from URL
3. **Backend is running**: Check deployment platform dashboard
4. **CORS enabled**: Backend should allow all origins (configured in app.py)

## Files Changed

- `index.html`: Added backend configuration UI and dynamic URL handling
- `DEPLOY_BACKEND.md`: Comprehensive backend deployment guide
- `render.yaml`: Render deployment configuration

## Next Steps

1. Deploy backend to Railway or Render
2. Configure frontend with backend URL
3. Test the full workflow
4. Share your working app! 🎉
>>>>>>> 3998df6f2eb9ed25e696e30ca04cacd75174931a
