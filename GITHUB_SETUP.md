# GitHub Setup Instructions

## Issue: Backend Configuration Not Visible

The "Backend Configuration" section exists in the code, but you need to **commit and push** the updated `index.html` file to GitHub.

## Steps to Fix

### 1. Commit and Push Updated Files

```bash
cd F:\Escritorio\OCR5
git add .
git commit -m "Add backend configuration UI and fix Railway deployment"
git push origin main
```

Or use GitHub Desktop:
1. Open GitHub Desktop
2. Stage all changes
3. Commit with message: "Add backend configuration UI and fix Railway deployment"
4. Push to origin

### 2. Wait for GitHub Pages to Update

- GitHub Pages updates automatically after push
- May take 1-2 minutes
- Refresh your site: `https://samuelrilling.github.io/OCR5/`

### 3. Verify Backend Configuration Section

After the page updates, you should see:
- **🔗 Backend Configuration** section at the top (below the action buttons)
- Input field for backend URL
- "Test Connection" button

## Railway Build Fix

I've removed `runtime.txt` which was causing the Railway build error. Railway auto-detects Python from `requirements.txt`.

### Re-deploy to Railway

1. Go to Railway dashboard
2. Your deployment should auto-redeploy, OR
3. Click "Redeploy" button
4. Build should succeed now

## Files Changed

✅ `index.html` - Added Backend Configuration section
✅ Removed `runtime.txt` - Was causing Railway build errors
✅ Added `railway.json` - Railway deployment config
✅ Added `.python-version` - For local development only

## After Pushing to GitHub

1. **Frontend**: Refresh GitHub Pages site - you'll see "Backend Configuration" section
2. **Backend**: Redeploy on Railway - build should succeed
3. **Connect**: Enter Railway URL in "Backend Configuration" section
4. **Test**: Click "Test Connection" button
