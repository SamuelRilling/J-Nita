<<<<<<< HEAD
# OCR5 Quick Start Guide

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
python app.py
```

Or use the run script:

```bash
python run.py
```

The backend will start on `http://localhost:5000`

### 3. Open the Frontend

**Option A: Direct File**
- Simply open `index.html` in your web browser
- Note: Update `API_BASE` in `index.html` if needed

**Option B: Local Server**
```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

## Using the Application

1. **Upload Image**: Click the upload area or drag and drop an image
2. **Condition Image**: Click "Condition Image" to preprocess
3. **Run OCR**: Click "Run OCR" to extract text
4. **Configure**: Expand configuration section to adjust settings

## Deployment

### GitHub Pages (Frontend Only)

1. Push to GitHub
2. Go to Settings > Pages
3. Select source branch (usually `main`)
4. Frontend will be available at `https://yourusername.github.io/OCR5/`

**Note**: You'll need a separate backend service for the API.

### Full Stack Deployment

#### Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. The app will be available at `https://your-app-name.herokuapp.com`

#### Railway

1. Connect your GitHub repository
2. Railway will auto-detect Python
3. Set start command: `python run.py`
4. Deploy automatically

#### Render

1. Create new Web Service
2. Connect GitHub repository
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python run.py`
4. Deploy

## Configuration

### Backend URL

If deploying frontend and backend separately, update the `API_BASE` variable in `index.html`:

```javascript
const API_BASE = 'https://your-backend-url.herokuapp.com';
```

## Troubleshooting

### Backend won't start
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Check port 5000 is available

### CORS errors
- Ensure `flask-cors` is installed
- Backend should allow requests from your frontend domain

### OCR API errors
- Requires internet connection
- Hugging Face API must be accessible
- Some models may be temporarily unavailable

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Set to `development` for debug mode
=======
# OCR5 Quick Start Guide

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
python app.py
```

Or use the run script:

```bash
python run.py
```

The backend will start on `http://localhost:5000`

### 3. Open the Frontend

**Option A: Direct File**
- Simply open `index.html` in your web browser
- Note: Update `API_BASE` in `index.html` if needed

**Option B: Local Server**
```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

## Using the Application

1. **Upload Image**: Click the upload area or drag and drop an image
2. **Condition Image**: Click "Condition Image" to preprocess
3. **Run OCR**: Click "Run OCR" to extract text
4. **Configure**: Expand configuration section to adjust settings

## Deployment

### GitHub Pages (Frontend Only)

1. Push to GitHub
2. Go to Settings > Pages
3. Select source branch (usually `main`)
4. Frontend will be available at `https://yourusername.github.io/OCR5/`

**Note**: You'll need a separate backend service for the API.

### Full Stack Deployment

#### Heroku

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. The app will be available at `https://your-app-name.herokuapp.com`

#### Railway

1. Connect your GitHub repository
2. Railway will auto-detect Python
3. Set start command: `python run.py`
4. Deploy automatically

#### Render

1. Create new Web Service
2. Connect GitHub repository
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python run.py`
4. Deploy

## Configuration

### Backend URL

If deploying frontend and backend separately, update the `API_BASE` variable in `index.html`:

```javascript
const API_BASE = 'https://your-backend-url.herokuapp.com';
```

## Troubleshooting

### Backend won't start
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Check port 5000 is available

### CORS errors
- Ensure `flask-cors` is installed
- Backend should allow requests from your frontend domain

### OCR API errors
- Requires internet connection
- Hugging Face API must be accessible
- Some models may be temporarily unavailable

## Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Set to `development` for debug mode
>>>>>>> 3998df6f2eb9ed25e696e30ca04cacd75174931a
