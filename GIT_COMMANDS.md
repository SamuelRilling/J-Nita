# Git Commands Guide for J-Nita

## Initial Setup (Already Done)

✅ Git repository initialized
✅ Remote repository configured: `https://github.com/samuelrilling/J-Nita.git`
✅ Initial commit created

## Daily Workflow

### 1. Check Status
```bash
cd "F:\Escritorio\OCR5"
git status
```

### 2. Stage Changes
```bash
# Stage all changes
git add .

# Or stage specific files
git add index.html app.py
```

### 3. Commit Changes
```bash
git commit -m "Your descriptive commit message"
```

### 4. Push to GitHub
```bash
git push origin main
```

## Common Workflow Example

```bash
cd "F:\Escritorio\OCR5"

# See what changed
git status

# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Fix CORS errors and update backend configuration UI"

# Push to GitHub
git push origin main
```

## Viewing Changes

```bash
# See what changed
git diff

# See commit history
git log --oneline

# See changes in a specific file
git diff index.html
```

## Useful Commands

### Check Current Branch
```bash
git branch
```

### See Remote Configuration
```bash
git remote -v
```

### Pull Latest Changes (if working on multiple machines)
```bash
git pull origin main
```

### Undo Last Commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Undo Last Commit (discard changes)
```bash
git reset --hard HEAD~1
```

## First Time Push

If you haven't pushed before, you might need to:

```bash
git push -u origin main
```

After the first push, you can just use:
```bash
git push
```

## Commit Message Guidelines

Write clear, descriptive commit messages:

✅ Good:
- "Fix OpenCV dependency for Railway deployment"
- "Add backend configuration UI with connection testing"
- "Rename project to J-Nita v5.0"
- "Update CORS configuration for GitHub Pages"

❌ Avoid:
- "fix"
- "update"
- "changes"
- "asdf"

## Authentication

### HTTPS (Current)
Uses username/password or personal access token.

### SSH (Alternative)
If you want to use SSH instead:
```bash
git remote set-url origin git@github.com:samuelrilling/J-Nita.git
```

Then you'll need to set up SSH keys on GitHub.
