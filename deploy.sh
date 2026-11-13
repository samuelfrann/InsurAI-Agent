#!/usr/bin/env bash
set -e
DEST=.hf-deploy

cp requirements.txt app.py ingest.py create_user.py "$DEST"/
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.bak*' \
  backend/ "$DEST"/backend/

cd "$DEST"
git add -A
git commit -m "${1:-deploy}" || echo "Nothing to commit"
git push
