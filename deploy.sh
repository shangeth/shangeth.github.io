#!/bin/bash
set -e

cd academic-portfolio-astro
npm run build
cd ..

git add docs/ academic-portfolio-astro/
git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M')"
git push origin master
