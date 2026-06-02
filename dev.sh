#!/bin/zsh
# Local dev server. Pass port as $1 (default 1313).
# Hugo best practice: explicitly set --baseURL to the localhost URL so
# generated <a href>, canonical, og:url stay local instead of jumping to
# the production domain (config.yml has baseURL: 'https://pillumina.github.io').

set -e

PORT="${1:-1313}"
BASE_URL="http://localhost:${PORT}/"

echo "[dev] Hugo server on ${BASE_URL}"

exec hugo server \
    --port "${PORT}" \
    --baseURL "${BASE_URL}" \
    --buildDrafts \
    --buildFuture \
    --disableFastRender
