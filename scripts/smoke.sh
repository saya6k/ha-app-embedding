#!/bin/sh
# ==============================================================================
# Full-container verification: boots the app image under s6 and checks the
# embeddings API the way its clients use it.
#
#   sh scripts/smoke.sh [image-tag]
#
# CI passes the tag it just built. With no argument the image is built here
# first, which is what you want locally. Run it wherever `docker` works; on
# this Mac that is socktainer on Apple `container` (aarch64 only):
#
#   sh <repo>/scripts/smoke.sh
#
# Env:
#   CACHE_DIR  where the GGUF model and container /data live (default below).
#              Keeping it outside the repo means repeat runs skip the ~296 MiB
#              download.
#   DOCKER_NET  extra `docker` network flags (default --network=host, for
#              hosts whose dockerd has no bridge; socktainer accepts it too)
# ==============================================================================
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "${SCRIPT_DIR}")"
APP="${REPO}/embedding"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/ha-app-embedding-smoke}"
DOCKER_NET="${DOCKER_NET:---network=host}"
DATA="${CACHE_DIR}/app-data"
MODEL="embeddinggemma-2-Q8_0.gguf"
KEY="smoke-test-key"
URL="http://127.0.0.1:8080"
IMAGE="${1:-}"
NAME=embedtest

fail() {
    echo "!! FAIL: $*"
    docker logs "${NAME}" 2>&1 | tail -20
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    exit 1
}

# POST a JSON body to /v1/embeddings from inside the container; prints the
# HTTP status on the first line and the body after it.
embed() {
    docker exec -i "${NAME}" sh -c "curl -s -w '\n%{http_code}' --max-time 120 \
        -H 'Content-Type: application/json' $2 \
        -d @- ${URL}/v1/embeddings" <<EOF | tail -1
$1
EOF
}

embed_body() {
    docker exec -i "${NAME}" sh -c "curl -s --max-time 120 \
        -H 'Content-Type: application/json' -H 'Authorization: Bearer ${KEY}' \
        -d @- ${URL}/v1/embeddings"
}

if [ -n "${IMAGE}" ]; then
    echo "########## 1. using prebuilt image ${IMAGE} ##########"
else
    IMAGE=ha-app-embedding:test
    echo "########## 1. docker build ##########"
    # shellcheck disable=SC2086
    docker build ${DOCKER_NET} -t "${IMAGE}" "${APP}"
fi

echo
echo "########## 2. prepare /data ##########"
mkdir -p "${DATA}/models"
cp "${SCRIPT_DIR}/ci-options.json" "${DATA}/options.json"
# Reuse a previously downloaded model if present; otherwise the app's own
# first-boot download path runs (and is itself part of what we verify).
if [ -f "${CACHE_DIR}/${MODEL}" ] && [ ! -f "${DATA}/models/${MODEL}" ]; then
    cp "${CACHE_DIR}/${MODEL}" "${DATA}/models/"
fi
ls -l "${DATA}" "${DATA}/models"

echo
echo "########## 3. run container ##########"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "${NAME}" ${DOCKER_NET} -v "${DATA}:/data" "${IMAGE}"

echo "waiting for /health on :8080 (first boot downloads the model) …"
ok=0
i=0
while [ "${i}" -lt 900 ]; do
    if docker exec "${NAME}" curl -sf --max-time 2 "${URL}/health" >/dev/null 2>&1; then
        ok=1; break
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "${NAME}")" != "true" ]; then
        echo "!! container exited early"; break
    fi
    sleep 2
    i=$((i + 1))
done

echo
echo "########## 4. container logs ##########"
docker logs "${NAME}" 2>&1 | tail -30
[ "${ok}" -eq 1 ] || fail "the server never became healthy"

# Keep the model for the next run.
[ -f "${CACHE_DIR}/${MODEL}" ] || cp "${DATA}/models/${MODEL}" "${CACHE_DIR}/"

echo
echo "########## 5. process ##########"
echo "-- llama-server's argv (the API key must NOT appear in it) --"
argv="$(docker exec "${NAME}" sh -c \
    'for p in /proc/[0-9]*; do c=$(tr "\0" " " < "$p/cmdline" 2>/dev/null); case "$c" in *llama-server\ --model*) echo "$c"; break;; esac; done')"
echo "${argv}"
case "${argv}" in
    *"${KEY}"*) fail "the API key is visible in the process list" ;;
    *"--host 0.0.0.0 --port 8080"*) echo "OK: listening on 0.0.0.0:8080" ;;
    *) fail "unexpected listen address" ;;
esac

echo
echo "########## 6. authentication ##########"
body='{"input":"hello"}'
code="$(embed "${body}" "")"
[ "${code}" = "401" ] || fail "no key answered HTTP ${code}, want 401"
echo "OK: no key -> 401"
code="$(embed "${body}" "-H 'Authorization: Bearer wrong'")"
[ "${code}" = "401" ] || fail "wrong key answered HTTP ${code}, want 401"
echo "OK: wrong key -> 401"
code="$(embed "${body}" "-H 'Authorization: Bearer ${KEY}'")"
[ "${code}" = "200" ] || fail "right key answered HTTP ${code}, want 200"
echo "OK: right key -> 200"

echo
echo "########## 7. embeddings by meaning (EmbeddingGemma 2 prefixes) ##########"
# A Korean query must land nearer the English description it means than an
# unrelated one: the same check a tool or memory search relies on.
result="$(embed_body <<'EOF'
{"input": [
  "task: search result | query: 거실 불 좀 꺼줘",
  "title: none | text: Turn off the lights in an area of the home",
  "title: none | text: Show the balance of a bank account"
]}
EOF
)"
dims="$(printf '%s' "${result}" | jq '[.data[].embedding | length] | unique')"
echo "dimensions: ${dims}"
[ "${dims}" = "$(printf '[\n  768\n]')" ] || fail "expected 768-dimensional vectors"
# DOCS.md promises unit-length vectors, so a dot product is a cosine.
printf '%s' "${result}" | jq -e \
    '[.data[].embedding | map(. * .) | add | sqrt | . - 1 | fabs < 0.001] | all' >/dev/null \
    || fail "vectors are not normalised to unit length"
echo "OK: vectors have unit length"
scores="$(printf '%s' "${result}" | jq -c '
    def dot(a; b): [a, b] | transpose | map(.[0] * .[1]) | add;
    def cos(a; b): dot(a; b) / ((dot(a; a) | sqrt) * (dot(b; b) | sqrt));
    [.data[].embedding] as $v
    | {lights: cos($v[0]; $v[1]), bank: cos($v[0]; $v[2])}')"
echo "cosine: ${scores}"
printf '%s' "${scores}" | jq -e '.lights > .bank' >/dev/null \
    || fail "the query is not nearest the text it means"
echo "OK: the Korean query is nearest the lights description"

echo
echo "########## 8. long input ##########"
# Near context_size (1024 in ci-options.json): a long input fits in one go.
long="$(printf 'Turn off the lights in the living room. %.0s' $(seq 1 110))"
tokens="$(printf '{"input":"%s"}' "${long}" | embed_body | jq '.usage.prompt_tokens // .error')"
echo "a ~1000-token input: prompt_tokens=${tokens}"
case "${tokens}" in
    ''|*[!0-9]*) fail "a long input was refused: ${tokens}" ;;
esac
toolong="$(printf 'Turn off the lights in the living room. %.0s' $(seq 1 180))"
err="$(printf '{"input":"%s"}' "${toolong}" | embed_body | jq -c '.error // empty')"
[ -n "${err}" ] || fail "an input over context_size was accepted (truncated?)"
echo "OK: an input over context_size is refused: ${err}"

echo
echo "########## 9. memory ##########"
docker exec "${NAME}" sh -c \
    'for p in /proc/[0-9]*; do case "$(tr "\0" " " < "$p/cmdline" 2>/dev/null)" in *llama-server\ --model*) grep -E "VmHWM|VmRSS" "$p/status";; esac; done'

echo
echo "########## 10. cleanup ##########"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
echo "ALL OK"
