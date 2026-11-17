#!/usr/bin/env bash
# End-to-end test without AWS or Docker:
#   mason -> LiteLLM (real, local process) -> translator -> fake Bedrock Mantle
# plus the catalog, and one request in each client wire format (Codex, Claude Code).
# Needs: pip install 'litellm[proxy]' 'strands-agents[openai]' aiohttp aws-bedrock-token-generator
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; gw="$here/.."; root="$gw/.."
tmp="$(mktemp -d)"; pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
export LITELLM_MASTER_KEY=sk-test TRANSLATOR_API_KEY=tr-test TRANSLATOR_URL=http://127.0.0.1:8081/v1 AWS_REGION=us-east-1

python3 "$here/fake_mantle.py" 4020 & pids+=($!)
MANTLE_BASE_URL=http://127.0.0.1:4020/openai/v1 MANTLE_API_KEY=bedrock-api-key-fake PORT=8081 \
  python3 "$gw/translator/translator.py" >"$tmp/translator.log" 2>&1 & pids+=($!)
litellm --config "$gw/litellm/config.yaml" --port 4001 >"$tmp/litellm.log" 2>&1 & pids+=($!)
CATALOG_FILE="$gw/catalog/models.toml" GATEWAY_URL=http://127.0.0.1:4001 GATEWAY_KEY=$LITELLM_MASTER_KEY PORT=8091 \
  python3 "$gw/catalog/catalog.py" & pids+=($!)
for i in $(seq 60); do curl -sf -H "Authorization: Bearer sk-test" localhost:4001/v1/models >/dev/null && break; sleep 1; done

echo "== catalog"
curl -sf localhost:8091/v1/catalog | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('  reachable:', d['gateway_reachable'], ' unlisted:', d['unlisted'])
for m in d['models']: print('  %-18s available=%s' % (m['id'], m['available']))"

echo "== Codex wire format (/v1/responses)"
curl -sf localhost:4001/v1/responses -H "Authorization: Bearer sk-test" -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-terra","input":"hi","reasoning":{"effort":"low"}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('  ', [c['text'] for o in d['output'] if o['type']=='message' for c in o['content']])"

echo "== Claude Code wire format (/v1/messages)"
curl -sf localhost:4001/v1/messages -H "Authorization: Bearer sk-test" -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"gpt-5.6-terra","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('  ', d['content'])"

echo "== mason, full pipeline"
mkdir "$tmp/repo" && cd "$tmp/repo" && git init -q
git config user.email t@localhost && git config user.name t
printf 'from calc import add\nassert add(2, 3) == 5, add(2, 3)\nprint("ok")\n' > test_calc.py
echo "# scratch" > README.md; echo notes > NOTES.md; git add -A; git commit -qm init
cat > "$tmp/mason.toml" <<TOML
gateway = "chain"
workdir = "$tmp/runs"
checks = ["python3 test_calc.py"]
[gateways.chain]
base_url = "http://127.0.0.1:4001/v1"
catalog_url = "http://127.0.0.1:8091"
api_key_env = "LITELLM_MASTER_KEY"
[gateways.chain.models.terra]
effort = "high"
[gateways.chain.models.grok]
id = "gpt-5.6-sol"      # Grok needs real AWS; route the reviewer through the translator too
TOML
python3 "$root/crosslab.py" "Add calc.add(a, b)" --repo "$tmp/repo" --config "$tmp/mason.toml" 2>&1 \
  | grep -E "gateway:|checks |review [0-9]:|RESULT|terra |luna |grok |total estimate|WARNING"

echo "== what the fake upstream saw"
curl -sf localhost:4020/stats | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('  requests:', d['requests'], ' bad auth:', d['bad_auth'], ' reasoning items carried back:', d['reasoning_items_returned'])
print('  efforts seen:', sorted({str(e) for e in d['efforts']}), ' models:', sorted(set(d['models'])))"
echo "chain test finished; logs in $tmp"
