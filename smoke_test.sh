#!/usr/bin/env bash
# Runs the whole pipeline against fake_llm.py on a scratch repo. No tokens, no network.
# Usage: ./smoke_test.sh [standard|duel|cheap]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mode="${1:-standard}"
tmp="$(mktemp -d)"
port=4010

python3 "$here/fake_llm.py" "$port" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
sleep 1

mkdir "$tmp/repo" && cd "$tmp/repo"
git init -q
git config user.email smoke@localhost && git config user.name smoke
printf 'from calc import add\nassert add(2, 3) == 5, add(2, 3)\nprint("ok")\n' > test_calc.py
echo "# scratch" > README.md
echo "notes" > NOTES.md
git add -A && git commit -qm init

cat > "$tmp/smoke.toml" <<EOF
base_url = "http://127.0.0.1:$port/v1"
workdir = "$tmp/runs"
checks = ["python3 test_calc.py"]
[models.terra]
id = "fake-terra"
[models.luna]
id = "fake-luna"
[models.grok]
id = "fake-grok"
EOF

LITELLM_API_KEY=fake python3 "$here/crosslab.py" "Add calc.add(a, b)" \
  --repo "$tmp/repo" --config "$tmp/smoke.toml" --mode "$mode"
echo "smoke test passed ($mode); scratch files in $tmp"
