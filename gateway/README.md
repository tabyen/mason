# Home gateway

Three containers that give any OpenAI- or Anthropic-compatible client one endpoint in front
of an AWS account's Bedrock models.

| Container | Port | Job |
|---|---|---|
| `litellm` | 4000 (localhost) | The gateway. Routes Grok and Claude to Bedrock Runtime, and OpenAI models to the translator. |
| `mantle-translator` | internal only | Fronts Bedrock Mantle: AWS auth, Chat Completions to Responses translation, reasoning carry-over. |
| `catalog` | 8090 (localhost) | Serves the model list with effort levels and pricing, and whether the gateway currently offers each model. |

## Run it

    cp .env.example .env          # set AWS_PROFILE, regions, and the two keys
    docker compose up -d --build
    curl -s localhost:8090/v1/catalog | head        # or open http://localhost:8090
    curl -s localhost:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
      -H "Content-Type: application/json" \
      -d '{"model":"gpt-5.6-luna","messages":[{"role":"user","content":"Say hi"}]}'

If your AWS credentials live in aws-vault, the mounted profile cannot work inside the
containers; start the stack with `aws-vault/up.sh` instead. See `aws-vault/README.md`, which
also holds the IAM role this was set up with.

AWS side: the profile needs `bedrock:InvokeModel` (including on the default project ARN for
OpenAI and Grok models), and model access enabled in the Bedrock console for each model.
OpenAI models are on Mantle only in us-east-1, us-east-2 and us-west-2 (Sol: not us-west-2).

`tests/chain_test.sh` runs mason through LiteLLM and the translator against a fake Mantle,
with no AWS account and no Docker. It is the quickest way to see the pieces working.

## Clients

All three wire formats work against every model in `litellm/config.yaml`.

**Strands / mason** (Chat Completions): base URL `http://localhost:4000/v1`, key = `LITELLM_MASTER_KEY`.

**Codex** (Responses), in `~/.codex/config.toml`:

    model = "gpt-5.6-terra"
    model_provider = "mason"

    [model_providers.mason]
    name = "Mason gateway"
    base_url = "http://localhost:4000/v1"
    env_key = "LITELLM_MASTER_KEY"
    wire_api = "responses"

**Claude Code** (Anthropic Messages):

    export ANTHROPIC_BASE_URL=http://localhost:4000
    export ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY
    export ANTHROPIC_MODEL=claude-sonnet-5-5          # or gpt-5.6-terra, grok-4.7, ...
    export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1   # lists the gateway's models in /model

## The translator

LiteLLM can reach Mantle on its own (`bedrock_mantle/` provider), and Mantle accepts Chat
Completions directly. The translator exists for three things those paths do not give you:

- **Responses upstream for every client.** On Mantle, prompt caching and reasoning summaries
  are Responses-only. Chat Completions clients get them because the translator converts.
- **Reasoning carry-over.** A Chat Completions client cannot send reasoning items back, so
  the model starts cold after each tool call. The translator keeps the encrypted reasoning
  items in memory by tool-call id and re-attaches them. Set `CARRY_REASONING=0` to turn it off.
- **AWS credentials stay in one place.** It mints short-lived Bedrock bearer tokens from the
  mounted profile (or uses `MANTLE_API_KEY`).

Set `UPSTREAM_API=chat` to relay Chat Completions untranslated if the Responses path gives
you trouble. `/v1/responses` requests (Codex) are always relayed as they are. Sampling
parameters (`temperature`, `top_p`) are dropped unless `PASS_SAMPLING=1`.

The reasoning store is per process and in memory: run one translator replica, and expect a
cold tool call after a restart.

## The catalog

`GET /v1/catalog` returns `mason.catalog/v1`:

    {"schema": "mason.catalog/v1", "gateway": "home", "as_of": "2026-11-16",
     "gateway_reachable": true, "unlisted": [],
     "models": [{
       "id": "gpt-5.6-terra",                  # what clients send
       "lab": "openai",
       "upstream": {"provider": "bedrock-mantle", "model_id": "openai.gpt-5.6-terra", "via": "translator"},
       "apis": ["chat", "responses", "messages"],
       "context_window": 1050000, "max_output_tokens": 128000,
       "effort": {"param": "reasoning_effort", "levels": ["none","low","medium","high","xhigh"], "default": "medium"},
       "pricing": {"currency": "USD", "unit": "1M tokens", "input": 2.2, "output": 13.2,
                   "cache_read": 0.22, "cache_write": 2.75, "long_context": {...}, "source": "...", "as_of": "..."},
       "available": true                       # live: does the gateway list this id right now
     }]}

The facts live in `catalog/models.toml`, which is mounted into the container and re-read on
change. Prices were read from the AWS model cards on the `as_of` date and will drift; each
entry links its source. `unlisted` names models the gateway serves that the catalog lacks.

mason only needs `id`, `effort.levels`, `pricing.input/output/cache_read` and `available`,
so a catalog at work can be much smaller than this and still plug in.

## Using it from mason, at home and at work

Point mason at a gateway by name; see `[gateways.*]` in `../crosslab.example.toml`. Anything
that speaks Chat Completions works as a gateway, with or without a catalog, so the stack at
work only has to match on base URL and model names, and those are per-gateway settings.
