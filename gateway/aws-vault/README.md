# Running the gateway with aws-vault

For a host where AWS credentials live in aws-vault (macOS keychain, MFA) instead of plain
`~/.aws` files. The base `docker-compose.yml` mounts `~/.aws` and picks `AWS_PROFILE`, which
cannot work here: a profile backed by aws-vault needs aws-vault and the keychain, and
neither exists inside a container.

## Run it

    cp .env.example .env      # in gateway/; set the two keys (AWS_PROFILE is not used here)
    aws-vault/up.sh           # whole stack; asks for an MFA code if the session has lapsed
    aws-vault/up.sh --no-deps litellm     # just LiteLLM
    docker compose down       # stop; the plain base file is enough for this

`up.sh` runs `docker compose up -d` inside `aws-vault exec bedrock-invoke`, with
`docker-compose.aws-vault.yml` layered on top to pass the temporary credentials into the
`litellm` and `mantle-translator` containers.

The credentials are fixed when the containers start and last 12 hours. After that Bedrock
calls fail with an expired-token error until you run `up.sh` again. Nothing refreshes them.

## The `bedrock-invoke` role

Created 2026-11-17 from the two policy files here. They are templates: replace `ACCOUNT_ID`
and `IAM_USER` with your AWS account ID and IAM user name before running these.

    aws-vault exec admin -- aws iam create-role --role-name bedrock-invoke \
      --max-session-duration 43200 \
      --assume-role-policy-document file://trust-policy.json
    aws-vault exec admin -- aws iam put-role-policy --role-name bedrock-invoke \
      --policy-name bedrock-invoke --policy-document file://permissions-policy.json

- `trust-policy.json`: the `IAM_USER` user may assume it, with MFA.
- `permissions-policy.json`: invoke any Bedrock foundation model, directly or through an
  inference profile in this account, and list models and inference profiles. Nothing else.

It has no Marketplace permissions on purpose, which has one consequence: see below.

## `~/.aws/config`

`IAM_USER` here is the aws-vault profile holding your IAM user's keys, and `MFA_DEVICE` is
the name of its MFA device.

    [profile bedrock-invoke]
    source_profile = IAM_USER
    mfa_serial = arn:aws:iam::ACCOUNT_ID:mfa/MFA_DEVICE
    role_arn = arn:aws:iam::ACCOUNT_ID:role/bedrock-invoke
    region = us-east-1

    # For LiteLLM run directly on the host (aws_profile_name: litellm-bedrock). Not usable
    # from a container.
    [profile litellm-bedrock]
    credential_process = /opt/homebrew/bin/aws-vault export --format=json --prompt=osascript --duration=12h bedrock-invoke
    region = us-east-1

## Model access: accept each model's agreement once

Bedrock enables a model on first use only if the caller may subscribe on AWS Marketplace,
and `bedrock-invoke` may not. Until someone with those permissions accepts the model's
agreement for the account, calls fail with:

    Model access is denied due to IAM user or service role is not authorized to perform
    the required AWS Marketplace actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)

Watch out: the first call or two to an unsubscribed model can succeed before the denials
start, so one passing request proves nothing. Check the status instead:

    aws-vault exec admin -- aws bedrock get-foundation-model-availability \
      --model-id anthropic.claude-opus-5-5 --query agreementAvailability.status

To accept (as `admin`; takes about two minutes to go from PENDING to AVAILABLE):

    M=anthropic.claude-opus-5-5
    TOK=$(aws-vault exec admin -- aws bedrock list-foundation-model-agreement-offers \
            --model-id $M --query 'offers[0].offerToken' --output text)
    aws-vault exec admin -- aws bedrock create-foundation-model-agreement --model-id $M --offer-token "$TOK"

The role's calls can keep being denied for a few minutes after the status reads AVAILABLE
(about three minutes for Sonnet and Haiku). Wait and retry before suspecting anything else.

Status on 2026-11-17:

| Model | Agreement |
|---|---|
| `anthropic.claude-opus-5-5` | AVAILABLE |
| `xai.grok-4.7` | AVAILABLE |
| `anthropic.claude-sonnet-5-5` | AVAILABLE |
| `anthropic.claude-haiku-5-5` | AVAILABLE |

## What has been tested

- `up.sh --no-deps litellm`, then Chat Completions to `claude-opus-5-5`: works.
- Direct Converse calls with the role to the `global.` Sonnet 5.5 and Haiku 5.5 profiles,
  repeated after their agreements were accepted: work.
- Earlier, against a standalone LiteLLM container with the same role: Chat Completions,
  streaming and `/v1/messages` to `us.anthropic.claude-opus-5-5`.
- Not tested: the translator and the OpenAI models on Mantle. The role's policy covers
  `foundation-model/*` and `inference-profile/*` only; the gateway README says Mantle also
  needs `bedrock:InvokeModel` on the default project ARN, which this policy does not grant.
