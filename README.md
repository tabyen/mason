# mason

Home of `crosslab`, a cross-lab coding orchestrator.

Orchestrates GPT-5.6 Terra, GPT-5.6 Luna and Grok 4.7 through a LiteLLM proxy:
Luna scouts, Terra plans and implements, Grok critiques the plan and reviews the diff,
and your own check commands decide what "working" means.

## Setup

    pip install 'strands-agents[openai]'        # Python 3.10+ (3.10 also needs: pip install tomli)
    cp crosslab.example.toml ~/.config/crosslab/config.toml   # set base_url and model ids
    export LITELLM_API_KEY_MYPROFILE=...        # or token_command; see the example config
    ./smoke_test.sh                             # optional: full pipeline against a fake LLM

## Use

    # existing repo: Terra drives, Grok reviews
    crosslab.py "Add retry with backoff to the S3 uploader" --check "pytest -q" --check "ruff check ."

    # hard task: Terra and Grok each implement, the other reviews, best one wins
    crosslab.py task.md --mode duel --check "pytest -q"

    # routine task: Luna tries first, Terra takes over only if Luna's result is not clean
    crosslab.py "Rename Foo.bar to Foo.baz everywhere" --mode cheap --check "pytest -q"

    # greenfield
    crosslab.py spec.md --repo ~/src/newthing --init --check "pytest -q"

Other flags: `--no-plan` (small obvious tasks), `--no-scout`, `--aws-profile`, `--config`.

The result is a branch `crosslab/<run>-<model>` in a separate worktree; your working tree
is never touched. The exit code is 0 only if checks pass and no blocking review issue
remains. Plans, critiques, check logs and reviews are kept under `~/crosslab-runs/<repo>/<run>/`.

## Things to know

- Agents work from HEAD. Uncommitted changes are invisible to them.
- The driver can run shell commands in the worktree. File tools are confined to the
  worktree, but the shell is not a sandbox. Secret-looking environment variables
  (AWS_*, *KEY*, *TOKEN*, ...) are removed from the driver's shell; your check commands
  keep everything except the LiteLLM tokens.
- With no `checks`, the only gate is the review. Always give it at least the test command.
- Greenfield work has no tests yet; put the test command in `--check` anyway (the driver
  is told what will be run and writes tests to match), and read the tests it wrote.
