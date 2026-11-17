#!/usr/bin/env python3
"""crosslab: orchestrate GPT-5.6 Terra, GPT-5.6 Luna and Grok 4.7 through a LiteLLM proxy.

Pipeline (each stage is a fresh Strands agent unless noted):

  1. scout     Luna x3 in parallel, read-only   -> context brief
  2. plan      Terra drafts, Grok attacks, Terra revises
  3. implement the driver edits files in an isolated git worktree
  4. checks    YOUR commands (tests, types, lint), run by this script, not by a model;
               failures go back to the driver
  5. review    the other lab reviews spec + diff in a fresh context and must give
               concrete failing cases; blockers go back to the driver

Modes: standard (Terra drives, Grok reviews), duel (Terra and Grok each implement,
the other reviews, best candidate wins), cheap (Luna drives first, Terra takes over
only if Luna's result fails checks or review).

Nothing touches your working tree: results land on a branch `crosslab/<run>-<model>`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

from strands import Agent, tool
from strands.models.openai import OpenAIModel

# --------------------------------------------------------------------------- config

DEFAULTS = {
    "base_url": "http://localhost:4000/v1",  # LiteLLM proxy, OpenAI-compatible
    "aws_profile": "",                       # default: $AWS_PROFILE
    "token_command": "",                     # e.g. "my-token-helper --profile {profile}"
    "checks": [],                            # e.g. ["pytest -q", "ruff check ."]
    "check_timeout_s": 900,
    "command_timeout_s": 300,
    "max_fix_rounds": 3,                     # driver attempts to get checks green
    "review_rounds": 2,                      # reviews per candidate
    "tool_call_budget": 200,                 # per agent, stops runaway loops
    "request_timeout_s": 900,
    "workdir": "~/crosslab-runs",
    "shell_env_passthrough": [],             # secret-looking vars the driver's shell may keep
    "models": {
        # `id` must match the model names your LiteLLM proxy exposes.
        "terra": {"id": "gpt-5.6-terra", "params": {}},
        "luna": {"id": "gpt-5.6-luna", "params": {}},
        "grok": {"id": "grok-4.7", "params": {}},
    },
    "roles": {
        "scout": "luna", "planner": "terra", "critic": "grok",
        "driver": "terra", "rival": "grok", "cheap_driver": "luna",
    },
    # Who reviews whose work. Always the other lab.
    "reviewer_for": {"terra": "grok", "luna": "grok", "grok": "terra"},
}


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(explicit: str | None, repo: Path) -> dict:
    candidates = [explicit] if explicit else [repo / "crosslab.toml", "~/.config/crosslab/config.toml"]
    for c in candidates:
        p = Path(str(c)).expanduser()
        if p.is_file():
            log(f"config: {p}")
            return merge(DEFAULTS, tomllib.loads(p.read_text()))
    if explicit:
        sys.exit(f"config file not found: {explicit}")
    log("config: built-in defaults (no crosslab.toml found)")
    return dict(DEFAULTS)


def token_for(profile: str, cfg: dict) -> str:
    """LiteLLM token for an AWS profile: per-profile env var, then helper command, then generic env var."""
    if profile:
        env_name = "LITELLM_API_KEY_" + re.sub(r"[^A-Za-z0-9]", "_", profile).upper()
        if os.environ.get(env_name):
            return os.environ[env_name]
        if cfg["token_command"]:
            cmd = cfg["token_command"].format(profile=shlex.quote(profile))
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            sys.exit(f"token_command failed for profile {profile!r}: {r.stderr.strip()[:300]}")
    else:
        env_name = "LITELLM_API_KEY_<PROFILE>"
    if os.environ.get("LITELLM_API_KEY"):
        return os.environ["LITELLM_API_KEY"]
    sys.exit(f"No LiteLLM token for AWS profile {profile or '(none)'}: set {env_name}, "
             "LITELLM_API_KEY, or token_command in crosslab.toml.")


# --------------------------------------------------------------------------- helpers

_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sh(cmd: list[str] | str, cwd: Path, timeout: int = 120, env: dict | None = None,
       check: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, errors="replace", timeout=timeout, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {r.stderr.strip()[:500]}")
    return r


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return sh(["git", *args], cwd, check=check).stdout.strip()


def clip(text: str, limit: int, keep: str = "tail") -> str:
    if len(text) <= limit:
        return text
    note = f"\n[... {len(text) - limit} characters omitted ...]\n"
    return (note + text[-limit:]) if keep == "tail" else (text[:limit] + note)


SECRETISH = re.compile(r"(^AWS_|^LITELLM|KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


def scrubbed_env(cfg: dict, everything: bool) -> dict:
    """Environment for subprocesses. Model-issued commands lose anything secret-looking;
    your own check commands only lose the LiteLLM tokens."""
    keep = set(cfg["shell_env_passthrough"])
    pattern = SECRETISH if everything else re.compile(r"^LITELLM", re.I)
    env = {k: v for k, v in os.environ.items() if k in keep or not pattern.search(k)}
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")  # no stale .pyc between rapid edits, none in the diff
    return env


# --------------------------------------------------------------------------- tools

def make_tools(root: Path, cfg: dict, write: bool) -> list:
    """File and shell tools confined to `root`. Read-only roles get no write or shell tools."""
    root = root.resolve()
    calls = {"n": 0}

    def spend() -> str | None:
        calls["n"] += 1
        if calls["n"] > cfg["tool_call_budget"]:
            return "TOOL BUDGET EXHAUSTED. Stop calling tools and give your final answer now."
        return None

    def safe(path: str) -> Path:
        p = (root / path).resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path escapes the repository: {path}")
        return p

    @tool
    def list_files(pattern: str = "") -> str:
        """List repository files (tracked and untracked, .gitignore respected).

        Args:
            pattern: Optional glob matched against the relative path, e.g. "src/*.py" or "*test*".
        """
        if (m := spend()):
            return m
        files = git(root, "ls-files", "-co", "--exclude-standard").splitlines()
        if pattern:
            files = [f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f, f"*{pattern}*")]
        extra = f"\n[... {len(files) - 500} more; narrow the pattern]" if len(files) > 500 else ""
        return "\n".join(files[:500]) + extra or "(no files)"

    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
        """Read a text file with line numbers.

        Args:
            path: Path relative to the repository root.
            start_line: First line to return (1-based).
            max_lines: Maximum number of lines to return.
        """
        if (m := spend()):
            return m
        try:
            lines = safe(path).read_text(errors="replace").splitlines()
        except (OSError, ValueError) as e:
            return f"ERROR: {e}"
        chunk = lines[start_line - 1:start_line - 1 + max_lines]
        body = "\n".join(f"{i:>5}  {l}" for i, l in enumerate(chunk, start_line))
        return f"{body}\n[{path}: lines {start_line}-{start_line + len(chunk) - 1} of {len(lines)}]"

    @tool
    def search(pattern: str, path: str = ".") -> str:
        """Search file contents with an extended regular expression (git grep).

        Args:
            pattern: Extended regex to search for.
            path: Directory or file to limit the search to, relative to the repository root.
        """
        if (m := spend()):
            return m
        r = sh(["git", "grep", "-n", "-I", "-E", "--untracked", "-e", pattern, "--", path], root)
        out = r.stdout.splitlines()
        extra = f"\n[... {len(out) - 200} more matches]" if len(out) > 200 else ""
        return "\n".join(l[:300] for l in out[:200]) + extra or "(no matches)"

    if not write:
        return [list_files, read_file, search]

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the given content.

        Args:
            path: Path relative to the repository root.
            content: Full new content of the file.
        """
        if (m := spend()):
            return m
        try:
            p = safe(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except (OSError, ValueError) as e:
            return f"ERROR: {e}"
        return f"wrote {path} ({len(content.splitlines())} lines)"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a file. Fails if `old` is absent or ambiguous.

        Args:
            path: Path relative to the repository root.
            old: Exact text to replace; include enough context to be unique.
            new: Replacement text.
        """
        if (m := spend()):
            return m
        try:
            p = safe(path)
            text = p.read_text()
        except (OSError, ValueError) as e:
            return f"ERROR: {e}"
        n = text.count(old)
        if n != 1:
            return f"ERROR: `old` occurs {n} times in {path}; it must occur exactly once."
        p.write_text(text.replace(old, new, 1))
        return f"edited {path}"

    @tool
    def run_command(command: str) -> str:
        """Run a shell command from the repository root (build, tests, linters, scripts).

        Args:
            command: The shell command line.
        """
        if (m := spend()):
            return m
        try:
            r = sh(command, root, timeout=cfg["command_timeout_s"], env=scrubbed_env(cfg, everything=True))
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {cfg['command_timeout_s']}s"
        return f"exit code {r.returncode}\n{clip(r.stdout + r.stderr, 8000)}"

    return [list_files, read_file, search, write_file, edit_file, run_command]


# --------------------------------------------------------------------------- prompts

SCOUT = """You are a SCOUT preparing a context brief for another engineer who will change this \
repository. You have read-only tools. Answer only the question you are given, from what you \
actually read. Cite file paths and line numbers. Say "not found" rather than guessing. \
Keep the answer under 400 words."""

SCOUT_ANGLES = [
    "Which files, functions and data structures are relevant to the task? List each with its path and a one-line role.",
    "How is this project built, tested, linted and type-checked? Give exact commands and where they are defined. "
    "Note the test layout and the conventions a new change should follow (style, naming, error handling).",
    "What existing code does something similar that should be reused or imitated, and what could this task "
    "break (callers, public interfaces, migrations, config)?",
]

PLANNER = """You are the PLANNER. Write an implementation plan for the task in this repository. \
You have read-only tools; verify claims in the brief before relying on them. The plan must list: \
the approach in two or three sentences, each file to create or change and what changes, the \
tests to add or update, and risks or open questions. Do not write the code itself. If the task \
is ambiguous, state the interpretation you chose."""

CRITIC = """You are the CRITIC. Another engineer wrote the plan below. Find how it fails: wrong \
assumptions about the code (verify with your read-only tools), missed callers or edge cases, \
missing tests, a simpler approach it overlooked, or a misreading of the task. Number each \
finding, mark it BLOCKING or MINOR, and say what to do instead. Do not rewrite the plan and do \
not praise it. If nothing is blocking, say "NO BLOCKING ISSUES" and list any minor points."""

DRIVER = """You are the DRIVER. Implement the task in this repository, using your tools. Paths \
are relative to the repository root. Rules:
- Read the relevant code before changing it, and follow the conventions already in use.
- Make the smallest change that fully solves the task; add or update tests for the behaviour you change.
- Run the checks yourself with run_command before you finish, and fix what fails.
- Never weaken, skip or delete a test to make it pass, and do not edit the check commands.
- Do not commit, switch branches or touch git history.
When done, reply with a short summary: what changed, and anything you could not verify."""

REVIEWER = """You are the REVIEWER. You did not write this change and you see it cold: the task \
and the diff. Use your read-only tools to read the surrounding code as it now stands. Look for \
bugs, unhandled edge cases, broken callers, missing or weak tests, and places where the diff \
does not do what the task asks. Every issue needs a concrete failing case: an input, call or \
scenario that goes wrong. Skip style opinions. Severity: "blocker" (wrong or breaks something), \
"major" (likely bug or untested core behaviour), "minor" (worth noting, not worth another round).

Reply with ONLY a JSON object:
{"verdict": "approve" | "request_changes",
 "issues": [{"severity": "blocker|major|minor", "file": "path", "problem": "...", "failing_case": "..."}]}"""


# --------------------------------------------------------------------------- orchestrator

@dataclass
class Candidate:
    model: str                 # model key: terra / luna / grok
    root: Path                 # git worktree
    branch: str
    checks_ok: bool = False
    check_log: str = ""
    blocking: list = field(default_factory=list)
    minor: list = field(default_factory=list)
    summary: str = ""
    error: str = ""
    reviewed: bool = True

    @property
    def score(self) -> tuple:
        return (not self.error, self.checks_ok, -len(self.blocking))

    @property
    def clean(self) -> bool:
        return not self.error and self.checks_ok and not self.blocking


class Orchestrator:
    def __init__(self, cfg: dict, repo: Path, task: str, run_id: str):
        self.cfg, self.repo, self.task, self.run_id = cfg, repo, task, run_id
        self.run_dir = Path(cfg["workdir"]).expanduser() / repo.name / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base = git(repo, "rev-parse", "HEAD")
        self.usage: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._tokens: dict[str, str] = {}

    # ---- model plumbing

    def model(self, key: str) -> OpenAIModel:
        m = self.cfg["models"][key]
        profile = m.get("aws_profile") or self.cfg["aws_profile"] or os.environ.get("AWS_PROFILE", "")
        with self._lock:
            if profile not in self._tokens:
                self._tokens[profile] = token_for(profile, self.cfg)
        extra = {"stream": m["stream"]} if "stream" in m else {}
        return OpenAIModel(
            client_args={"api_key": self._tokens[profile], "base_url": m.get("base_url") or self.cfg["base_url"],
                         "timeout": self.cfg["request_timeout_s"], "max_retries": 4},
            model_id=m["id"], params=m.get("params") or None, **extra)

    def agent(self, key: str, system: str, root: Path, write: bool = False) -> Agent:
        agent = Agent(model=self.model(key), system_prompt=system, callback_handler=None,
                      tools=make_tools(root, self.cfg, write))
        agent._crosslab_model = key  # for usage accounting
        return agent

    def ask(self, agent: Agent, prompt: str) -> str:
        result = agent(prompt)
        used = result.metrics.accumulated_usage or {}
        with self._lock:
            u = self.usage.setdefault(agent._crosslab_model, {"in": 0, "out": 0})
            # accumulated_usage is cumulative per agent; record the delta
            prev = getattr(agent, "_crosslab_seen", {"in": 0, "out": 0})
            now = {"in": used.get("inputTokens", 0), "out": used.get("outputTokens", 0)}
            u["in"] += now["in"] - prev["in"]
            u["out"] += now["out"] - prev["out"]
            agent._crosslab_seen = now
        return str(result).strip()

    def save(self, name: str, text: str) -> None:
        (self.run_dir / name).write_text(text)

    # ---- stages 1 and 2: scout, plan

    def scout(self) -> str:
        tracked = git(self.repo, "ls-files").splitlines()
        if len(tracked) < 3:
            log("scout: skipped (near-empty repository, treating as greenfield)")
            return "(Greenfield: the repository is essentially empty. Choose a conventional layout.)"
        key = self.cfg["roles"]["scout"]
        log(f"scout: {len(SCOUT_ANGLES)} x {key} in parallel")

        def one(angle: str) -> str:
            try:
                return self.ask(self.agent(key, SCOUT, self.repo), f"Task:\n{self.task}\n\nQuestion: {angle}")
            except Exception as e:  # one scout failing should not sink the run
                return f"(scout failed: {e})"

        with cf.ThreadPoolExecutor(len(SCOUT_ANGLES)) as pool:
            answers = list(pool.map(one, SCOUT_ANGLES))
        brief = "\n\n".join(f"### {a}\n{ans}" for a, ans in zip(SCOUT_ANGLES, answers))
        self.save("brief.md", brief)
        return brief

    def plan(self, brief: str) -> str:
        roles = self.cfg["roles"]
        log(f"plan: {roles['planner']} drafts")
        planner = self.agent(roles["planner"], PLANNER, self.repo)
        checks = "\n".join(f"- {c}" for c in self.cfg["checks"]) or "(none configured)"
        draft = self.ask(planner, f"Task:\n{self.task}\n\nContext brief from scouts (unverified):\n{brief}\n\n"
                                  f"Checks the result must pass:\n{checks}")
        self.save("plan-draft.md", draft)
        log(f"plan: {roles['critic']} critiques")
        critique = self.ask(self.agent(roles["critic"], CRITIC, self.repo),
                            f"Task:\n{self.task}\n\nPlan:\n{draft}")
        self.save("plan-critique.md", critique)
        if "NO BLOCKING ISSUES" in critique.upper() and "BLOCKING:" not in critique.upper():
            final = draft + f"\n\nMinor points from review:\n{critique}"
        else:
            log(f"plan: {roles['planner']} revises")
            final = self.ask(planner, "A reviewer from another team critiqued your plan. Check each finding "
                                      "against the code, accept the ones that are right, and reply with the "
                                      f"complete revised plan only.\n\nCritique:\n{critique}")
        self.save("plan.md", final)
        return final

    # ---- stages 3 to 5: implement, check, review

    def worktree(self, key: str) -> Candidate:
        branch = f"crosslab/{self.run_id}-{key}"
        root = self.run_dir / f"wt-{key}"
        git(self.repo, "worktree", "add", "-q", "-b", branch, str(root), self.base)
        return Candidate(model=key, root=root, branch=branch)

    def run_checks(self, c: Candidate) -> None:
        if not self.cfg["checks"]:
            c.checks_ok, c.check_log = True, "(no checks configured)"
            return
        logs, ok = [], True
        for cmd in self.cfg["checks"]:
            try:
                r = sh(cmd, c.root, timeout=self.cfg["check_timeout_s"], env=scrubbed_env(self.cfg, everything=False))
                out, code = r.stdout + r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                out, code = f"timed out after {self.cfg['check_timeout_s']}s", -1
            ok = ok and code == 0
            logs.append(f"$ {cmd}\nexit code {code}\n{clip(out, 6000)}")
        c.checks_ok, c.check_log = ok, "\n\n".join(logs)

    def green_loop(self, c: Candidate, driver: Agent, tag: str) -> None:
        """Run checks; feed failures to the driver until green or out of rounds."""
        for attempt in range(1, self.cfg["max_fix_rounds"] + 2):
            self.run_checks(c)
            self.save(f"{c.model}-checks-{tag}-{attempt}.log", c.check_log)
            log(f"[{c.model}] checks {'PASS' if c.checks_ok else 'FAIL'} ({tag}, attempt {attempt})")
            if c.checks_ok or attempt > self.cfg["max_fix_rounds"]:
                return
            self.ask(driver, "The checks were run for you and FAILED. Find the cause and fix the code "
                             f"(not the tests, unless a test is itself wrong).\n\n{c.check_log}")

    def diff(self, c: Candidate) -> str:
        git(c.root, "add", "-A")
        return git(c.root, "diff", "--cached", self.base)

    def review(self, c: Candidate, n: int) -> None:
        key = self.cfg["reviewer_for"][c.model]
        diff = self.diff(c)
        if not diff:
            c.blocking = [{"severity": "blocker", "file": "", "problem": "The driver made no changes.",
                           "failing_case": "The task is not implemented."}]
            return
        log(f"[{c.model}] review {n} by {key}")
        reviewer = self.agent(key, REVIEWER, c.root)
        raw = self.ask(reviewer, f"Task:\n{self.task}\n\nDiff against the base commit:\n```diff\n"
                                 f"{clip(diff, 80000, keep='head')}\n```")
        verdict = parse_json(raw)
        if verdict is None:
            raw = self.ask(reviewer, "Reply again with ONLY the JSON object described in your instructions.")
            verdict = parse_json(raw)
        self.save(f"{c.model}-review-{n}.json", json.dumps(verdict, indent=2) if verdict else raw)
        if verdict is None:
            log(f"[{c.model}] review {n}: could not parse the reviewer's reply; see the run directory")
            c.minor.append({"severity": "minor", "file": "", "problem": "Review reply was unparseable.",
                            "failing_case": ""})
            c.blocking = []
            return
        issues = [i for i in verdict.get("issues", []) if isinstance(i, dict)]
        c.blocking = [i for i in issues if str(i.get("severity", "")).lower() in ("blocker", "major")]
        c.minor = [i for i in issues if i not in c.blocking]
        log(f"[{c.model}] review {n}: {len(c.blocking)} blocking, {len(c.minor)} minor")

    def build(self, key: str, brief: str, plan: str) -> Candidate:
        """One candidate, start to finish: implement -> checks -> review -> fix -> re-check."""
        c = self.worktree(key)
        try:
            log(f"[{key}] implementing in {c.root}")
            driver = self.agent(key, DRIVER, c.root, write=True)
            checks = "\n".join(f"- {x}" for x in self.cfg["checks"]) or "(none configured; write and run your own tests)"
            c.summary = self.ask(driver, f"Task:\n{self.task}\n\nPlan (reviewed; deviate if the code proves it "
                                         f"wrong, and say so):\n{plan}\n\nContext brief from scouts (unverified):\n"
                                         f"{brief}\n\nChecks that will be run on your work:\n{checks}")
            self.green_loop(c, driver, "impl")
            if not c.checks_ok:
                log(f"[{key}] checks still failing after {self.cfg['max_fix_rounds']} fix rounds; skipping review")
                c.reviewed = False
            for n in range(1, self.cfg["review_rounds"] + 1 if c.checks_ok else 0):
                self.review(c, n)
                if not c.blocking or n == self.cfg["review_rounds"]:
                    break
                c.summary = self.ask(
                    driver, "An independent reviewer found the issues below. For each one, reproduce the "
                            "failing case, then fix it and add a test, or explain briefly why it is not a "
                            f"real problem.\n\n{json.dumps(c.blocking, indent=2)}")
                self.green_loop(c, driver, f"fix{n}")
        except Exception as e:
            c.error = f"{type(e).__name__}: {e}"
            log(f"[{key}] FAILED: {c.error}")
        self.commit(c)
        return c

    def commit(self, c: Candidate) -> None:
        git(c.root, "add", "-A")
        if not git(c.root, "diff", "--cached", "--name-only"):
            return
        ident = [] if git(c.root, "config", "user.email", check=False) else \
            ["-c", "user.name=crosslab", "-c", "user.email=crosslab@localhost"]
        first = self.task.strip().splitlines()[0][:60]
        git(c.root, *ident, "commit", "-q", "--no-verify", "-m",
            f"crosslab({c.model}): {first}\n\nRun {self.run_id}. Review before merging.")

    # ---- top level

    def run(self, mode: str, do_scout: bool, do_plan: bool) -> Candidate:
        roles = self.cfg["roles"]
        brief = self.scout() if do_scout else "(scouting skipped)"
        plan = self.plan(brief) if do_plan else "(planning skipped; plan as you go)"
        if mode == "duel":
            keys = [roles["driver"], roles["rival"]]
            log(f"duel: {keys[0]} vs {keys[1]}")
            with cf.ThreadPoolExecutor(2) as pool:
                cands = list(pool.map(lambda k: self.build(k, brief, plan), keys))
            winner = max(cands, key=lambda c: c.score)  # ties go to the primary driver
            log(f"duel: winner is {winner.model}")
        elif mode == "cheap":
            winner = self.build(roles["cheap_driver"], brief, plan)
            cands = [winner]
            if not winner.clean:
                log(f"cheap: {winner.model} did not finish clean, escalating to {roles['driver']}")
                second = self.build(roles["driver"], brief, plan)
                cands.append(second)
                winner = max(cands[::-1], key=lambda c: c.score)  # ties go to the stronger model
        else:
            winner = self.build(roles["driver"], brief, plan)
            cands = [winner]
        self.report(winner, cands)
        return winner

    def report(self, w: Candidate, cands: list[Candidate]) -> None:
        stat = git(self.repo, "diff", "--stat", self.base, w.branch, check=False)
        lines = [
            "", "=" * 72,
            f"RESULT: {'CLEAN' if w.clean else 'NEEDS ATTENTION'}  (driver: {w.model}, run {self.run_id})",
            f"  checks:   {'pass' if w.checks_ok else 'FAIL'}" + ("" if self.cfg["checks"] else "  (none configured!)"),
            f"  review:   {len(w.blocking)} blocking, {len(w.minor)} minor issue(s) outstanding"
            if w.reviewed else "  review:   not run (checks never passed)",
        ]
        if w.error:
            lines.append(f"  error:    {w.error}")
        lines += [f"  branch:   {w.branch}", f"  worktree: {w.root}", f"  run dir:  {self.run_dir}", "", stat, ""]
        for i in w.blocking + w.minor:
            lines.append(f"  [{i.get('severity')}] {i.get('file', '')}: {i.get('problem', '')}")
            if i.get("failing_case"):
                lines.append(f"      case: {i['failing_case']}")
        if len(cands) > 1:
            lines += ["", "Candidates:"] + [
                f"  {c.model:6} checks={'pass' if c.checks_ok else 'fail'} blocking={len(c.blocking)} "
                f"branch={c.branch}" + (f" error={c.error}" if c.error else "") for c in cands]
        lines += ["", "Driver's summary:", w.summary or "(none)", "", "Tokens (input / output):"]
        lines += [f"  {k:6} {u['in']:>10,} / {u['out']:>9,}" for k, u in sorted(self.usage.items())]
        lines += ["", "Next:", f"  git diff {self.base[:10]}..{w.branch}",
                  f"  git merge --squash {w.branch}    # when you are satisfied",
                  f"  git worktree remove --force {w.root}    # clean up", "=" * 72]
        text = "\n".join(lines)
        self.save("report.txt", text)
        print(text)


def parse_json(text: str) -> dict | None:
    """Pull the review verdict out of a reply that may wrap it in prose or a code fence."""
    for candidate in [text] + re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) + \
            ([text[text.find("{"):text.rfind("}") + 1]] if "{" in text else []):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "issues" in obj:
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-lab coding orchestrator (Terra / Luna / Grok via LiteLLM).")
    ap.add_argument("task", help="the task, a path to a file containing it, or - for stdin")
    ap.add_argument("--repo", default=".", help="repository to work on (default: current directory)")
    ap.add_argument("--mode", choices=["standard", "duel", "cheap"], default="standard")
    ap.add_argument("--check", action="append", default=[], metavar="CMD",
                    help="check command; repeatable; overrides `checks` in the config")
    ap.add_argument("--config", help="path to a crosslab.toml")
    ap.add_argument("--aws-profile", help="AWS profile whose LiteLLM token to use (default: $AWS_PROFILE)")
    ap.add_argument("--no-scout", action="store_true", help="skip the scouting stage")
    ap.add_argument("--no-plan", action="store_true", help="skip plan and critique (small, obvious tasks)")
    ap.add_argument("--init", action="store_true", help="greenfield: create the repo and an empty first commit if needed")
    a = ap.parse_args()

    repo = Path(a.repo).expanduser().resolve()
    if a.init:
        repo.mkdir(parents=True, exist_ok=True)
        if sh(["git", "rev-parse", "--git-dir"], repo).returncode != 0:
            git(repo, "init", "-q")
        if sh(["git", "rev-parse", "HEAD"], repo).returncode != 0:
            ident = [] if git(repo, "config", "user.email", check=False) else \
                ["-c", "user.name=crosslab", "-c", "user.email=crosslab@localhost"]
            git(repo, *ident, "commit", "-q", "--allow-empty", "-m", "Initial commit")
    if not repo.is_dir() or sh(["git", "rev-parse", "HEAD"], repo).returncode != 0:
        sys.exit(f"{repo} is not a git repository with at least one commit (use --init for greenfield work).")
    repo = Path(git(repo, "rev-parse", "--show-toplevel"))

    task = sys.stdin.read() if a.task == "-" else (Path(a.task).read_text() if os.path.isfile(a.task) else a.task)
    if not task.strip():
        sys.exit("empty task")

    cfg = load_config(a.config, repo)
    if a.check:
        cfg["checks"] = a.check
    if a.aws_profile:
        cfg["aws_profile"] = a.aws_profile
    if not cfg["checks"]:
        log("WARNING: no check commands configured. Executable checks are the main quality lever; "
            "pass --check 'pytest -q' (or similar) or set `checks` in crosslab.toml.")
    if git(repo, "status", "--porcelain"):
        log("WARNING: uncommitted changes in your working tree are NOT visible to the agents; "
            "they work from HEAD in a separate worktree.")

    orch = Orchestrator(cfg, repo, task, time.strftime("%m%d-%H%M%S"))
    log(f"run {orch.run_id}: mode={a.mode} repo={repo} base={orch.base[:10]}")
    winner = orch.run(a.mode, do_scout=not a.no_scout, do_plan=not a.no_plan)
    return 0 if winner.clean else 1


if __name__ == "__main__":
    sys.exit(main())
