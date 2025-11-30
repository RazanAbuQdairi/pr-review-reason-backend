from pathlib import Path
import csv
from typing import List, Tuple, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# -------------------------------------------------
# FastAPI app + CORS
# -------------------------------------------------
app = FastAPI(title="PR Review Reason Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Canonical 9 labels (from the paper)
# -------------------------------------------------
LABEL_SPEC_INTENT = "1) Specification / Intent Mismatch"
LABEL_LOGIC = "2) Logic / Semantic Defects"
LABEL_BUILD_CI = "3) Build / CI / Environment Failures"
LABEL_STYLE = "4) Style / Convention Violations"
LABEL_TESTING = "5) Testing Inadequacy (Missing, Weak, or Incorrect Tests)"
LABEL_DESIGN = "6) Architectural / Design Misfit"
LABEL_PROCESS = "7) Process / Policy Violations (Governance Gates)"
LABEL_TOOLING = "8) Tool-Use / Automation Errors"
LABEL_OTHER = "9) Other / Unclear or Not Defect-Related"

ALL_LABELS = [
    LABEL_SPEC_INTENT,
    LABEL_LOGIC,
    LABEL_BUILD_CI,
    LABEL_STYLE,
    LABEL_TESTING,
    LABEL_DESIGN,
    LABEL_PROCESS,
    LABEL_TOOLING,
    LABEL_OTHER,
]

# -------------------------------------------------
# Human-readable explanations per label
# -------------------------------------------------
LABEL_EXPLANATIONS = {
    LABEL_SPEC_INTENT: (
        "Cases where the contributor misunderstood what problem should be solved. "
        "The change does not faithfully implement the user story, issue description, "
        "API contract, or design spec. The code may be internally consistent, but "
        "it implements the wrong or only a partial requirement."
    ),
    LABEL_LOGIC: (
        "The task is understood correctly, but the algorithm, control flow, or data "
        "handling is wrong. Includes off-by-one errors, wrong conditions, broken "
        "invariants, mishandled edge cases, and incorrect assumptions that lead to "
        "wrong results for some inputs."
    ),
    LABEL_BUILD_CI: (
        "The main issue is that the patch cannot be reliably built, tested, or executed "
        "in the project environment. Includes CI failures, dependency drift, platform "
        "differences, flaky / non-deterministic jobs, and missing or broken build scripts."
    ),
    LABEL_STYLE: (
        "Concerns how the code is written and organized, not what it does. Includes "
        "formatting, naming, linting, and local conventions that are enforced by "
        "maintainers and automated tools for readability and consistency."
    ),
    LABEL_TESTING: (
        "Defects in the test layer: missing tests, weak tests, or incorrect tests. "
        "The implementation may be reasonable, but there is not enough or not the "
        "right kind of testing to trust the change (coverage gaps, no regression tests, etc.)."
    ),
    LABEL_DESIGN: (
        "Changes that technically work but conflict with the system’s architectural "
        "principles or long-term maintainability. Includes tight coupling, layering "
        "violations, bad abstractions, duplicated functionality, and performance "
        "misfits (e.g., slow algorithms in hot paths)."
    ),
    LABEL_PROCESS: (
        "Violations of repository governance rules, independent of technical correctness. "
        "Examples: missing CLA/DCO, wrong target branch, missing changelog entry, required "
        "status checks or approvals not satisfied, or templates not followed."
    ),
    LABEL_TOOLING: (
        "Failures caused by misusing project tools and automation rather than business logic. "
        "Examples: running generators with wrong flags, skipping required pre-commit hooks, "
        "stale caches, wrong test runner/command, or inconsistent generated artifacts."
    ),
    LABEL_OTHER: (
        "Comments that are unclear, social (thanks / chit-chat), or do not fit any of the "
        "eight defect-related categories. Often non-blocking remarks or general discussion."
    ),
}

# -------------------------------------------------
# Helper: deduplicate and normalize keyword lists
# -------------------------------------------------
def make_keywords(*terms: str) -> List[str]:
    """
    Build a sorted, unique, lower-cased list of keywords.
    This automatically removes duplicates and keeps the lists clean.
    """
    uniq = {t.strip().lower() for t in terms if t and t.strip()}
    return sorted(uniq)


# -------------------------------------------------
# Rule-based keywords per label (deduplicated)
# -------------------------------------------------
KEYWORDS = {
    LABEL_SPEC_INTENT: make_keywords(
        # core taxonomy words
        "spec", "specs", "specification", "specifications",
        "requirement", "requirements", "req",
        "ticket", "tickets", "story", "stories", "user story",
        "issue", "issues", "backlog",
        "scope", "scoping",
        "contract", "api contract", "interface", "protocol",
        "acceptance criteria",
        "business rule", "business rules",
        # mismatch / misalignment
        "spec mismatch", "specification mismatch",
        "requirements mismatch", "requirement mismatch",
        "wrong requirement", "wrong feature",
        "wrong behavior", "expected behavior", "expected result",
        "unexpected behavior", "functional mismatch",
        "out of scope", "out-of-scope",
        "misaligned", "misalignment",
        "inconsistent with spec",
        "inconsistent with ticket",
        "inconsistent with story",
        "docs out of date",
        "documentation does not match behavior",
        # partial coverage
        "partially addresses", "partial fix",
        "partial implementation", "incomplete feature",
        "missing requirement", "missing behavior",
        "does not cover all cases", "doesn't cover case",
        "not fully implemented",
        # interpretation problems
        "misinterpreted requirement", "misinterpreted the ticket",
        "misinterpreted the story", "misinterpreted spec",
        "misunderstood requirement", "misunderstood the spec",
        "incorrect interpretation of spec",
        "incorrect interpretation of ticket",
        "misread ticket", "misread story",
        # compatibility / regressions
        "backward compatibility issue",
        "breaking public api contract",
        "regression vs previous behavior",
        "legacy behavior changed",
        "product requirement not met",
        "ux spec not followed",
        "ui behavior does not match ux spec",
    ),

    LABEL_LOGIC: make_keywords(
        "bug", "logic", "logic bug", "logic error",
        "semantic", "semantic defect", "semantic bug",
        "fault", "defect", "error", "exception",
        "crash", "crashes", "hang", "timeout",
        "null pointer", "nullpointer", "npe",
        "null", "none", "nil",
        "off by one", "off-by-one",
        "edge case", "edge cases",
        "invariant", "invariants",
        "race condition", "data race", "concurrency",
        "deadlock", "livelock", "starvation",
        "overflow", "underflow", "wraparound",
        "division by zero",
        "index out of bounds", "out of range",
        "wrong condition", "incorrect condition",
        "wrong branch", "unreachable branch",
        "unreachable code", "dead code",
        "never executed",
        "incorrect algorithm", "wrong algorithm",
        "wrong result", "incorrect result",
        "violates invariant", "invariant not preserved",
        "breaks precondition", "breaks postcondition",
        "wrong variable", "wrong field updated",
        "uses wrong parameter",
        "wrong return value",
        "incorrect default value",
        "incorrect comparison",
        "floating point precision", "rounding error",
        "time calculation bug", "date handling bug",
        "edge case not handled",
        "null not handled",
        "empty list not handled",
        "exception not handled",
        "exception swallowed",
        "incorrect error handling",
        "incorrect fallback behavior",
        "silent failure", "silent bug",
        "data corruption", "state corruption",
    ),

    LABEL_BUILD_CI: make_keywords(
        "build", "builds", "broken build",
        "fails to build", "build failure",
        "ci", "ci pipeline", "pipeline",
        "job failed", "failing job", "failing pipeline",
        "non reproducible", "non-reproducible", "not reproducible",
        "works locally but fails in ci",
        "flaky test", "flaky tests",
        "nondeterministic", "non deterministic",
        "dependency", "dependencies", "dependency drift",
        "missing dependency", "version conflict",
        "platform difference", "os specific",
        "windows build failure",
        "linux build failure",
        "macos build failure",
        "toolchain mismatch", "compiler version mismatch",
        "linker error", "linking error",
        "missing header file", "missing include file",
        "environment", "env mismatch", "environment variable not set",
        "ci environment misconfigured",
        "ci configuration issue",
        "docker image does not build",
        "docker build failure",
        "container build failure",
        "test suite times out in ci",
        "timeout in ci job",
        "required status checks failing",
        "required check red",
        "merge blocked by failing checks",
        "unmergeable due to ci",
        "cache corruption in ci", "cache mismatch in pipeline",
        "disk space exhausted in ci",
    ),

    LABEL_STYLE: make_keywords(
        "style", "code style", "styling",
        "convention", "conventions",
        "format", "formatting", "formatter",
        "lint", "linter", "linting",
        "pylint", "eslint", "flake8", "checkstyle",
        "prettier", "gofmt", "clangformat", "black", "isort",
        "naming", "rename", "naming convention",
        "camelcase", "snake_case", "pascalcase",
        "indentation", "indent", "whitespace",
        "line length", "max line length",
        "spacing", "imports order", "import order",
        "readability", "readable",
        "docstring", "docstrings", "doc style",
        "header comment",
        "console.log", "debug log", "debug logging",
        "logging level misuse",
        "commented out code", "commented code",
        "nitpick", "nit", "cosmetic change",
        "minor style nit",
        "inconsistent indentation",
        "trailing whitespace",
        "missing newline at end of file",
        "unused import", "unused variable",
        "dead assignment",
        "inconsistent code style",
        "not following project style guide",
    ),

    LABEL_TESTING: make_keywords(
        "test", "tests", "testing", "test suite",
        "unit test", "unit tests",
        "integration test", "integration tests",
        "e2e", "end to end test",
        "regression test", "regression tests",
        "smoke test", "sanity test",
        "fixture", "fixtures",
        "mock", "mocks", "mocking",
        "stub", "stubs", "fake", "fakes",
        "assert", "asserts", "assertion",
        "expect", "expects", "matcher", "matchers",
        "coverage", "test coverage",
        "coverage threshold",
        "coverage drop", "coverage decreased",
        "missing tests", "no tests",
        "no unit tests added",
        "missing unit tests",
        "missing integration tests",
        "missing regression tests",
        "weak tests", "weak test coverage",
        "insufficient test coverage",
        "low test coverage",
        "tests incorrect", "incorrect tests",
        "wrong assertion", "assertion is wrong",
        "tests do not match spec",
        "tests not updated",
        "no negative test cases",
        "no error path tests",
        "happy path only",
        "flaky test", "flaky tests",
        "unstable tests",
        "test suite failing",
        "test failure in ci",
        "please add more test cases",
        "need regression tests for the bug",
        "code path is not covered by any test",
    ),

    LABEL_DESIGN: make_keywords(
        "architecture", "architectural", "architected",
        "design", "design smell", "poor design",
        "design misfit", "design flaw",
        "layer", "layers", "layering",
        "violates layering", "layering violation",
        "service boundary", "boundary context",
        "domain model", "domain layer",
        "abstraction", "bad abstraction", "leaky abstraction",
        "encapsulation", "coupling", "tight coupling",
        "high coupling", "low cohesion",
        "separation of concerns", "mixes concerns",
        "god object", "god class",
        "overly complex method", "method too complex",
        "class too big",
        "dependency cycle", "circular dependency",
        "module cycle", "package cycle",
        "wrong ownership of data", "wrong ownership of logic",
        "belongs in a different class",
        "belongs in the service layer",
        "business logic in controller",
        "business logic in view",
        "persistence logic in controller",
        "violates clean architecture",
        "not following hexagonal architecture",
        "not following ddd principles",
        "performance budget", "performance regression",
        "hot path", "hotpath", "performance hotspot",
        "scalability concern", "not extensible",
        "maintainability issue", "hard to maintain",
        "api surface too large",
        "leaking internal details",
    ),

    LABEL_PROCESS: make_keywords(
        "process", "processes",
        "policy", "policies",
        "governance", "governance gate",
        "workflow", "workflows",
        "standard", "standards",
        "procedure", "procedures",
        "guideline", "guidelines",
        "cla", "license agreement",
        "dco", "sign-off", "sign off",
        "license", "licenses", "licensing",
        "compliance", "noncompliant",
        "required reviewers", "required reviewer",
        "code owners", "code owner",
        "approval", "approvals", "approver",
        "required approvals", "mandatory review",
        "changelog", "change log",
        "release notes",
        "coverage threshold policy",
        "status check", "status checks",
        "branch protection", "protected branch",
        "unmergeable", "cannot be merged", "merge blocked",
        "target branch not allowed",
        "direct push forbidden",
        "pull request template", "template not filled", "missing template",
        "checklist", "checklist not completed",
        "ticket id missing", "missing issue link",
        "security review missing",
        "policy violation",
        "does not follow our contribution process",
        "process step skipped",
        "governance check failed",
        "unmergeable due to policy",
    ),

    LABEL_TOOLING: make_keywords(
        "tool", "tools", "tooling",
        "cli", "command", "commands",
        "script", "scripts", "automation", "automated job",
        "generator", "code generator", "scaffolding",
        "wrapper script",
        "linter command", "test runner",
        "release job", "ci job",
        "runner configuration", "runner config",
        "plugin", "plugins", "extension", "extensions",
        "pre commit", "pre-commit", "precommit",
        "hook", "hooks",
        "cache", "cache issue", "stale cache",
        "docker script", "docker build script",
        "deployment script",
        "wrong flag", "wrong option", "wrong argument",
        "command line option",
        "tool misuse", "misuse of tool",
        "tooling issue", "tooling error",
        "automation script error",
        "script failure", "script error",
        "release script failed",
        "build script failed",
        "generator not run", "generator was not run correctly",
        "database migration missing",
        "schema migration missing",
        "tool configuration wrong", "misconfigured tool",
        "misconfigured linter", "misconfigured ci job",
        "local tool version mismatch",
        "missing plugin for tool",
        "missing tool dependency",
        "pre-commit hooks were not run on this change",
    ),

    LABEL_OTHER: make_keywords(
        "looks good to me", "lgtm",
        "thanks", "thank you",
        "nice refactor", "nice change", "good job",
        "general question", "not a rejection",
        "small nit", "nit but not blocking",
        "just a comment", "just a thought",
    ),
}


def rule_based_label(text: str) -> Optional[str]:
    """
    Try to infer the label using simple keyword matching based on the
    taxonomy definitions and their explanations.
    Returns a label or None if there is no strong signal.
    """
    t = text.casefold()  # more robust than lower()
    words = t.split()
    scores = {}

    for label, kws in KEYWORDS.items():
        if not kws:
            continue
        count = sum(1 for kw in kws if kw in t)
        if count > 0:
            scores[label] = count

    if not scores:
        return None

    best_label = max(scores, key=scores.get)
    max_hits = scores[best_label]

    # at least 2 hits → strong signal
    if max_hits >= 2:
        return best_label

    # short text with a single strong keyword
    if max_hits == 1 and len(words) <= 8:
        return best_label

    return None


# -------------------------------------------------
# 1) Small hand-written seed examples
# -------------------------------------------------
SEED_LABEL_TEXTS = {
    LABEL_SPEC_INTENT: [
        "this change does not match the requirement",
        "wrong behavior compared to the spec",
        "misinterprets the issue description",
        "does not implement the requested feature correctly",
        "behavior is different from what the ticket describes",
        "implements the wrong requirement entirely",
        "the API contract says null must be allowed",
        "this does not cover all cases from the user story",
    ],
    LABEL_LOGIC: [
        "null pointer exception risk",
        "off by one error in the loop",
        "logic bug when flag is false",
        "wrong condition in the if statement",
        "breaks edge cases when list is empty",
        "incorrect algorithm for this data structure",
        "this branch will never be executed",
        "updating the wrong variable here",
        "this will overflow for large n",
    ],
    LABEL_BUILD_CI: [
        "fails to build on the ci server",
        "ci pipeline fails on linux environment",
        "non reproducible build between local and ci",
        "missing dependency in the build configuration",
        "works locally but fails in ci",
        "docker image does not build successfully",
        "test suite times out only in ci environment",
        "flaky test in the CI job",
    ],
    LABEL_STYLE: [
        "coding style problem according to our guide",
        "formatting issue: please run the formatter",
        "naming convention is not followed for this class",
        "please run the linter before pushing",
        "does not follow our style guide for imports",
        "indentation and spacing are inconsistent",
        "line length exceeds the maximum allowed",
        "this pattern is not idiomatic for this codebase",
    ],
    LABEL_TESTING: [
        "missing tests for this new behavior",
        "weak test coverage around edge cases",
        "no unit tests were added for this change",
        "tests are incorrect and assert the wrong behavior",
        "please add more test cases for negative scenarios",
        "this code path is not covered by any test",
        "need regression tests for the bug you are fixing",
        "coverage threshold is not met for this change",
    ],
    LABEL_DESIGN: [
        "breaks our architecture by crossing service boundaries",
        "not aligned with the design of this module",
        "violates layering between api and data access",
        "poor design: mixes concerns from different layers",
        "bad abstraction, this logic belongs in a different class",
        "introduces tight coupling between unrelated modules",
        "creates a circular dependency in the package structure",
        "this adds an O(n^2) algorithm in a hot path",
    ],
    LABEL_PROCESS: [
        "does not follow our contribution process",
        "missing cla signature for this author",
        "policy violation: target branch is not allowed",
        "needs approval from maintainer according to policy",
        "required labels are missing from this pull request",
        "coverage threshold not met according to policy",
        "you must update the changelog before we can merge",
    ],
    LABEL_TOOLING: [
        "misuse of tool: please use our wrapper script",
        "failed linter command due to wrong arguments",
        "automation script error when running the release job",
        "wrong build flag passed to the compiler",
        "generator was not run correctly for the new schema",
        "using the wrong test runner command for this project",
        "please rerun the code generator before committing",
        "pre-commit hooks were not run on this change",
    ],
    LABEL_OTHER: [
        # it's fine if this stays empty; OTHER is mainly fallback
    ],
}


def seed_examples_from_dict() -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for label, examples in SEED_LABEL_TEXTS.items():
        for txt in examples:
            texts.append(txt)
            labels.append(label)
    return texts, labels


# -------------------------------------------------
# 2) Load labeled comments from CSV
# -------------------------------------------------
DATASET_FILES = [
    "pr_reasons_labeled.csv",
    "pr_comments_dataset_publish.csv",
]


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def normalize_category(raw: str) -> Optional[str]:
    if not raw:
        return None

    c = raw.strip().strip('"').strip("'")
    if ")" in c:
        _, after = c.split(")", 1)
        c_core = after.strip()
    else:
        c_core = c

    core_lower = c_core.lower()

    if core_lower.startswith("specification") or "intent mismatch" in core_lower:
        return LABEL_SPEC_INTENT

    if core_lower.startswith("logic") or "semantic" in core_lower:
        return LABEL_LOGIC

    if core_lower.startswith("build") or "ci/environment" in core_lower or "environment failure" in core_lower:
        return LABEL_BUILD_CI

    if core_lower.startswith("style") or "convention violation" in core_lower:
        return LABEL_STYLE

    if (
        core_lower.startswith("testing")
        or "testing inadequacy" in core_lower
        or "missing, weak, or incorrect tests" in core_lower
    ):
        return LABEL_TESTING

    if core_lower.startswith("architectural") or "design misfit" in core_lower:
        return LABEL_DESIGN

    if core_lower.startswith("process") or core_lower.startswith("policy"):
        return LABEL_PROCESS

    if core_lower.startswith("tool-use") or core_lower.startswith("tool use") or "automation error" in core_lower:
        return LABEL_TOOLING

    if core_lower.startswith("other"):
        return LABEL_OTHER

    return LABEL_OTHER


def load_csv_examples() -> Tuple[List[str], List[str]]:
    base_dir = Path(__file__).resolve().parent
    texts: List[str] = []
    labels: List[str] = []

    encodings_to_try = ["utf-8-sig", "utf-8", "cp1256", "latin-1"]

    for fname in DATASET_FILES:
        path = base_dir / fname
        if not path.exists():
            continue

        loaded_this_file = 0

        for enc in encodings_to_try:
            try:
                with path.open("r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        raw_text = (
                            row.get("body_comment")
                            or row.get("comment_body")
                            or row.get("body")
                            or row.get("text")
                        )
                        raw_cat = row.get("Category") or row.get("category") or row.get("label")

                        text = clean_text(raw_text or "")
                        if not text:
                            continue

                        label = normalize_category(raw_cat or "")
                        if label is None:
                            continue

                        texts.append(text)
                        labels.append(label)
                        loaded_this_file += 1

                print(f"[INFO] Loaded {loaded_this_file} rows from {path.name} using encoding '{enc}'.")
                break

            except UnicodeDecodeError:
                # try next encoding
                continue
            except Exception as e:
                print(f"[WARN] Failed to load {path.name} with encoding {enc}: {e}")
                break

        if loaded_this_file == 0:
            print(f"[WARN] Could not read any rows from {path.name} with encodings {encodings_to_try}.")

    print(f"[INFO] Loaded {len(texts)} labeled rows from CSV files in total.")
    return texts, labels


# -------------------------------------------------
# 3) Train model (seed + CSV)
# -------------------------------------------------
seed_texts, seed_labels = seed_examples_from_dict()
csv_texts, csv_labels = load_csv_examples()

train_texts: List[str] = seed_texts + csv_texts
train_labels: List[str] = seed_labels + csv_labels

if not train_texts:
    raise RuntimeError("No training data found. Please add at least some seed examples or CSV rows.")

print(f"[INFO] Total training examples: {len(train_texts)}")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train_texts)

clf = LinearSVC()
clf.fit(X_train, train_labels)

# if SVM is too unsure, fallback to OTHER
CONFIDENCE_MARGIN = 0.25


# -------------------------------------------------
# Schemas
# -------------------------------------------------
class TextInput(BaseModel):
    text: str


class ClassificationResult(BaseModel):
    predicted_label: Optional[str]
    margin: float
    best_score: float


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    """
    Simple health-check + metadata.
    Now also returns label explanations so the frontend
    can show tooltips or documentation for each category.
    """
    return {
        "message": "PR Review Reason Classifier is running 🚀",
        "labels": ALL_LABELS,
        "label_explanations": LABEL_EXPLANATIONS,
    }


@app.post("/classify", response_model=ClassificationResult)
def classify(input: TextInput):
    """
    1) Rule-based using taxonomy keywords and explanations.
    2) If no strong rule hit → SVM.
    3) If SVM margin small → 'Other'.
    """
    text = clean_text(input.text or "")
    if not text:
        return ClassificationResult(predicted_label=None, margin=0.0, best_score=0.0)

    # Step 1: rule-based
    rb_label = rule_based_label(text)
    if rb_label is not None:
        # We use (margin=1, best_score=1) as a clear signal in the UI
        # that this came from the taxonomy rules, not SVM.
        return ClassificationResult(predicted_label=rb_label, margin=1.0, best_score=1.0)

    # Step 2: SVM
    X = vectorizer.transform([text])
    scores = clf.decision_function(X)[0]
    classes = clf.classes_

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    sorted_scores = np.sort(scores)
    second_best = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_best

    if margin < CONFIDENCE_MARGIN:
        predicted = LABEL_OTHER
    else:
        predicted = str(classes[best_idx])

    return ClassificationResult(predicted_label=predicted, margin=float(margin), best_score=best_score)
