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
# (summarized from the taxonomy paragraphs)
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
# Rule-based keywords per label (from taxonomy definitions)
# Extended using phrases from the written explanations.
# -------------------------------------------------
KEYWORDS = {
    LABEL_SPEC_INTENT: [
        "spec", "specification", "intent", "requirement", "requirements",
        "ticket", "story", "user story", "issue description", "api contract",
        "contract says", "does not match", "wrong behavior", "expected behavior",
        "expected result", "misinterpret", "misinterprets", "scope",
        "partially addresses", "outdated requirement", "wrong thing",
        "required was to", "doesn't cover case", "does not cover case",        "spec","specs","specification","specifications","requirement","requirements","req",
        "story","stories","ticket","tickets","issue","issues","jira","backlog","scope",
        "scoping","mismatch","misaligned","misalign","incorrect","wrong","unexpected",
        "behavior","behaviour","expected","actual","contract","apicontract","interface",
        "protocol","semantic","semantics","acceptance","criteria","ux","designspec",
        "product","business","rule","rules","incomplete","partial","partially","missing",
        "omitted","unimplemented","unsupported","misread","misunderstood","misinterpret",
        "misinterpreted","ambiguity","ambiguous","inconsistent","consistency","compatible",
        "compatibility","breaking","regression","legacy","deprecated","docs","documentation",
        "docstring","comment","comments","description","descriptions","scenario","scenarios",
        "usecase","usecases","workflow","workflows","flag","flags","configuration","config",
        "default","defaults","option","options","parameter","parameters","input","inputs",
        "output","outputs","payload","response","status","statuses","http","endpoint",
        "endpoints","contractual","specbug","feature","features", "spec mismatch",
        "specification mismatch",
        "requirements mismatch",
        "requirement mismatch",
        "does not match requirement",
        "does not match the spec",
        "wrong requirement implemented",
        "implements wrong requirement",
        "implements wrong feature",
        "implements wrong behavior",
        "wrong behavior vs spec",
        "behavior different from spec",
        "inconsistent with ticket",
        "ticket not fully addressed",
        "issue not fully addressed",
        "partially addresses the issue",
        "partial fix for the issue",
        "partial implementation of story",
        "misinterpreted requirement",
        "misinterpreted the ticket",
        "misinterpreted the story",
        "misinterpreted spec",
        "misunderstood requirement",
        "misunderstood the spec",
        "incorrect interpretation of spec",
        "incorrect interpretation of ticket",
        "does not follow acceptance criteria",
        "acceptance criteria not met",
        "acceptance criteria missing",
        "user story not satisfied",
        "user story only partially satisfied",
        "business rule violated",
        "business rules not followed",
        "functional mismatch",
        "functional requirements not met",
        "feature does not match description",
        "feature not implemented as described",
        "unexpected behavior for user",
        "expected behavior is different",
        "expected behavior not implemented",
        "expected result not achieved",
        "missing part of requirement",
        "missing required behavior",
        "missing requirement",
        "out of scope change",
        "out of scope for this ticket",
        "api contract mismatch",
        "api does not match docs",
        "documentation does not match behavior",
        "protocol mismatch",
        "wrong http status code",
        "breaking public api contract",
        "backward compatibility issue",
        "regression vs previous behavior",
        "does not respect feature flag semantics",
        "product requirement not met",
        "stakeholder expectation not met",
        "ux spec not followed",
        "ui behavior does not match ux spec",  "wrongfeature",
        "wrongflow",
        "wrongscenario",
        "wrongusecase",
        "wrongticketid",
        "wrongstoryid",
        "wrongendpointbehavior",
        "wronguserflow",
        "wrongbusinesslogic",
        "wrongacceptance",
        "wrongcontract",
        "wrongprotocol",
        "wrongapibehavior",
        "wrongrequestshape",
        "wrongresponseshape",
        "wrongfieldmapping",
        "wrongfieldname",
        "wrongenumvalue",
        "wrongstatuscode",
        "wrongstatebehavior",
        "requirementsgap",
        "requirementsbreach",
        "requirementsdrift",
        "requirementsmismatch",
        "requirementsviolation",
        "requirementgap",
        "requirementbreach",
        "requirementdrift",
        "requirementconflict",
        "requirementdeviation",
        "specgap",
        "specbreach",
        "specdrift",
        "specconflict",
        "specdeviation",
        "specerror",
        "specmismatchbug",
        "specregression",
        "specoutofsync",
        "specoutdated",
        "specincorrect",
        "specincomplete",
        "specuncovered",
        "specunimplemented",
        "specnotfollowed",
        "specnotapplied",
        "specnotrespected",
        "misalignedbehavior",
        "misalignedfeature",
        "misalignedflow",
        "misalignedcontract",
        "misalignedinterface",
        "misalignedrequirements",
        "misalignedspec",
        "misalignedticket",
        "misalignedstory",
        "misalignedux",
        "misinterpretedflow",
        "misinterpretedticket",
        "misinterpretedstory",
        "misinterpretedacceptance",
        "misinterpretedrequirement",
        "misinterpretedusecase",
        "misinterpretedapi",
        "misinterpretedstatus",
        "misinterpretedcontract",
        "misinterpretedprotocol",
        "misinterpretedfield",
        "misreadticket",
        "misreadstory",
        "misreadrequirement",
        "misreadspecref",
        "misreadspecline",
        "misreadspeccase",
        "outofscoperequirement",
        "outofscopechange",
        "outofscopebehavior",
        "outofscopefeature",
        "outofscopeendpoint",
        "outofscopeparameter",
        "outofscopeflag",
        "outofscopevalidation",
        "partiallyimplemented",
        "partialrequirement",
        "partialbehavior",
        "partialcoverage",
        "partialticketfix",
        "partialstoryfix",
        "partialuxcompatibility",
        "partialbusinessrule",
        "partialacceptance",
        "featuregap",
        "featuremissingpiece",
        "featuremisbehavior",
        "featureincomplete",
        "featureunderimplemented",
        "featuremisaligned",
        "featurewrongmapping",
        "featurewrongstate",
        "featurewrongtransition",
        "uxbehavior mismatch",
        "productbehavior mismatch",
        "businessbehavior mismatch",
        "ticketbehavior mismatch",
        "storybehavior mismatch",
        "interface mismatch",
        "contract mismatch bug",
        "apicontract drift",
        "apicontract violation",
        "breakingapiintent",
        "breakinguxintent",
        "breakingproductintent",
        "inconsistentwithspec",
        "inconsistentwithticket",
        "inconsistentwithstory",
        "inconsistentwithdocs",
        "inconsistentwithux",
   
    ],
    LABEL_LOGIC: [
        "bug", "logic", "semantic", "null pointer", "nullpointer",
        "off by one", "off-by-one", "edge case", "edge cases", "invariant",
        "race condition", "data race", "overflow", "underflow",
        "wrong condition", "incorrect condition", "never executed",
        "dead code", "wrong branch", "wrong variable",
        "incorrect algorithm", "incorrect result", "wrong result",
        "fails for large", "fails when list is empty",
        "what happens when", "counter example", "counterexample","bug","bugs","logic","semantic","semantics","fault","faults","defect","defects",
        "error","errors","exception","exceptions","null","none","nil","pointer","npe",
        "crash","crashes","overflow","underflow","wraparound","truncate","truncation",
        "rounding","precision","race","races","concurrency","thread","threads","locking",
        "lock","mutex","deadlock","livelock","starvation","hang","timeout","timeouts",
        "loop","loops","branch","branches","condition","conditions","predicate","predicates",
        "boolean","bool","flag","flags","state","states","invariant","invariants",
        "consistency","overflowing","index","indices","bounds","boundary","boundaries",
        "offbyone","leak","leaks","leakage","corrupt","corruption","stale","dangling",
        "aliasing","recursion","recursive","stack","heap","overflowed","rollback",
        "rollbacking","rollbacked","rollbacks","rollover","wraps","wrap","divide","division",
        "zero","nan","infinity","inf","NaN","cast","casting","coerce","coercion",
        "serialization","deserialization","encoding","decoding","logic bug",
        "logic error",
        "semantic defect",
        "semantic bug",
        "null pointer",
        "null pointer exception",
        "npe risk",
        "off by one",
        "off-by-one error",
        "index out of bounds",
        "array index out of bounds",
        "out of range access",
        "wrong condition",
        "incorrect condition",
        "broken condition check",
        "wrong boolean logic",
        "incorrect boolean logic",
        "infinite loop",
        "possible infinite loop",
        "race condition",
        "data race",
        "concurrency bug",
        "thread safety issue",
        "deadlock risk",
        "overflow bug",
        "integer overflow",
        "integer underflow",
        "division by zero",
        "wrong branch taken",
        "unreachable branch",
        "unreachable code",
        "dead code",
        "never executed branch",
        "incorrect algorithm",
        "wrong algorithm complexity",
        "violates invariant",
        "invariant not preserved",
        "breaks precondition",
        "breaks postcondition",
        "wrong variable updated",
        "updates wrong field",
        "uses wrong parameter",
        "wrong return value",
        "incorrect default value",
        "incorrect comparison",
        "floating point precision issue",
        "rounding error",
        "time calculation bug",
        "date handling bug",
        "edge case not handled",
        "edge cases broken",
        "null not handled",
        "empty list not handled",
        "error path not handled",
        "exception not handled",
        "exception swallowed",
        "incorrect error handling",
        "incorrect fallback behavior","logicregression",
        "logicfault",
        "logicglitch",
        "logicbreak",
        "logicfailure",
        "logicviolation",
        "logiccornercase",
        "logicboundarybug",
        "logicedgefailure",
        "logicedgecasebug",
        "semanticbug",
        "semanticfault",
        "semanticregression",
        "semanticviolation",
        "semanticbreak",
        "semanticmismatch",
        "semanticedgecase",
        "semanticbordercase",
        "wrongbranchlogic",
        "wrongguardcheck",
        "wrongguardcondition",
        "wrongpredicate",
        "wrongboolean",
        "wrongflagvalue",
        "wrongflagbranch",
        "wrongstatebranch",
        "wrongfallbackbranch",
        "wrongerrorbranch",
        "invalidstate",
        "illegalstate",
        "invalidtransition",
        "invalidflow",
        "invalidbranchpath",
        "invalidguard",
        "invalidpredicate",
        "invalidassumption",
        "invalidprecondition",
        "invalidpostcondition",
        "preconditionviolation",
        "postconditionviolation",
        "invariantbreak",
        "invariantfailure",
        "invariantbreach",
        "invariantnotheld",
        "invariantnotpreserved",
        "overflowpath",
        "underflowpath",
        "overflowbug",
        "underflowbug",
        "wraparoundbug",
        "roundingbug",
        "precisionloss",
        "precisionbug",
        "floatinstability",
        "floatcomparison",
        "floatmismatch",
        "nanbug",
        "infbug",
        "naninstate",
        "naninresult",
        "invalidcast",
        "invalidcoercion",
        "invalidconversion",
        "dangerousconversion",
        "dangerouscast",
        "overlynestedlogic",
        "deepnesting",
        "nestedbranching",
        "duplicatedbranch",
        "duplicatedcondition",
        "contradictorycondition",
        "contradictorybranch",
        "unreachablepath",
        "deadlogic",
        "deadbranch",
        "unusedbranch",
        "unusederrorpath",
        "unhandlederrorpath",
        "unhandledexceptioncase",
        "unhandlededgecase",
        "unhandlednullcase",
        "unhandledemptycase",
        "unhandledoverflow",
        "unhandledtimeout",
        "exceptionpathmissing",
        "exceptionnotreported",
        "exceptionswallowed",
        "silentfailure",
        "silentbug",
        "silentcorruption",
        "statecorruption",
        "datacorruptionlogic",
        "timerlogicbug",
        "dateboundarybug",
        "calendarbug",
        "timezonebug",
        "localeedgecase",
        "stringboundarybug",
        "indexboundarybug",
    ],
    LABEL_BUILD_CI: [
        "build", "builds", "does not build", "fails to build",
        "ci", "ci pipeline", "pipeline", "job failed",
        "non reproducible", "non-reproducible", "not reproducible",
        "flaky", "flaky test", "nondeterministic", "non deterministic",
        "test matrix", "matrix job", "dependency", "dependencies",
        "build failure", "linker error", "cannot run tests",
        "platform difference", "environment", "env mismatch",
        "status checks", "required status checks", "required check",
        "lockfile", "lock file", "version drift", "upstream change",
        "docker image does not build", "cannot reproduce in ci","build","builds","builder","compilation","compile","compiler","link","linker",
        "linking","artifact","artifacts","binary","binaries","package","packages",
        "dependency","dependencies","deps","version","versions","versioning","upgrade",
        "downgrade","migrate","migration","migrations","ci","pipeline","pipelines",
        "workflow","workflows","job","jobs","step","steps","matrix","runner","runners",
        "agent","agents","executor","executors","environment","environments","env",
        "envvars","variable","variables","config","configuration","configurations","yml",
        "yaml","docker","container","containers","image","images","kubernetes","cluster",
        "node","nodes","cache","caches","caching","artifactcache","lockfile","lockfiles",
        "timeout","timeouts","flaky","flake","nondeterministic","nonreproducible",
        "platform","platforms","os","windows","linux","macos","unix","darwin","status",
        "check","checks","gate","gates","green","red","queue","queued","retries","retry",
        "rerun","rebuild","retrigger","rerunnable","fail","failure","failed", "build fails",
        "build failure",
        "broken build",
        "cannot build the project",
        "does not build on ci",
        "ci fails",
        "ci pipeline fails",
        "ci job failed",
        "failing ci job",
        "failing pipeline",
        "non reproducible build",
        "non-reproducible build",
        "not reproducible locally",
        "works locally but fails in ci",
        "dependency drift",
        "dependency mismatch",
        "missing dependency",
        "wrong dependency version",
        "version conflict in dependencies",
        "platform difference",
        "os specific failure",
        "windows-only build failure",
        "linux-only build failure",
        "macos build failure",
        "toolchain mismatch",
        "compiler version mismatch",
        "linker error",
        "linking error",
        "missing header file",
        "missing include file",
        "missing binary artifact",
        "environment variable not set",
        "ci environment misconfigured",
        "ci configuration issue",
        "pipeline configuration issue",
        "docker image does not build",
        "docker build failure",
        "container build failure",
        "flaky test in ci",
        "flaky integration test",
        "non deterministic test",
        "nondeterministic test outcome",
        "timeout in ci job",
        "test suite timeout",
        "insufficient ci resources",
        "disk space exhausted in ci",
        "cache corruption in ci",
        "cache mismatch in pipeline",
        "required status checks failing",
        "required checks not passing",
        "status check is red",
        "merge blocked by failing checks",
        "unmergeable due to ci", "buildbreak",
        "buildred",
        "buildunstable",
        "buildpipeline",
        "buildmatrix",
        "buildstepfailure",
        "buildconfigbreak",
        "buildconfigerror",
        "buildenvmismatch",
        "buildenvmissing",
        "buildtoolmismatch",
        "buildtoolversion",
        "buildtoolfailure",
        "buildcacheissue",
        "buildartifactmissing",
        "buildartifactmismatch",
        "buildartifactcorrupt",
        "buildlogerror",
        "buildsystemerror",
        "pipelinestepbreak",
        "pipelinegatefailure",
        "pipelineunstable",
        "pipelinesporadic",
        "cifailingjob",
        "cifailurepattern",
        "ciflakyjob",
        "ciflakystep",
        "ciinfrastructure",
        "ciinfraissue",
        "ciinfrabug",
        "ciworkerfailure",
        "ciworkeroffline",
        "ciagentoffline",
        "ciagentfailure",
        "cirunnermismatch",
        "ciimageerror",
        "ciimagepullerror",
        "ciimagecache",
        "cipermissiondenied",
        "ciscriptpermission",
        "ciresourceexhausted",
        "cidiskfull",
        "cimemoryexhausted",
        "ciquotaexceeded",
        "ciratelimit",
        "nondeterministicbuild",
        "nondeterministictest",
        "nondeterministicpipeline",
        "nonreproducibleci",
        "nonreproducibletest",
        "platformdrift",
        "platformincompatibility",
        "compilerversiondrift",
        "toolchainversiondrift",
        "linkerplatformbug",
        "systemlibrarymissing",
        "systemdependency",
        "osdependency",
        "kernelversionissue",
        "containerdrift",
        "dockerdrift",
        "dockercacheissue",
        "dockerlayercache",
        "dockernetworkissue",
        "dockervolumemount",
        "kubernetesclusterissue",
        "kubernetesnodeissue",
        "kubernetespodcrash",
        "artifactpublishfailure",
        "artifactdownloaderror",
        "artifactstoreerror",
        "artifactregistryissue",
        "statuscheckblocking",
        "statuscheckrequired",
        "statuschecktimeout",
        "requiredcheckred",
        "requiredcheckstuck",
        "queuedbuildforever",
        "queuedpipelineforever",
        "retryingbuild",
        "rerunningjob",
        "rerunningpipeline",
        "manualrerunneeded",
        "branchciincomplete",
        "branchnotcoveredci",
        "missingcijob",
        "missingplatformjob",
        "missingmatrixentry",
        "inconsistentmatrix",
        "inconsistentstatus",
        "inconsistentciresult",
        "inconsistentbuild",
        "ephemeralflakiness",
    ],

    LABEL_STYLE: [
        "style", "code style", "convention", "conventions",
        "format", "formatting", "formatter", "lint", "linter",
        "naming", "name", "rename", "camelcase", "snake_case",
        "indentation", "whitespace", "line length", "max line length",
        "spacing", "imports order", "import order",
        "idiomatic", "non idiomatic", "non-idiomatic",
        "readability", "readable",
        # logging / nitpick cases
        "console.log", "debug log", "debug", "logging",
        "log statement", "commented out code", "commented code",
        "nitpick", "nit", "cosmetic change","style","styling","format","formatted","formatting","formatter","lint","lints",
        "linting","linter","linters","pylint","eslint","flake8","checkstyle","detekt",
        "ktlint","prettier","clangformat","gofmt","black","isort","whitespace","spaces",
        "tabs","indent","indents","indentation","align","alignment","column","columns",
        "margin","margins","wrap","wrapping","linewidth","linebreak","newline","newlines",
        "braces","brace","brackets","semicolon","semicolons","quote","quotes","quoting",
        "singlequote","doublequote","camelcase","snakecase","pascalcase","kebabcase",
        "naming","name","names","rename","renaming","unused","useless","deadcode",
        "cleanup","cleanups","refactor","refactoring","tidy","tidying","ordering","order",
        "sorted","sorting","imports","importorder","comment","comments","docstyle",
        "docblock","docblocks","docstring","docstrings","header","headers","footer",
        "footers","todo","todos","fixme","nit","nitpick","cosmetic","aesthetic","readability",
        "readable","consistent","inconsistent","convention","conventions","guideline",
        "guidelines", "coding style issue",
        "style violation",
        "style problem",
        "style guidelines not followed",
        "convention violation",
        "naming convention not followed",
        "naming issue",
        "bad variable name",
        "bad method name",
        "rename this variable",
        "rename this method",
        "formatting issue",
        "code not formatted",
        "please run the formatter",
        "please run prettier",
        "please run black",
        "please run gofmt",
        "please run clang format",
        "lint error",
        "linter error",
        "lint warnings",
        "eslint error",
        "flake8 error",
        "pylint warning",
        "checkstyle violation",
        "imports not ordered",
        "wrong import order",
        "unused import",
        "unused variable",
        "dead assignment",
        "trailing whitespace",
        "extra whitespace",
        "inconsistent indentation",
        "tabs instead of spaces",
        "spaces instead of tabs",
        "line length too long",
        "line exceeds maximum length",
        "brace style issue",
        "braces on same line",
        "braces on next line",
        "missing newline at end of file",
        "missing end of file newline",
        "commented out code",
        "debug log left in",
        "console.log left in",
        "leftover debug logging",
        "temporary logging statement",
        "nitpick comment",
        "small nit",
        "minor style nit",
        "logging level misuse",
        "inconsistent code style",
        "not following project style guide","stylenoncompliant",
        "styleinconsistency",
        "styledeviation",
        "stylenotfollowed",
        "stylewarning",
        "styleerror",
        "codeformatissue",
        "unformattedblock",
        "unformattedfile",
        "unformattedchunk",
        "whitespacenoise",
        "indentnoise",
        "indentmixed",
        "indenttabmixed",
        "indentspacemixed",
        "misalignedblock",
        "misalignedcase",
        "misalignedparam",
        "misalignedarg",
        "misalignedcomment",
        "braceposition",
        "braceplacement",
        "bracemisaligned",
        "bracketplacement",
        "parensspacing",
        "operatorspacing",
        "commaspacing",
        "colons spacing",
        "semicolonspacing",
        "longchainedcall",
        "onelettername",
        "nonmeaningfulname",
        "crypticidentifier",
        "badclassname",
        "badinterfacename",
        "badpackagename",
        "badmodulename",
        "noisycomment",
        "obviouscomment",
        "outdatedcomment",
        "commentdrift",
        "commentnotsynced",
        "commentedblock",
        "commenteddebug",
        "commentedlegacy",
        "logleftover",
        "printleftover",
        "printfdebug",
        "debugprint",
        "debugstatement",
        "debuggingartifact",
        "nitstyle",
        "nitlevelissue",
        "cosmeticonly",
        "aestheticonly",
        "layoutonly",
        "reflowneeded",
        "reformatneeded",
        "reindentneeded",
        "renamenecessary",
        "renameidentifier",
        "cleanuplocal",
        "cleanupimports",
        "cleanupstyle",
        "styleruleviolation",
        "stylerulenotapplied",
        "stylerulenotrespected",
        "styleconfigmissing",
        "styleconfigdrift",
        "styletoolmismatch",
        "lintercomplaint",
        "linterwarning",
        "lintersuggestion",
        "linterautofix",
        "lintautofix",
        "lintnoise",
        "lintonlychange",
        "formatonlypatch",
        "formatnoise",
        "diffnoisy",
        "diffnoisypadding",
        "blanklinesnoise",
        "blanklinespamming",
        "headerstyleissue",
        "filenamestyle",
        "foldernamestyle",
        "constantnamestyle",
        "enumstyle",
        "macrostyle",
        "annotationstyle",
        "decoratorstyle",
        "docblockstyle",
        "apidocstyle",
        "markdownstyle",
        "markdownformat",
        "spurioussemicolon",
        "spuriouscomma",
        "redundantparentheses",
        "redundantcast",
        "redundantimports",
        "redundantwhitespace",
    ],
    LABEL_TESTING: [
        "missing tests", "no tests", "add tests", "test is missing",
        "weak tests", "weak test", "weak assertions",
        "test coverage", "coverage", "coverage drop",
        "unit test", "unit tests", "integration test", "integration tests",
        "regression test", "regression tests",
        "incorrect tests", "wrong assertion", "assertion is wrong",
        "failing test", "failing tests", "brittle test",
        "test case", "test cases", "edge-case tests",
        "coverage threshold", "threshold not met", "test","tests","testing","tester","suite","suites","unittest","pytest","junit",
        "nunit","mocha","jest","karma","spec","specs","specification","assert","asserts",
        "assertion","assertions","expect","expects","expectation","matcher","matchers",
        "mock","mocks","mocking","stub","stubs","fake","fakes","spy","spies","fixture",
        "fixtures","harness","harnesses","coverage","coverages","coveragexml","coveragerc",
        "threshold","thresholds","metric","metrics","flaky","flake","regression",
        "regressions","oracle","oracles","golden","snapshot","snapshots","baseline",
        "baselines","scenario","scenarios","case","cases","edgecase","edgecases","negative",
        "positive","smoke","sanity","integration","e2e","endtoend","manual","automation",
        "automated","ci","matrix","slow","fast","timing","timeout","timeouts","rerun",
        "reruns","rerunnable","skipped","skip","xfail","fail","fails","failed","failing",
        "pass","passed","passing","nonpassing","nontestable","untested","uncovered","ignored",
        "ignore","reliable", "missing tests",
        "no tests added",
        "no test added",
        "no unit tests added",
        "missing unit tests",
        "missing integration tests",
        "missing regression tests",
        "missing test coverage",
        "weak tests",
        "weak test coverage",
        "insufficient test coverage",
        "low test coverage",
        "test coverage too low",
        "coverage threshold not met",
        "coverage drop",
        "coverage decreased",
        "tests incorrect",
        "incorrect test assertions",
        "test asserts wrong behavior",
        "tests do not match spec",
        "tests not updated",
        "tests not adjusted",
        "tests not covering edge cases",
        "no negative test cases",
        "no error path tests",
        "happy path only",
        "flaky test",
        "flaky tests",
        "unstable tests",
        "test suite failing",
        "test failure in ci",
        "regression test missing",
        "regression scenario not tested",
        "no e2e tests for this",
        "no end to end tests",
        "no integration coverage",
        "no database tests",
        "mocking is incorrect",
        "test double misuse",
        "test data incomplete",
        "test data unrealistic",
        "test naming unclear",
        "test case missing for new behavior",
        "add tests for this change",
        "add unit tests for this path",
        "please strengthen tests",
        "please add regression tests","testmissing",
        "testabsent",
        "testomitted",
        "testnotadded",
        "testshallow",
        "testweak",
        "testincomplete",
        "testpartial",
        "thedgecaseuntested",
        "edgepathuntested",
        "errorpathuntested",
        "boundaryuntested",
        "boundarynotcovered",
        "happyonly",
        "happyonlypath",
        "noerrorcoverage",
        "noexceptioncoverage",
        "nonegativepath",
        "onlypositivepath",
        "onlyhappyflow",
        "onlysmoketest",
        "nointegrationtest",
        "noe2etest",
        "nosystemtest",
        "noperformancetest",
        "noloadtest",
        "nocoveragetarget",
        "coveragedrop",
        "coveragemissing",
        "coverageregression",
        "coveragelow",
        "coverageslip",
        "coveragegap",
        "coveragehole",
        "coverageunreported",
        "coverageignored",
        "testflakiness",
        "testintermittent",
        "testrandomfailure",
        "testnondeterministic",
        "testorderdependence",
        "testorderbug",
        "testtimingbug",
        "testtimeout",
        "testhang",
        "testsuitehang",
        "testsuitecrash",
        "testsuitebroken",
        "testenvdrift",
        "testenvmissing",
        "testdataleakage",
        "testpollution",
        "testsharedstate",
        "testcoupling",
        "assertweak",
        "assertnarrow",
        "asserttoobroad",
        "asserttoospcific",
        "asserterror",
        "expectedwrong",
        "expectedvaluewrong",
        "expectedmismatch",
        "oracleweak",
        "oracleincorrect",
        "oracleblindspot",
        "falsepositivepass",
        "falsepass",
        "falsenegativefail",
        "failingbutcorrect",
        "passingbutwrong",
        "snapshotoutdated",
        "snapshotdrifted",
        "goldenfiledrift",
        "goldenfileoutdated",
        "testrefactoringneeded",
        "testdatabrittle",
        "testrandomseed",
        "testcasegenerator",
        "testscenariogap",
        "testmatrixgap",
        "noapprovaltest",
        "noregressionguard",
        "missingnonregression",
        "incompletefixture",
        "fixturebug",
        "fixtureleak",
        "testnamingbad",
        "testsmisleading",
        "manualtestonly",
        "manualverificationonly",
        "nonautomatedtest"
    ],
    LABEL_DESIGN: [
        "architecture", "architectural", "design", "design misfit",
        "coupling", "tight coupling", "cohesion", "layering", "layered",
        "boundary", "service boundary", "api boundary",
        "abstraction", "bad abstraction", "leaky abstraction",
        "dependency cycle", "circular dependency",
        "duplicated functionality", "duplicate logic", "duplicate code",
        "performance budget", "performance regression",
        "hot path", "hotpath", "maintainability", "hard to maintain",
        "system evolution", "extensibility", "scalability",
        "belongs in a different class", "belongs in the service layer",  "architecture","architectural","architected","design","designs","designed","pattern",
        "patterns","antipattern","antipatterns","component","components","module","modules",
        "layer","layers","layering","tier","tiers","service","services","domain","domains",
        "entity","entities","aggregate","aggregates","boundary","boundaries","boundarycontext",
        "context","contexts","coupling","decoupling","cohesion","encapsulation","abstraction",
        "abstractions","interface","interfaces","api","apis","contract","contracts",
        "ownership","owner","owners","orchestration","orchestrator","orchestrators",
        "coordination","coordinator","coordinators","strategy","strategies","factory",
        "factories","repository","repositories","controller","controllers","view","views",
        "presenter","presenters","usecase","usecases","dto","dtos","mapper","mappers",
        "adapter","adapters","gateway","gateways","port","ports","plugin","plugins",
        "extension","extensions","hook","hooks","monolith","monolithic","microservice",
        "microservices","scalable","scalability","performance","latency","throughput",
        "capacity","maintainable","maintainability","testability","reusability","flexibility",
        "extensibility","responsibility","architecture issue",
        "architectural violation",
        "breaks our architecture",
        "violates architecture boundaries",
        "violates service boundary",
        "crosses service boundary",
        "layering violation",
        "violates layering",
        "wrong layer for this logic",
        "logic in wrong layer",
        "poor design",
        "design smell",
        "design misfit",
        "not aligned with design",
        "not aligned with module design",
        "bad abstraction",
        "leaky abstraction",
        "abstraction leak",
        "mixes concerns",
        "mixed responsibilities",
        "violates separation of concerns",
        "too much responsibility in this class",
        "god object",
        "overly complex method",
        "method too complex",
        "class too big",
        "high coupling",
        "tight coupling",
        "low cohesion",
        "circular dependency",
        "dependency cycle",
        "service dependency cycle",
        "wrong ownership of data",
        "wrong ownership of logic",
        "business logic in controller",
        "business logic in view",
        "persistence logic in controller",
        "domain logic in repository",
        "performance budget violation",
        "performance regression risk",
        "scalability concern",
        "memory footprint concern",
        "maintainability issue",
        "hard to maintain code",
        "not extensible design",
        "violates domain model",
        "inconsistent domain model",
        "api surface too large",
        "leaking internal details",
        "violates clean architecture",
        "not following hexagonal architecture",
        "not following ddd principles","designsmell",
        "designanti pattern",
        "designproblem",
        "designflaw",
        "designregression",
        "designtradeoffbad",
        "designnonoptimal",
        "designovercomplicated",
        "designoversimplified",
        "designrigid",
        "designnotflexible",
        "designtightcoupling",
        "designloosecontract",
        "designinconsistency",
        "designunsound",
        "architecturaldrift",
        "architecturalerosion",
        "architectureerosion",
        "architecturemisfit",
        "architecturemismatch",
        "architectureviolation",
        "layerbreach",
        "layercrossing",
        "layerleak",
        "layerinversion",
        "domainleak",
        "domainpollution",
        "infrastructureleak",
        "persistenceleak",
        "uileak",
        "controllerfat",
        "servicefat",
        "godservice",
        "godsaga",
        "godcontroller",
        "godmanager",
        "godclass",
        "megaobject",
        "multipleresponsibilities",
        "noncohesive",
        "overcoupled",
        "underabstracted",
        "overabstracted",
        "prematureabstraction",
        "wrongabstraction",
        "wrongownership",
        "wrongresponsibility",
        "wrongboundary",
        "wrongmoduleboundary",
        "wrongservicelayout",
        "wrongaggregation",
        "wronggranularity",
        "wronglayer",
        "wronglayerplacement",
        "modulecycle",
        "packagecycle",
        "cyclicdependency",
        "circularreference",
        "fatinterface",
        "fatapi",
        "bloatedapi",
        "leakyapi",
        "chatt yapi",
        "chattyprotocol",
        "overchatt yservice",
        "oversharedschema",
        "sharedschema",
        "tightschema",
        "performancehotspot",
        "perfhotspot",
        "perfregression",
        "latencyregression",
        "latencyhotspot",
        "memoryhotspot",
        "memoryregression",
        "allocationheavy",
        "gcpressure",
        "noncachefriendly",
        "nonstreamable",
        "nonscalableapproach",
        "horizontalnon scalable",
        "verticalnon scalable",
        "maintenancetricky",
        "maintenancenightmare",
        "testabilitypoor",
        "testabilityissue",
        "configurationhard",
        "extensionhard",
        "changescary",
        "designnotdocumented",
        "designnotaligneddoc",
        "designnotalignedspec",
        "architecturedecisionignored",
        "adrignored"
    ],
    LABEL_PROCESS: [
        "process", "policy", "policies", "governance",
        "cla", "license agreement", "dco", "sign-off", "sign off",
        "required reviewers", "required reviewer", "code owners",
        "code owner", "changelog", "change log",
        "coverage threshold", "status check", "required status check",
        "unmergeable", "cannot be merged", "merge blocked",
        "branch protection", "protected branch",
        "release notes", "template not filled", "missing template",
        "metadata", "follow the template",        "process","processes","workflow","workflows","policy","policies","governance",
        "rules","rule","guideline","guidelines","standard","standards","procedure",
        "procedures","protocol","protocols","checklist","checklists","template","templates",
        "approval","approvals","approve","approved","reviewer","reviewers","assignee",
        "assignees","stakeholder","stakeholders","owner","owners","ownership","governor",
        "gate","gates","gatekeeper","gatekeepers","signoff","signoffs","signoffed",
        "signedoff","cla","dco","license","licenses","licensing","compliance","compliant",
        "noncompliant","violation","violations","breach","breaches","policycheck",
        "securitycheck","qualitycheck","statuscheck","statuschecks","blocked","blocking",
        "unmergeable","protected","protection","branch","branches","branching","workflowrule",
        "required","mandatory","optional","deprecated","sunset","lifecycle","milestone",
        "milestones","sprint","sprints","iteration","iterations","backlog","grooming",
        "triage","triaging","escalation","escalations","handoff","handoffs","release",
        "releases","releasing","freeze","freezes","audit","audits","auditlog","auditing",
        "traceability","retention","process violation",
        "policy violation",
        "does not follow our process",
        "does not follow project policy",
        "governance rule not satisfied",
        "governance gate failing",
        "missing cla signature",
        "cla not signed",
        "missing dco sign off",
        "dco sign-off missing",
        "missing signoff",
        "sign-off footer missing",
        "required reviewers not assigned",
        "required reviewer missing",
        "required approvals missing",
        "approval from maintainer required",
        "approval from code owner required",
        "codeowners approval missing",
        "missing changelog entry",
        "changelog entry required",
        "release notes missing",
        "jira ticket missing",
        "missing ticket reference",
        "missing issue link",
        "branch protection rule failing",
        "branch protection violation",
        "target branch not allowed",
        "cannot merge into master directly",
        "must open pr against develop branch",
        "missing label on pull request",
        "required label missing",
        "required status checks not configured",
        "coverage threshold policy",
        "performance budget policy",
        "security policy violation",
        "license policy violation",
        "third party license not allowed",
        "missing security review",
        "requires security review",
        "requires design review",
        "process step skipped",
        "template not followed",
        "pull request template not filled",
        "checklist not completed",
        "checklist items unchecked",
        "governance check failed",
        "unmergeable due to policy",
        "policy check is red","processbreach",
        "processgap",
        "processskipped",
        "processnotfollowed",
        "processoffpath",
        "processnoncompliant",
        "processdeviation",
        "processexception",
        "processviolationbug",
        "reviewpolicyviolation",
        "reviewprocessskipped",
        "reviewmissingstep",
        "reviewerrolemissing",
        "approvalworkflowbroken",
        "approvalmissing",
        "approvermissing",
        "mandatoryreviewmissing",
        "mandatorycheckmissing",
        "mandatorygatefailed",
        "mandatorypolicy",
        "policyenforced",
        "policybreach",
        "policynotmet",
        "policynotfollowed",
        "policynotrespected",
        "licenserequired",
        "licensenotchecked",
        "licensenotdeclared",
        "licensetextmissing",
        "licenserestriction",
        "compliancegap",
        "complianceissue",
        "compliancebreach",
        "securitygatefailed",
        "securityreviewmissing",
        "securitysignoffmissing",
        "qualitygatefailed",
        "qualitygatenotmet",
        "statuscheckmissing",
        "statuscheckskipped",
        "requiredstatusabsent",
        "requiredgateabsent",
        "branchpolicybreach",
        "branchrulebreach",
        "branchprotectionbreak",
        "directpushforbidden",
        "directpushattempt",
        "directpushblocked",
        "rebaserequired",
        "rebasenotdone",
        "syncwithmainrequired",
        "syncwithdeveloprequired",
        "outofdatebranch",
        "outofdatewithtarget",
        "outofdatewithmain",
        "missingtemplatedata",
        "missingprtemplate",
        "templateignored",
        "checklistignored",
        "checklistincomplete",
        "unfilledcheckitem",
        "ticketidmissing",
        "ticketidwrong",
        "issuereferencemissing",
        "issuereferencewrong",
        "milestonemissing",
        "milestonewrong",
        "sprintnotlinked",
        "epicnotlinked",
        "audittrailgap",
        "auditmissingentry",
        "auditeventmissing",
        "retentionpolicyignored",
        "governanceflag",
        "governancewarning",
        "governancefail",
        "releaseprocessbreach",
        "releasecheckmissing",
        "releaseapprovermising",
        "rolloutpolicybreach",
        "rollbackpolicybreach"
    ],
    LABEL_TOOLING: [
        "tool", "tooling", "automation", "automated job",
        "script", "generator", "code generator", "scaffolding",
        "linter command", "test runner", "release job", "ci job",
        "wrong flag", "wrong option", "command line option",
        "runner configuration", "runner config",
        "cache", "cache issue", "stale cache",
        "pre commit", "pre-commit", "precommit",
        "docker script", "wrapper script",
        "rerun the generator", "rerun the code generator",        "tool","tools","tooling","cli","command","commands","script","scripts","scripting",
        "automation","automations","job","jobs","runner","runners","executor","executors",
        "task","tasks","hook","hooks","precommit","prepush","postmerge","prebuild",
        "postbuild","plugin","plugins","extension","extensions","wrapper","wrappers",
        "launcher","launchers","binary","binaries","utility","utilities","helper","helpers",
        "generator","generators","scaffold","scaffolds","scaffolding","linter","linters",
        "formatter","formatters","bundler","bundlers","webpack","rollup","gulp","grunt",
        "make","cmake","ninja","maven","gradle","ant","sbt","yarn","npm","pnpm","pip",
        "poetry","tox","pytest","runnercli","coverage","profiler","profilers","analyzer",
        "analyzers","inspector","inspectors","debugger","debuggers","emulator","emulators",
        "simulator","simulators","monitor","monitors","daemon","daemons","service","services",
        "agent","agents","orchestrator","orchestrators","scheduler","schedulers","pipeline",
        "pipelines","workflow","workflows","cron", "tool misuse",
        "misuse of tool",
        "tooling issue",
        "tooling error",
        "automation error",
        "automation script failed",
        "script failure",
        "release script failed",
        "build script failed",
        "wrong cli flag",
        "wrong command line option",
        "incorrect flag passed",
        "incorrect command used",
        "using wrong command",
        "using wrong script",
        "linter command failed",
        "linter not run",
        "formatter not run",
        "generator not run",
        "code generator not executed",
        "forgot to run code generation",
        "migration not generated",
        "database migration missing",
        "schema migration missing",
        "tool configuration wrong",
        "misconfigured tool",
        "misconfigured linter",
        "misconfigured ci job",
        "ci job misconfiguration",
        "wrong test runner",
        "wrong test command",
        "wrong environment for tool",
        "stale cache",
        "tool cache corrupted",
        "tool cache outdated",
        "pre-commit hook failed",
        "pre commit hook failed",
        "pre-commit not configured",
        "local tooling not aligned with ci",
        "local tool version mismatch",
        "different tool versions",
        "missing plugin for tool",
        "missing tool dependency",
        "automation pipeline broken",
        "release job failed",
        "artifact upload failed",
        "artifact download failed",
        "package publishing script failed",
        "docker push script failed",
        "deployment script failed","toolmisconfiguration",
        "toolmisuse",
        "toolabuse",
        "tooldrift",
        "toolversiondrift",
        "toolversionconflict",
        "toolversionold",
        "toolversionunsupported",
        "tooloptionwrong",
        "toolargumentwrong",
        "toolflag wrong",
        "toolinvocationbug",
        "toolinvocationerror",
        "toolinvocationmissing",
        "toolpathmissing",
        "toolexecutablemissing",
        "toolexecutionfailure",
        "tooltimeout",
        "toolhang",
        "toolcrash",
        "toolstacktrace",
        "toolstackoverflow",
        "toolmemoryexhausted",
        "toolresourceexhausted",
        "toolpermissions",
        "toolpermissiondenied",
        "scriptcrash",
        "scriptmissingdep",
        "scriptmissingbinary",
        "scriptnotexecutable",
        "scriptpathwrong",
        "scriptargwrong",
        "scriptflagwrong",
        "scriptenvmissing",
        "scriptenvmismatch",
        "scriptconfigdrift",
        "automationbreak",
        "automationpipelinebreak",
        "automationjobfail",
        "automationworkflowfail",
        "automationschedulerfail",
        "autodeployfail",
        "autoreleasefail",
        "autotagfail",
        "autobumpfail",
        "automergeblocked",
        "automergefail",
        "generatornotrun",
        "generatorstale",
        "generatoroutofdate",
        "generatoroutputmissing",
        "generatoroutputdirty",
        "migrationsnotapplied",
        "migrationsnotgenerated",
        "migrationsoutofdate",
        "schemaoutofsync",
        "schemadrift",
        "dbtoolmismatch",
        "dbmigrationtool",
        "dbmigrationfailure",
        "profilernotconfigured",
        "analysistoolmissing",
        "analysistoolmisconfigured",
        "scanjobfail",
        "scanresultmissing",
        "lintjobfail",
        "formatjobfail",
        "bundlejobfail",
        "bundlerconfigerror",
        "packagetoolerror",
        "packagetoolmissing",
        "publishscriptfail",
        "publishtoolfail",
        "dockerbuildscriptfail",
        "dockerrunscriptfail",
        "helmtoolfail",
        "clustertoolfail",
        "monitoringtoolfail",
        "loggingtoolfail",
        "tracingtoolfail",
        "metricscollectorfail",
        "agenttooloffline",
        "agenttoolcrash",
        "workeragenttool",
        "toolinggatefailure",
        "toolingincompatibility",
        "toolingsetupmissing",
        "toolingsetupbroken"
    ],
    LABEL_OTHER: [
        # explicitly non-defect or unclear content
        "looks good to me", "lgtm", "thanks", "thank you",
        "nice refactor", "nice change", "good job",
        "general question", "not a rejection",
        "small nit", "nit but not blocking",
        "just a comment", "just a thought",
    ],
}


def rule_based_label(text: str) -> Optional[str]:
    """
    Try to infer the label using simple keyword matching based on the
    taxonomy definitions and their explanations.
    Returns a label or None if there is no strong signal.
    """
    t = text.lower()
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
