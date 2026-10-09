360° Review of Trackey

Phase 1: My Understanding of the Product

What it is [FACT]: A YAML-configured, Python computer-vision pipeline framework. A source (camera/video/RTSP) feeds a DAG of nodes (detector → tracker → spatial index → analyzer → condition → publisher, plus ReID/identify nodes). Events go to an async bus with mail/null subscribers, and frames are rendered to an OpenCV window or image sink.

Who it's for [ASSUMPTION]: Developers and integrators who want to compose detection, tracking, zone analytics and re-identification without writing glue code. The README says "multi-platform" and "activity analysis", but nothing defines a customer.

What a user can do today [FACT]: Edit base_pipeline.yaml and run Engine.from_config(...) from the __main__ block. They get people counting in a polygon zone, an email when a count condition is met, and body ReID against FAISS plus Postgres.

Data flow [FACT]: Frame → FrameContext (immutable, replaced per node) → PipelineExecutor (Kahn-style queue, branch skipping) → PipelineResult (pydantic) → sinks. Events bypass the context and go straight to the EventBus thread.

Deployment [FACT]: There is none: no Dockerfile, CLI, service wrapper, or entry point. It runs as a script on the developer's machine, with an OpenCV window as the only UI.

Dependencies [FACT]: Ultralytics YOLO, DeepSORT, torchreid (OSNet), insightface, mediapipe, FAISS, Postgres, Gmail SMTP.

Direction [ASSUMPTION, from TODOs and half-built files]: Face recognition and association, watchlists, a unified identity across body and face, and a pluggable everything-registry.

What Is Still Unclear
Who is the customer: hobbyists, integrators building bespoke systems, or end-user businesses such as retail or security?
Is this a library, a service, or an appliance?
Single camera or many? The engine is hard-wired to one source.
Is identity recognition (people re-ID and faces) the core value or a side experiment? It dominates the recent code but not the README.
Target hardware: a GPU workstation, Jetson-class edge, or cloud?
Is any commercial intent real? That decides the licensing and privacy sections below.
Phase 2: Customer / User Review
Persona	Goal	Today's verdict
Python CV developer (the only realistic user now)	Compose a pipeline quickly	Plausible, but only after reading the source
Integrator	Deploy for a client site	Blocked: no service mode, headless mode, auth or monitoring
Operator / non-technical	See counts, get alerts	Impossible: no UI beyond cv2.imshow
Admin	Manage cameras, users, retention	Nothing exists

Journey, condensed:

Before purchase: The README is two lines. There's no demo, benchmark or screenshots. [FACT]
Install: There are at least four conflicting dependency definitions. requirements.txt is a frozen environment that includes a self-referencing editable trackey install. requirements/ pins torch 2.6 with numpy 1.26, requirements_dev.txt pins torch 2.10 with numpy 2.4, and pyproject.toml pins yet another set. Both faiss-gpu (requirements) and faiss-cpu (pyproject) appear. A stray file named 1.8.0 (a pip output) is committed. [FACT]
First use: Plugin and registry classes register only via import side effects. Engine.from_config works only if someone has already imported every plugin, which currently happens only inside the __main__ block of engine.py. A fresh user's first run raises "Unsupported source type… Available: []". [FACT]
Failure: A camera read failure raises, RTSP end-of-stream becomes "source exhausted", and there is no reconnect. A node exception kills the whole engine. [FACT]
Upgrade, rollback, support: There are no versions, migrations (the Postgres schema is created inline), diagnostics or health endpoints.
Exit and lock-in: Embeddings are raw float32 bytes in BYTEA. Exportable, but undocumented.
Phase 3: Customer-Readiness Gap Analysis (severity: Blocker / High / Med)
Area	Current state	Severity	Recommendation
Installation	Conflicting pins, no lockfile, side-effect imports	Blocker	One pyproject with extras, a lockfile, and a single trackey run config.yaml entry point that auto-imports built-ins
Headless / UI	OpenCV window only	Blocker for any non-dev	Add an MJPEG/WebRTC or REST/WebSocket output, then a minimal web dashboard
Auth / authorization	None	Blocker if networked	Defer until an API exists, then API keys at minimum
Privacy / compliance	Persistent biometric embeddings with no consent, retention or deletion path	Blocker commercially	Retention TTL, delete-by-identity, opt-in identity mode. Check biometric rules in your target markets (GDPR, Illinois BIPA, Egypt's Personal Data Protection Law)
Licensing	Ultralytics is AGPL-3.0. InsightFace pretrained models (buffalo) are, as I understand it, non-commercial-research only. Verify both.	Blocker commercially	Document it now. Offer permissive alternatives (e.g. RT-DETR/YOLOX/ONNX models, your own trained face model)
Reliability	No reconnect, no frame-drop policy, serial execution, unbounded queues	High	See Phase 8
Monitoring	print("latency") per node per frame, and log handlers attached at import	High	Structured logging, Prometheus metrics (FPS, per-node latency, queue depth)
Testing	pytest is a dependency, but I see no tests. Many __main__ blocks contain stale, broken demo code	High	Unit tests for geometry, counter, executor routing, stores. A golden-video integration test
Docs	README of two lines	High	Quickstart, config reference, architecture doc
Config UX	Validation partly broken (see below)	Med	Pydantic config models
Data mgmt	No retention, backup or migration	Med	Alembic or versioned schema
API / integrations	Only SMTP. Webhook and image-storage subscribers are mentioned in docs but I see no plugin	Med	Webhook subscriber is the highest-value integration
Billing, user mgmt, permissions	Absent	N/A today	Don't build until a customer exists
Phase 4: Product Requirements (independently derived)

Must have [CUSTOMER REQ, for a "deployable CV pipeline framework"]

One-command install and run, headless-capable
Config validation with readable errors
Source reconnect and frame-drop policy
Webhook or API output for results and events
Docs and example configs
Tests and CI
Correct, documented counting semantics (entered / exited / current)

Should have

Docker image with GPU variant
Metrics endpoint
Per-class detection filtering
Foot-point zone membership
Configurable thresholds (currently hard-coded 0.4 in DeepSORT)

Could have

Multi-camera support
Face recognition and watchlists
Pose and keypoints
A web UI

Should NOT have (yet)

A "unified identity manager" across body and face before single-modality ReID is correct
Billing or user management
More pluggable registries. You already have five parallel plugin/registry systems (detector, tracker, sink/source plugins, store plugins, subscribers). This is the main source of accidental complexity.
Phase 5: User Stories (top priorities)
As an integrator, I want to install and run a config with one command, so that I can evaluate in minutes. AC: clean venv → trackey run base_pipeline.yaml works. Edge: no GPU (OSNet and RetinaFace default to cuda and will crash). Failure: missing weights give a clear message. P0.
As an operator, I want an alert when occupancy exceeds a limit, so that I can act. AC: one event per crossing, not per frame. Edge: ID switches inflating counts. P0. Note: today the condition re-fires every frame once true and relies on subscriber throttling. [FACT]
As an integrator, I want events delivered to my system via webhook, so that I can automate. P0.
As an operator, I want the system to recover from a camera drop, so that I don't babysit it. P0.
As a privacy-conscious deployer, I want to disable or purge identity data, so that I stay compliant. P1.
As a developer, I want to add a custom detector in <30 lines, so that the framework is worth adopting. P1 (this is your real differentiator).
Phase 6: Workflows

Core workflow: Configure → run → observe → alert.

Configure: Editing a 200-line YAML where the format hint says on: but the code requires listen:. [FACT: SUB_CFG_FORMAT vs _validate_required_fields] The renderer config includes lines.StartLine for a line that doesn't exist, and stores.people.type: identification, which is never read.
Run: Import-order dependence, then crashes on missing GPU or weights.
Observe: Only an OpenCV window.
Alert: Gmail SMTP with an app password. This works, but it's a demo-grade channel.
Recovery: Restart manually. Track IDs and counts reset, because the cumulative counter lives in memory.
Phase 7: UX / Usability

Could a customer operate this without asking you? No. The main reasons:

Errors surface as tracebacks.
Terminology is inconsistent: listen vs on, zone vs zone_name, source as both an input and a detection tag.
Defaults are demo-specific (a hardcoded email list in the committed base config).
There are no empty, loading or failure states because there's no UI.
Dangerous actions (persisting biometrics every frame) happen silently.
Phase 8: Architecture

What's good [FACT]:

Immutable FrameContext with functional updates
DAG executor with branch routing and a wiring pre-validation pass
Clean interfaces (Detector, Tracker, Analyzer, FeatureExtractor, OutputSink)
Separation of EmbeddingStore / IdentityRepository / EmbeddingRepository, with Identifier owning the match decision
Normalized coordinates throughout
A genuinely good instinct to make the pipeline declarative

Real defects I found [FACT, unrun]:

Executor _merge duplicates tracks and drops state. With multiple terminal nodes (your base config has two: the publisher and reid_person), tracks are concatenated, so every track appears twice in the final context and renderer. per_track and zone_memberships aren't merged at all, so identities never reach the renderer or result.
Duplicate-node detection never fires. All nodes are validated before any is added to self.nodes, so if node_name in self.nodes is always false.
Wiring validation is order-based, not ancestry-based. body_embedding "requires" zone_memberships, which is satisfied by an unrelated branch that happens to sort earlier. The prefix match (startswith) can also wrongly pass.
Identification runs per track, per frame. Each match does a Postgres load, an upsert with commit, a FAISS add and an embedding-row insert. That's several DB round trips per person per frame. A non-match creates a brand-new identity, so one unmatched person can spawn dozens of identities. Cache tracker_id → global_id and re-identify every N frames or only on low confidence.
Negative crop indices in OSNet. to_pixel_xyxy isn't clamped, so a box touching the left or top edge gives a negative slice start. NumPy wraps it silently and you get a wrong or empty crop, which can crash the extractor.
ImageStorageSink.open() and NullViewer.open() return None. Engine._open_all treats falsy as failure, so both raise "Failed to open sink". The sink also isn't registered, writes files named by camera_id (a file path for video sources, so invalid), and doesn't create the output directory.
CameraSource ignores width/height. The base config comment says "all nodes see 1280×720", but only VideoFileSource resizes.
RtspSource.read calls Frame(frame, width=w, height=h). Frame has no such fields, so this raises TypeError.
Detection schema silently drops data. Detection has no metadata, keypoints or face_bbox fields. Pydantic ignores extras, so RetinaFace landmarks and MediaPipe keypoints vanish. head_region.py has no base class and no imports, and calls track.model_copy on a dataclass. arcface.py is a copy of retina.py that registers a class still named RetinaFaceDetector.
DeepSortTracker.self.tracks is never pruned. It's a memory leak on long runs, and the stored track never receives the identity assigned downstream.
Zone membership uses bbox center. For people, the foot point (bottom-center) is the standard. Center-based membership misplaces anyone near a zone edge.
MailSubscriber._build_body guards against np.ndarray but not Frame. IntrusionEvent and SnapshotEvent carry a Frame, so the email would embed a huge array repr.
Hygiene: logging.basicConfig and file handlers are attached at import time, so duplicate logs and a trackey.log in the CWD. There are leftover prints in the executor. DSNs are built by string concatenation, so passwords with special characters break. The Postgres embedding plugin passes repo_name= to a constructor that takes table=. Personal file paths and email addresses are committed.

Structural concerns [REC]:

Dependency direction is inverted. core/register.py and core/registries/store.py import from plugins/. Core should not know plugins exist.
Registration by import side effect is the root cause of the first-run failure. Use entry points or one explicit trackey.builtins import.
Five overlapping registry/plugin mechanisms is too many. Collapse to one Registry[T] generic.
Serial single-threaded loop. A slow node stalls capture, so RTSP latency grows without bound. You need a capture thread, a bounded queue and a drop-oldest policy. drop_strategy is accepted by VideoFileSource but unused.
PipelineResult is a pydantic model wrapping dataclasses and ndarrays. It revalidates detections and tracks every frame. Make it a plain dataclass.
Config: replace the ad-hoc dict validation with pydantic models. You already depend on pydantic.
One DB connection per repository, not thread-safe, with no reconnect or pooling.
Verdict

Could this become a commercially viable product as-is? No. Three reasons:

The product definition is missing. There's no customer, no vertical and no UI.
The dependency licenses (AGPL YOLO, non-commercial face models) conflict with a closed commercial product, and the biometric handling has no compliance story.
The space is crowded: Frigate, NVIDIA DeepStream/Metropolis, Roboflow Inference and supervision, and Ultralytics Solutions all cover pieces of this.

Could it be an excellent open-source framework or portfolio-grade project? Yes. The core architecture is better than most hobby CV projects. The differentiator to lean into is declarative, testable, hot-swappable pipelines, not "features".

Recommended order of work:

Fix defects 1, 2, 4, 5, 6, 8 and 9. These are correctness bugs, and they're cheap.
Unify dependencies, ship an entry point and a Docker image, and write tests plus CI.
Pick one wedge (for example "occupancy/people-flow analytics with webhooks") and cut identity and face work until it ships.
Add the capture thread with bounded queue, a webhook subscriber and a metrics endpoint.
Decide the licensing and privacy position before anyone else depends on it.]

Phase 8 verdict: Architecture

Mixed: under-engineered in the runtime, over-engineered in the extensibility layer.

Under-engineered: The runtime is a single synchronous loop with no capture/processing split, no backpressure, no supervision and no lifecycle (close() is never called on detectors or the DB).
Over-engineered: There are five registry/plugin mechanisms, a NODE_REGISTRY and NODE_WRAPPERS that nothing uses, and a FACTORY_ROUTER that nothing uses. All of this exists for a project with no second user yet.
Coupling: Core imports plugins (register.py, registries/store.py), which inverts the dependency direction. Registration depends on import side effects.
Phase 9: Codebase
Location	Problem	Impact	Change
factories/pipeline.py	God builder with a hand-rolled NODE_BUILDERS string→method map. The __main__ block calls insert_node/remove_node, which don't exist.	Hard to extend, and the file lies about its API	Node-type registry; delete the dead __main__
core/register.py	Pulls in plugins, so it imports the world	Circular-import risk, slow import, core depends on plugins	Registries in core, plugins register themselves, one builtins.py loads them
stores/identification.py	Four parallel maps (_int_to_global, _embedding_ids, _next_int_id, repo) hold the same truth	Drift on any failure mid-upsert	One EmbeddingIndex object per modality owning id↔global_id↔vector
Identifier.identify	Mutates an Identity loaded from the repo, which has no embeddings (see Phase 10)	Eviction never fires with Postgres	Make the store own embeddings, not Identity
interfaces/store.py	IdentificationStore.upsert(identity) but callers pass modality=, added=, evicted=	The ABC doesn't describe the real contract	Fix the signature
trackers/deepsort.py	defaultdict() with no factory, never pruned, and replace() on a slots=True dataclass that mutates history copies each frame	Leak, wasted copies	Prune on delete; mutate the deque in place
data/schemas/*	Two Point classes, one pydantic and one with different semantics (feature.py clamps)	Silent confusion	One geometry module
schemas/pipeline.py	PipelineNode is a @dataclass(ABC) with a custom __eq__/__hash__, and NodeCfg is unused	Noise	Delete or use
__main__ blocks	Stale demo code in nearly every file (YoloDetector is undefined in osnet.py, stray readFrame() calls, Frame(width=...) that doesn't exist)	Misleads readers, can't run	Move to examples/ or delete
Typing	-> any, Optional[...] = [] mutable defaults in Scene, missing UUID import in memory.py	Latent bugs	mypy in CI (already a dependency)
style builders	RendererBuilder constructs AnalyticsStyle.panels as a dict but the dataclass declares list	Works by accident	Align types
renderer.py	ZonesStyles.per_zone is filled from cfg.items() where show (a bool) is skipped, but the default style is never configurable	Per-zone fallback only	Expose default:

Duplication: arcface.py is a verbatim copy of retina.py. MemoryRepository and the Postgres repo duplicate lookup logic. Source plugins repeat their validation boilerplate.

What is clean: Naming is consistent, docstrings explain why, and functions are short. The code reads like it was written by someone who cares.

Phase 10: Domain & Data

Entities [FACT]: Frame, Detection, Track, Identity, Zone/Line, Event, FrameContext, PipelineResult.

Findings:

Source-of-truth conflict for embeddings. Identity.embeddings (in-memory, capped at 10) is one truth. The FAISS index plus the embeddings table is another. PostgresIdentityRepository.load() returns an Identity with an empty embeddings dict. In Identifier.identify, match.add_embedding(...) therefore always appends to an empty bucket, never evicts, and upsert(..., added=embedding) pushes a new vector to FAISS and the DB for every matched track on every frame. The "max 10 per identity" invariant only holds with MemoryRepository. At 20 FPS and 5 people that's roughly 100 new vectors per second, so millions per day. This is the most serious data defect. [FACT, unrun]
Track identity is not on the stored track. DeepSortTracker.self.tracks keeps the pre-identity Track, and replace() on the next frame starts from it. Identity is re-attached each frame by the manager node, so identity never persists across frames unless re-identification re-runs every frame (which is what causes issue 1).
Track.bbox vs history: the renderer draws history[-1], the analyzer filter uses bbox. They are equal today, but there are two fields for one concept.
IDs: tracker_id is annotated int but is a DeepSORT string in practice. Zone maps are typed UUID.
Timestamps: FrameContext.timestamp is time.monotonic() (not wall-clock), events use datetime.now, and tracks use UTC. There are three clocks, so events can't be correlated with video time.
No schema versioning for the embeddings table, no model/version recorded alongside vectors (re-embedding with a new model silently corrupts matching), and no migration tool.
Counting semantics: "cumulative" means distinct tracker IDs ever seen in the zone. Every ID switch inflates it, and it never resets. [FACT]
Phase 11: API & Integration Design

There is no external API. [FACT] The only contracts are the YAML schema, the Python interfaces and the event dataclasses.

YAML contract: unversioned, not schema-validated, and internally inconsistent (on vs listen, zone vs zone_name, source overloaded).
Event contract: BaseEvent is a Python dataclass with no serialization, no event ID (so no idempotency), no schema version, and Frame embedded in some events. An external consumer can't use it.
Delivery semantics: Mail is fire-and-forget on a worker thread with no retry, timeout or dead-letter handling. smtplib.SMTP without timeout= can hang the single bus worker forever and block every other subscriber. [FACT]
Idempotency: none. Events fire per frame while the condition holds, and throttling is per (event type, camera), not per entity.

Could an integration team use it without reading the source? No. [REC] Define a versioned JSON event envelope (id, type, version, ts, camera, payload), a webhook subscriber with timeout and retry, and a JSON Schema for the config.

Phase 12: Performance & Scalability

Current expected scale [ASSUMPTION]: one 720p camera on one GPU workstation.

Level	Actual bottleneck [FACT-ish]	Likely bottleneck	Theoretical
1×	None measurable (no benchmarks exist). The executor prints metadata every frame.	YOLO + OSNet on the same thread; OSNet runs on every track every frame	Pydantic revalidation of PipelineResult
10× (10 cameras)	Architecture can't do it: one Engine, one source, one loop	GPU contention from per-process model copies	Python GIL
100×	Needs a service/queue architecture	Postgres commit-per-embedding, FAISS flat index with linear scan	FAISS GPU vs CPU

Specific, justified items:

Per-frame OSNet: ReID on every track every frame is the likely top cost. Run it on new tracks, then every N frames or when the crop is good (size, not occluded). That's justified by the DB growth in Phase 10 as well.
Postgres commit per row in PostgresEmbeddingRepository.save: batch, or use a write-behind queue.
faiss-gpu vs faiss-cpu: at ≤ 100k vectors a flat CPU index is fine. GPU FAISS adds a painful install for no gain here.
Image copies: render() copies the full frame, and trails copy the full image per segment (img.copy() inside the loop). That's O(segments × pixels) and a real cost when trails are on.
No GPU/CPU fallback: device="cuda" defaults will crash CPU-only machines.

Don't optimize anything else until there's a benchmark.

Phase 13: Reliability
Failure	Behavior today [FACT]	Needed
Camera unplugs	CameraSource.read raises, run() only catches KeyboardInterrupt and StopIteration, so the process dies (finally closes)	Reconnect with backoff
RTSP dies	Returns None → "Source exhausted" → clean exit, which looks like success	Distinguish EOF from error and reconnect
Any node throws	Whole engine stops	Per-node error policy (skip frame / disable node / fail)
Postgres down at start	ConnectionError, which is fine	Retry or degrade to memory
Postgres down mid-run	Unhandled psycopg2 error kills the engine, and an aborted transaction state leaves the connection unusable (no rollback)	Rollback, reconnect, pool
SMTP down	Caught and logged, but a hang blocks the bus worker	Timeouts
Process restart	Counts, tracks and IDs reset. Embeddings reload from Postgres (a good design).	Persist counters if they matter
Disk full / log growth	trackey.log is unrotated	RotatingFileHandler
Bad config	Mostly good failure (fail fast at build), but some silently ignored	Strict schema
Memory growth	DeepSortTracker.tracks, FAISS vectors, event queue (unbounded Queue())	Bounds

No health check, watchdog or supervisor exists. Can a customer recover without you? No: no runbook, no restart policy, no diagnostics.

Phase 14: Security
Item	Finding	Severity
Biometric data at rest	ReID/face embeddings stored unencrypted in Postgres with no retention policy, access control or consent mechanism. Embeddings are personal data, and legally biometric in many jurisdictions.	High (privacy/compliance)
Secrets	SMTP/DB passwords via ${ENV} substitution is a good pattern. But base_pipeline.yaml commits real-looking email addresses, and a password string is the default in __main__ blocks.	Low-Med
DSN construction	String concat with no URL-encoding. A password containing @///: breaks parsing and can mis-route the connection.	Low
Env substitution	Substituted into raw YAML text before parsing, so a value containing newlines or : can alter config structure (config injection). Only matters if env values are untrusted.	Low
Model downloads	MediaPipe model via urlretrieve with no checksum into a CWD-relative path. Ultralytics and InsightFace download weights at runtime.	Medium (supply chain)
Pickle-based weights	torch.load in torchreid and .pt checkpoints can execute code. Fine if you only load your own, risky for user-supplied paths.	Medium
Network exposure	None today (no listener). That's a strength, and it changes the day you add an API.	n/a
RTSP URLs	Credentials in URLs; camera_id/path flow into filenames and emails	Low
Email contents	Event fields are dumped verbatim, and could later include identities or frames	Medium once identity is live
SQL injection	Table names use psycopg2.sql.Identifier, values are parameterized. Good.	None found
YAML	yaml.safe_load. Good.	None found
Dependencies	~300 pinned packages across conflicting files. Pins include black==26.x etc., which I can't verify. No audit step.	Unknown, needs validation
Licensing	AGPL (Ultralytics) and restricted-use face models, per my earlier note (verify)	Legal, not security

I found no critical vulnerabilities, and I'm not claiming any. The real risk is privacy and compliance, not exploitation.

Phase 15: Testing

Today [FACT]: I see no test directory. pytest, pytest-cov and mypy are declared but unused.

Missing, in priority order:

Executor: branch routing, skip propagation, merge semantics. The duplicate-tracks defect would be caught immediately.
Geometry: Polygon.contains, Line.crossed. The ray-casting code has an xinters variable that can be unbound when p1y == p2y and x > p1x. Property tests would find it. (Note: xinters is only assigned inside the if p1y != p2y branch, then read in if p1x == p2x or x <= xinters. A horizontal edge with p1x != p2x hits an UnboundLocalError or a stale value.) [FACT]
Counter: metrics, peak and cumulative semantics, ID switches.
Identification: Identifier + CompositeStore with in-memory fakes. This pins down the unbounded-growth bug.
Config: golden YAML → built graph; every example config must build.
Builders: unknown types, missing fields, env-var failures.
Integration: a 10-second golden video with a stub detector, asserting counts and events.

Strategy [REC]:

Because Detector, Tracker and FeatureExtractor are interfaces, write deterministic fakes and test the framework without GPUs. This is the architecture's best testability asset.
Mock only network edges (SMTP, Postgres via testcontainers or sqlite-compatible fake).
Add a nightly job with real weights on one clip.
Add a performance test recording FPS per node on a fixed clip so regressions are visible.
Phase 16: Observability & Support

Customer says "the system isn't working." Can you answer:

Question	Answerable?
What failed?	Partly (a traceback or log line)
When?	Log timestamps exist, but the file is unrotated and shared across runs
Which component?	PipelineNodeError names the node. Good.
Which customer / version?	No: no version string, no build info
What configuration?	No: effective config isn't logged (and must be redacted if it were)
Reproducible?	No: no recorded input or seed
Auto-recover?	No

Logging: basicConfig is called at import in two modules, and MailSubscriber calls it too, so handlers attach multiple times and you get duplicate lines. The per-node print("latency") floods stdout. Metrics: none, though latency is already collected into ctx.metadata, which is a great head start. Missing: a /health endpoint, FPS counters, queue depth, last-frame-age watchdog, and a trackey doctor command (GPU, weights, DB, SMTP reachability).

Phase 17: Deployment & Operations

New customer receives it today. What must happen [FACT]:

Clone the repo (no package published).
Pick one of four dependency files; most combos conflict (torch 2.6 / 2.10, numpy 1.26 / 2.4, faiss-gpu / faiss-cpu).
Install CUDA, cuDNN, onnxruntime-gpu matching versions.
Install torchreid from Git (editable, pinned to a commit).
Install and configure PostgreSQL manually, create user and DB.
Create a Gmail app password and set env vars.
Hand-edit YAML (including zone polygons in normalized coordinates, with no tool to draw them).
Manually import every plugin module, since no entry point does so.
Run python trackey/core/engine.py with the config in the CWD.
Download model weights implicitly at first run (network required).

Automate: 1 and 2 (package + lockfile), 4, 5 (docker-compose with Postgres), 8 (builtins), 9 (CLI), 10 (trackey models pull with checksums), and 7 (a zone-drawing helper). Also add CI (lint, mypy, tests), semver, a changelog and DB migrations.

Phase 18: Project Management

Blockers: product definition, dependency licensing, missing entry point, the unbounded-embedding bug.
Dependencies: webhook/API → versioned event schema → executor fix. Identity features → data-retention policy.
Risks:

Scope sprawl: the identity subsystem (face, body, watchlist, unified manager) is large and unvalidated.
Solo bus factor.
Licensing discovered late.
Dependency rot (the stack is pinned to ultralytics==8.3.40, torch 2.6).

Unknowns: customer, hardware, camera count, accuracy requirements.
Decisions required: library vs service, open-source vs commercial, identity in scope or out, multi-camera.
Parallel: CI/tests, packaging, docs, webhook subscriber, bug fixes.
First: correctness fixes → packaging → one wedge use case.

Phase 19: Delivery Roadmap
Phase 0 (Stabilize): fix the listed defects, add tests for the executor, geometry and identification, kill duplicate log handlers, delete dead code, resolve dependency files into one.
Phase 1 (Customer MVP): trackey run, a built-in plugin loader, headless mode, a webhook subscriber, Docker image, a config schema with good errors, one polished example (people counting in a zone with a webhook alert), and a real README.
Phase 2 (Production-ready): reconnect and bounded-queue runtime, metrics and health, rotating structured logs, DB migrations, retention controls, CI with a golden-clip test.
Phase 3 (Early customers): a minimal web view (MJPEG + counts), API keys, multi-camera via multiple processes, a zone editor, audit log.
Phase 4 (Scale): a service split (ingest / inference / event delivery), shared inference server, a queue (Redis/NATS), pgvector or Qdrant if identity persists.
Phase 5 (Mature): plugin marketplace, model management, fleet/config management, SSO. Don't plan these before Phase 3 produces demand.
Phase 20: Technical Debt
Issue	Category	Severity	Customer Impact	Technical Impact	Effort	Priority
Unbounded embedding growth (Postgres identities)	Data	Critical	Disk/DB blowup, slow matching	Index bloat	S	P0
Executor _merge duplicates tracks/loses state	Correctness	High	Wrong render/result	Hidden bugs	S	P0
No entry point, side-effect registration	Packaging	High	Can't start	Fragile imports	M	P0
Conflicting dependency files	Install	High	Install fails	Unreproducible	S	P0
RTSP Frame(...) TypeError, camera ignores resize	Correctness	High	Source unusable	n/a	S	P0
Sinks' open() returns None	Correctness	High	Sinks unusable	n/a	XS	P0
Detection schema drops metadata/keypoints	Data	Med	Face/pose features dead	Misleading code	S	P1
Polygon xinters unbound	Correctness	Med	Wrong/crashing zones	n/a	XS	P1
No tests/CI	Quality	High	Regressions	Slow change	M	P0
Five registry mechanisms	Design	Med	none	Complexity	M	P1
Core→plugins dependency	Design	Med	none	Cycles	M	P1
Serial runtime / no reconnect	Reliability	High	Outages	n/a	M	P1
Logging side effects, prints	Ops	Med	Noisy logs	n/a	XS	P1
Pydantic PipelineResult overhead	Perf	Low	none	Frame cost	XS	P2
Unpruned DeepSortTracker.tracks	Memory	Med	Long-run leak	n/a	XS	P1
Stale __main__ demos	Hygiene	Low	none	Confusion	XS	P2
Pay now: the six P0 rows.
Can remain: pydantic overhead, hand-written style builders, PipelineNode hash/eq oddities.
Accept intentionally: the frozen-context copy-on-write pattern. Its cost is small next to inference and it buys safety.
Phase 21: Architecture Decisions (expensive to change)
Decision	Current	Alternatives	Trade-off	Recommendation
Deployment model	Script	Library + CLI; service; both	Library is cheapest, service is what customers buy	Library + CLI now, service wrapper later
Execution model	Sync, single thread	Capture thread + bounded queue; multiprocess per camera	Complexity vs latency/robustness	Capture thread now, process-per-camera for multi-cam
Config	Hand-validated YAML	Pydantic schema, versioned	Upfront effort	Pydantic + version: key now (cheap now, painful later)
Plugin system	5 registries, import side effects	Entry points / one generic registry	Setup vs discoverability	One Registry[T] plus entry points
Identity model	Identity carries embeddings, store hides them	Store-owned vectors, Identity metadata only	Rewrite	Store-owned embeddings (fixes the Phase 10 bug)
Persistence	Postgres + FAISS rebuilt at start	pgvector; SQLite for edge	Dependency weight	SQLite default, Postgres optional (cuts install pain)
Event contract	Python dataclasses	Versioned JSON envelope	Adds a layer	Do it before any external consumer exists
Coordinates	Normalized floats	Pixels	n/a	Keep normalized (it decouples resolution)
Multi-tenancy	Not considered	Per-tenant stores/config	Heavy	Don't build until a SaaS decision exists
Hardware	CUDA assumed	Pluggable device / ONNX runtime	Effort	Make device a config value with CPU fallback
Phase 22: Customer-Readiness Score
Area	Score	Reason
Customer value	3	A real problem class, but no defined customer or differentiation
Product clarity	2	README is two lines, scope drifts into identity
UX	1	OpenCV window and YAML
Core functionality	5	Detect/track/zone/count/alert path is plausible, with bugs
Architecture	6	Strong core idea, unneeded plugin layers, weak runtime
Code quality	5	Clean style, many latent defects and dead code
Reliability	2	No recovery anywhere
Security	4	No exposure, but unprotected biometrics and downloads
Testing	0	None found
Deployment	1	Manual, conflicting dependencies
Documentation	1	Docstrings only
Observability	2	Latency captured but only printed
Supportability	1	No diagnostics or versioning
Scalability	2	Single pipeline, single process
Maintainability	5	Readable, but untested and with duplicated registries
Extensibility	7	Interfaces and DAG are genuinely good, if over-built
Developer Project Score: 6/10. Better than typical hobby CV code, with a sound dataflow model.
Production Product Score: 1.5/10.
Customer Experience Score: 1/10.
Phase 23: Red Flags
Unbounded embedding/DB growth in identity matching with Postgres.
No definable customer or wedge, and the product's most complex part (identity) is the least validated.
Licensing and biometric-privacy exposure (verify AGPL and model terms).
Not installable or runnable by anyone but you: conflicting dependencies and import-side-effect registration.
Executor merge defect corrupts results in any multi-terminal graph, including your own base config.
No tests, so every fix is unverified.
Runtime dies on any camera/DB hiccup.
Event delivery can hang the single bus worker (no SMTP timeout).
Competes with strong incumbents without a stated advantage.
Bus factor of one.
Phase 24: Strengths
Immutable FrameContext with functional updates. Nodes can't corrupt each other's state, which makes the DAG testable and parallelism-ready.
Wiring validation at build time (get_inputs/get_outputs + topological check). Misconfigurations fail before the first frame. Fix the ancestry-versus-order flaw and keep it.
Normalized coordinates everywhere. Pipelines and zones are resolution-independent, so zone config survives a camera change.
Clean interface seams (Detector, Tracker, FeatureExtractor, OutputSink, Subscriber). Fakes are trivial, and that is your testing lever.
Store decomposition (vector index / metadata repository / embedding repository) with the matching policy in Identifier. Right separation of concerns, even though the data-ownership detail needs fixing.
ThrottledSubscriber as a decorator and the async EventBus that keeps slow delivery off the frame loop.
Per-node latency capture already in context metadata. It is the foundation for real metrics.
Safe SQL and YAML handling (sql.Identifier, parameters, safe_load) and env-var secrets with fail-fast on missing variables.
Embedding rebuild from the repository at startup, so FAISS is a derived index, not a source of truth.
Phase 25: My Ideas vs Independent Recommendations

You supplied no ideas in this prompt. These are decisions visible in your code and positioning, treated as hypotheses.

My Idea	Problem it attempts to solve	Evidence	Verdict	Better alternative
Position as mid-market CV pipeline framework	Teams that outgrow scripts but not DeepStream	None in repo (no users, no customer data)	Needs customer validation	Choose a wedge, e.g. people-flow analytics with webhooks
Declarative YAML DAG	Reusable, no-glue pipelines	Implemented and works	Keep (add schema + versioning)	n/a
Registry/decorator plugin pattern	Extensibility	5 mechanisms, side-effect imports	Modify	One generic registry + entry points
Full factory family + node wrappers	Construction uniformity	FACTORY_ROUTER/NODE_WRAPPERS/NODE_REGISTRY unused	Reject (delete dead ones)	Single builder path
EventBus + ThrottledSubscriber	Alerts without blocking frames	Works	Keep (add timeouts, retry, ids)	n/a
Identity resolution (body + face + watchlist + unified manager)	Persistent people identity	Zero evidence of demand, high legal exposure	Defer	Finish body ReID correctly, hide behind a flag, drop face/watchlist until validated
Per-modality int ID counters, composite PK	Multiple embedding types	Works	Keep	n/a
Face detector (RetinaFace) + association node	Link faces to tracks	head_region.py doesn't import or run	Defer	Per-track cropped detector only if face ID is validated
Pose/keypoints (MediaPipe)	Behavior analytics	Output is discarded by Detection	Defer	Not in MVP
Multi-platform claim (README)	Reach	CUDA-only defaults, Windows/Linux paths mixed	Modify	State supported platforms honestly
Async capture/render sinks	Throughput	Not present	Add (my recommendation)	Capture thread + bounded queue
Phase 26: If I Took Ownership Tomorrow
Freeze: all new features, identity/face/pose work, and the plugin-system redesigns.
Immediately fix: embedding growth, executor merge, sink open() returns, RTSP/camera frame handling, polygon xinters, duplicate-node check, the DSN encoding, logging side effects.
Remove: NODE_REGISTRY/NODE_WRAPPERS/FACTORY_ROUTER, stale __main__ blocks, arcface.py (until it differs from retina.py), the stray 1.8.0 file, committed personal emails and paths, three of four dependency files.
Redesign: identity data ownership, the runtime loop (capture/bounded queue/supervision), the event envelope, config schema, and plugin discovery.
Document: a README that sells and shows (GIF, 10-line quickstart), config reference, architecture overview, "how to write a detector", licensing and privacy notes.
Test: the executor, geometry, counter, identification and one golden-clip e2e test, all using fakes.
Automate: CI, Docker build, release tagging, dependency audit, DB migrations.
Ask customers: the Phase 27 list.
Refuse to build: a UI framework, billing, multi-tenant SaaS, cloud orchestration, a face-recognition product, a model zoo.
Build first: trackey run + one polished people-counting-with-webhook scenario that works on a clean machine in 15 minutes.
Phase 27: Questions for Real Customers

Problem validation

What do you do today to count or analyze people, vehicles or objects on video?
What decision does it drive, and what does a wrong count cost you?
Do you already own cameras, and what kind?

Workflow
4. Who watches the output, and how often?
5. What triggers an alert, and who should receive it?
6. How do you define the zones or lines today?

Usability
7. Would you edit config files, or do you need a UI?
8. Who sets it up, and who maintains it afterwards?

Missing features
9. Do you need identity (same person across cameras or days) or just counts and flows?
10. Do you need reports and history, or only live alerts?

Deployment
11. On-prem, edge device or cloud? What hardware is available?
12. How many cameras, and at what resolution/FPS?

Reliability
13. What uptime do you need? What happens when a camera drops?
14. What accuracy is "good enough," and how would you measure it?

Security / privacy
15. Are you allowed to store biometric or personal data? Who approves it?
16. What retention period is acceptable?

Integrations
17. What systems must receive events (webhook, MQTT, VMS, Slack, BI tool)?
18. Do you already use a video management system?

Pricing / commercial
19. What do you pay today (software, integrator)? Per camera, per site, or flat?
20. Would you pay for a framework or for a finished solution?
21. Would an open-source (AGPL-compatible) product be acceptable to you?

Support
22. Who supports it when it breaks, and what response time do you expect?

Highest-leverage: 1, 9, 11, 15, 20, 21. Each can change the product direction.

Phase 28: Final Product Gap
Current State	Missing Capability	Customer Impact	Required Work	Priority
Script run from __main__	CLI, package, builtins loader	Can't start	Entry point + packaging	P0
4 conflicting dependency files	Reproducible install	Can't install	One pyproject + lock + Docker	P0
No tests	Regression safety	Fixes break things	Fakes + executor/geometry/identity tests + CI	P0
Known correctness bugs	Trustworthy results	Wrong counts/render	Fix list from Phase 20	P0
OpenCV window only	Headless output + results API	Can't integrate	Webhook, JSON events, optional MJPEG	P0
Dies on any fault	Supervision, reconnect, bounded queues	Outages	Runtime rewrite	P1
Embeddings grow forever	Data lifecycle	DB blowup	Store-owned embeddings, caps, retention	P0
Personal data stored raw	Privacy controls	Legal/commercial block	Opt-in identity, TTL, delete API, encryption option	P1
AGPL/restricted models	License clarity	Can't sell	License audit + permissive alternatives	P1
2-line README	Docs/onboarding	No adoption	Quickstart, reference, examples	P1
Prints and unrotated logs	Observability	Unsupportable	Structured logs, metrics, health, doctor	P1
No customer or wedge	Product definition	Nothing to sell	10 customer interviews, one use case	P0
Final Verdict

1. Meaningful problem? The general problem (turn video into reliable, actionable events without writing glue code) is real. Your specific problem is undefined. Insufficient evidence, this needs validation.

2. Architecture appropriate? The core dataflow (immutable context, DAG, interfaces, normalized coordinates) is appropriate and good. The plugin layers are over-built and the runtime is under-built.

3. Implementation maintainable? Readable but unsafe to change: no tests, dead code, duplicated state, and defects in the executor and identity paths.

4. Usable by a real customer today? No.

5. What prevents adoption? No defined customer, no install path, no headless integration, no documentation, and no proof it beats existing tools.

6. What prevents production deployment? No reconnect or supervision, unbounded data growth, no observability, no tests, no health checks, and licensing and privacy questions.

7. Biggest technical risks

(a) Identity data growth and consistency.
(b) The synchronous runtime and its failure modes.
(c) Dependency and GPU stack fragility and licensing.

8. Biggest product risks

(a) No validated customer or wedge.
(b) Crowded market of incumbents and free alternatives (Frigate, DeepStream, Roboflow Inference, supervision).
(c) Biometric legal exposure if identity becomes the headline feature.

9. Biggest customer-experience problems

(a) Install and first run.
(b) YAML-only configuration with weak errors.
(c) No visibility or integration outputs beyond a desktop window.

10. Build next: the fixes, packaging, tests and one polished wedge scenario, delivered through trackey run, a webhook and Docker.

11. NOT build: face recognition, watchlists, pose, multi-tenancy, a UI framework, new registries, billing.

12. Redesign before continuing: identity data ownership, the runtime loop, the event/config contracts and plugin discovery.

13. Validate with customers: whether anyone needs identity at all, the hardware and deployment constraints, and whether a framework or a finished solution is what they'd pay for.

14. If this were my product: I would pick occupancy and people-flow analytics with webhook alerts as the wedge, ship a bulletproof version in a few weeks, and run ~10 interviews while doing it. I'd keep the pipeline engine as the internal moat and the open-source story. Identity would stay an experimental feature behind a flag until a customer asks for it by name.

🟠 Stabilize Before Expanding

The direction is plausible and the core design is good, so this isn't a redesign or a rethink. But the identity work was built on top of unfixed foundations (executor, data ownership, packaging, testing), and the product definition is missing. Stop adding features until the base is correct, installable and tested.

Top 10 Actions
Rank	Action	Type	Why Now	Customer Impact	Technical Impact	Effort	Priority
1	Do 8-10 customer/user interviews and choose one wedge use case	Customer validation	Everything else depends on it, and it's the largest unknown	Very high	Focuses scope	M	P0
2	Fix the P0 correctness bugs (executor merge, embedding growth, sinks, RTSP/camera, polygon, duplicate-node check)	Engineering	Results are wrong or unbounded today	High	High	S	P0
3	Build the test suite on fakes (executor, geometry, counter, identification) plus CI	Testing	Makes every later change safe	Medium	Very high	M	P0
4	Single pyproject, lockfile, trackey run entry point, builtins loader	DevOps	Nobody else can run it	Very high	High	M	P0
5	Webhook subscriber + versioned JSON event envelope (ids, timeouts, retry)	Product	The first real integration point	High	Medium	M	P0
6	License and privacy audit (AGPL, model terms, biometric law); decide the identity policy	Security	Cheap now, fatal later	High	Medium	S	P1
7	Redesign identity: store-owned embeddings, caps, retention, opt-in flag	Architecture	Prevents data-growth and compliance problems	Medium	High	M	P1
8	Runtime hardening: capture thread, bounded queue, reconnect, per-node error policy	Architecture	Needed before any unattended run	High	High	M	P1
9	README, quickstart, config schema (pydantic) with clear errors, Docker compose	Documentation	Turns a repo into a product evaluation	High	Medium	M	P1
10	Structured logging, metrics from existing latency data, health check, trackey doctor	Operations	You can't support what you can't see	Medium