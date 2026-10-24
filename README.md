# Trackey

**Trackey** is a modular, declarative computer vision pipeline framework for real-time video analytics, multi-object tracking, spatial intelligence, and identity re-identification (Re-ID).

```
Camera / RTSP Stream
        │
        ▼
┌──────────────────┐       ┌────────────────────────────────────────────────────────┐
│  Capture Thread  │ ────► │ Bounded Frame Queue (Drop-Oldest Under Slow Inference) │
└──────────────────┘       └───────────────────────────┬────────────────────────────┘
                                                       │
                                                       ▼
                                            ┌────────────────────┐
                                            │  PipelineExecutor  │
                                            └──────────┬─────────┘
                                                       │
                     ┌─────────────────────────────────┼──────────────────────────────┐
                     ▼                                 ▼                              ▼
            ┌─────────────────┐               ┌─────────────────┐            ┌─────────────────┐
            │  YOLO Detector  │               │ DeepSort Tracker│            │  Spatial Index  │
            └────────┬────────┘               └────────┬────────┘            └────────┬────────┘
                     │                                 │                              │
                     └────────────────►  Immutable FrameContext  ◄────────────────────┘
                                                       │
                                   ┌───────────────────┴───────────────────┐
                                   ▼                                       ▼
                         ┌───────────────────┐                   ┌───────────────────┐
                         │   People Counter  │                   │     OSNet Re-ID   │
                         └─────────┬─────────┘                   └─────────┬─────────┘
                                   │                                       │
                                   ▼                                       ▼
                         ┌───────────────────┐                   ┌───────────────────┐
                         │  Event Publisher  │                   │ FAISS / Postgres  │
                         └─────────┬─────────┘                   └───────────────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
      ┌──────────────────┐                   ┌──────────────────┐
      │  JSON Webhooks   │                   │ Email Alerting   │
      └──────────────────┘                   └──────────────────┘
```

---

## Key Features

- **Declarative YAML Pipelines**: Define computer vision DAG workflows without writing boilerplate glue code.
- **Resilient Bounded Queue Ingestion**: Dedicated capture threads isolate slow model inference from video feeds with automatic drop-oldest policies and auto-reconnect backoff for RTSP cameras.
- **Immutable State Machine**: Execution flows through frozen `FrameContext` dataclasses, guaranteeing branch isolation and deterministic DAG execution.
- **Spatial Intelligence**: Ray-casting point-in-polygon zone testing with normalized resolution-agnostic coordinates.
- **Identity Re-ID with Eviction**: Vector indexing with FAISS and persistent metadata repositories with configurable TTL retention policies.
- **Enterprise Integrations**: Out-of-the-box support for JSON Webhooks, Prometheus metrics export (`:8000`), and structured JSON logging.

---

## Installation

### Prerequisites
- Python 3.10+
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 12+ (for model inference)

### 1-Step Setup Script
The project includes a unified setup script that manages base dependencies, PyTorch, and deep Re-ID tools cleanly:

```bash
# Clone the repository
git clone https://github.com/3mad10/Trackey.git
cd Trackey

# Run the installer
bash scripts/install.sh all
```

---

## Quickstart

### 1. Configure Environment
Copy the example environment file and set your credentials:
```bash
cp .env.example .env
```

### 2. Run with Trackey CLI
```bash
# Run any YAML pipeline specification
trackey run base_pipeline.yaml
```

Once running:
- **Visual Display**: Live OpenCV window with zone boundaries, bounding boxes, trajectories, and real-time FPS HUD.
- **Prometheus Metrics**: Available at `http://localhost:8000/metrics`.
- **Logs**: Human-readable on stdout; structured JSON written to `trackey.log`.

---

## Docker Deployment

Run Trackey with PostgreSQL using Docker Compose:

```bash
docker compose up --build
```

---

## Configuration Reference (`pipeline.yaml`)

A Trackey pipeline is defined using seven top-level sections:

```yaml
# 1. Video input source
source:
  type: camera  # or rtsp, video
  params:
    index: 0
    width: 1280
    height: 720

# 2. Geometric zones & boundaries (Normalized 0.0 - 1.0)
scene:
  zones:
    - name: Entrance
      polygon:
        - [0.1, 0.3]
        - [1.0, 0.3]
        - [1.0, 1.0]
        - [0.1, 1.0]

# 3. Directed Acyclic Graph (DAG) Pipeline
pipeline:
  nodes:
    - name: detectorA
      type: detector
      processor: yolo
      source: primary
      params:
        weights: yolov8n.pt

    - name: trackerA
      type: tracker
      processor: deepsort

    - name: spatial_analyzer
      type: spatial_index

    - name: people_counter
      type: analyzer
      processor: counter
      zone_name: Entrance
      params:
        target_classes: [person]
        metrics: [current, cumulative]

    - name: publisherA
      type: publisher
      events:
        - type: count_exceeded_event
          extract:
            count: analytics.people_counter.person.cumulative
            zone_name: analytics.people_counter.zone_name

  edges:
    - from: detectorA
      to: trackerA
    - from: trackerA
      to: spatial_analyzer
    - from: spatial_analyzer
      to: people_counter

# 4. Asynchronous Subscribers
subscribers:
  - type: webhook
    listen: [count_exceeded_event]
    params:
      url: https://api.mycompany.com/v1/events
      timeout: 2.0

# 5. Persistent Identity Stores
stores:
  people:
    type: identification
    repository:
      backend: postgres
      params:
        host: localhost
        port: 5432
        database: trackey
        user: trackey
        password: ${DB_PASSWORD}
        retention_days: 90  # Embedding TTL purge on startup

# 6. Overlay Renderer
renderer:
  type: opencv

# 7. Output Sinks
sinks:
  - type: display
    params:
      window_name: Trackey
```

---

## Running Unit Tests

The test suite validates geometry ray-casting, DAG execution and merging, and analyzer state machines without requiring GPU or camera hardware:

```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
