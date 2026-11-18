# Product Requirements Document (PRD): LocalSight CCTV Scrubber (MVP v1.0)

**Owner:** Kane Arvin Ching  
**Target Release:** MVP v1.0 (Validation & Early Access)  
**Status:** Ready for Engineering Kickoff  

---

## 1. Problem Statement & Market Context
Homeowners and small residential landlords with local NVR/SD-card CCTV systems (Hikvision, Dahua, Tapo, Reolink, Xiaomi) waste 20–60 minutes manually scrubbing through linear video timelines on mobile apps to investigate incidents (e.g., vehicle scratches, missing parcels, gate intrusions), while basic motion alerts trigger hundreds of false positives from street traffic and swaying trees.

* **Target Beachhead:** Tech-comfortable homeowners, car enthusiasts, and small residential landlords (1–4 properties) owning **4 to 16 local/offline cameras** who refuse recurring cloud subscriptions ($10–$30/month per camera) yet want natural-language search over their local footage.
* **Pricing Strategy:** One-time lifetime license ($29–$69) using local offline AI models (Ollama) or user-supplied Bring-Your-Own-API-Keys (BYOK), avoiding churn from infrequent residential security incidents.
* **Core Solution:** **LocalSight** is a privacy-first, local-first web application that ingests exported CCTV files, filters out static background frames locally, and lets users search hours of home footage in seconds using natural-language queries.

---

## 2. Goals & Success Metrics
* **Primary Goal:** Cut post-incident home CCTV review time from 30+ minutes of manual scrubbing to under 60 seconds without requiring new camera hardware or mandatory cloud video uploads.
* **Metric 1 (Indexing Speed):** Process and index a 6-hour 1080p CCTV file into event storyboards in **$\le 30$ seconds** on a standard consumer laptop.
* **Metric 2 (Noise Reduction):** Strip **$\ge 90\%$** of static/background frames via local motion and Region-of-Interest (ROI) polygon filtering prior to AI inference.
* **Metric 3 (Search Precision):** Achieve **$\ge 85\%$ Top-3 recall** on benchmark home incident queries (*"delivery parcel"*, *"person near parked car"*, *"gate opened"*).
* **Metric 4 (Validation Conversion):** Convert **$\ge 5$ of 20** early-access homeowner testers to submit real footage and **$\ge 3$** to commit to a one-time license ($39) or BYOK credit pack.

---

## 3. Core User Stories
* **US-01 (Zero-Setup Ingestion):** As a homeowner, I want to drag and drop raw `.mp4`, `.mkv`, or `.dav` NVR exports into a local browser UI so I can review incidents without installing complex NVR servers, RTSP streams, or Docker networks.
* **US-02 (Noise Exclusion):** As a homeowner, I want to draw a polygon over my driveway or porch so passing street cars and swaying trees do not flood my event timeline.
* **US-03 (Semantic Search):** As a homeowner, I want to type plain-English questions (e.g., *"who approached my parked SUV between 1 AM and 4 AM?"*) and jump directly to matching video timestamps.
* **US-04 (Privacy Control):** As a privacy-conscious user, I want to choose between 100% offline local AI inference (Ollama) or my own cloud API key (Gemini Flash) so my home footage never touches a vendor's cloud storage.

---

## 4. Functional Requirements (MVP Scope)
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-01** | **Universal Video Ingestion & Remuxing** | Must Have | System accepts `.mp4`, `.mkv`, and `.dav` files up to 10 GB; automatically remuxes H.265/`.dav` streams via FFmpeg into browser-playable H.264 `.mp4`. |
| **FR-02** | **Interactive ROI Polygon Masking** | Must Have | User can click $\ge 3$ points on the video frame to define a detection zone; OpenCV masks out pixels outside the polygon during frame diffing. |
| **FR-03** | **Local Motion Clustering & Contact Sheets** | Must Have | Backend samples at 1 FPS, groups contiguous motion into events (4s gap tolerance), and generates $2 \times 3$ timestamp-burned JPEG grids. |
| **FR-04** | **Hybrid VLM Semantic Querying** | Must Have | User enters a text prompt and selects **Local (Ollama)** or **BYOK (Gemini Flash)**; backend returns ranked JSON matches with confidence scores and 1-sentence summaries. |
| **FR-05** | **Timestamp Jump Player** | Must Have | Clicking any matched event card seeks the HTML5 `<video>` player to `startSec` and begins playback immediately. |

---

## 5. Non-Functional Requirements & Guardrails
* **Data Privacy:** Zero server-side cloud retention; all raw videos, thumbnails, and SQLite indexes remain strictly on `localhost` unless the user explicitly enables BYOK mode (which transmits only compressed $2 \times 3$ contact sheets).
* **Unit Economics:** BYOK cloud inference cost must remain **$\le \$0.03$ per 6-hour video** via contact-sheet batching (6 frames per API call).
* **Usability:** Zero YAML configuration files; entire setup runs via a single local launch command or desktop wrapper.

---

## 6. Out of Scope (Non-Goals for v1.0)
* **24/7 Live RTSP Camera Streaming & Real-Time Push Alerts:** LocalSight is strictly a post-incident forensic scrubber, not a replacement for Frigate or Blue Iris live NVR recording.
* **Multi-Camera Cross-Tracking:** v1.0 processes one camera export file at a time; automated person re-identification across multiple cameras is deferred to v2.0.
* **Hosted SaaS Video Storage:** No proprietary cloud video hosting or monthly per-camera subscription billing in v1.0.

---

## 7. Key Risks & Mitigations
* **Infrequent Homeowner Usage ("Once-a-Year Problem"):** Homeowners experience security incidents sporadically; mitigate churn by offering a **one-time lifetime license ($39)** rather than a recurring monthly subscription.
* **Proprietary DVR File Formats (`.dav`, `.h265`):** Older DVRs export non-standard streams that fail in HTML5 `<video>` tags; mitigate via automatic local FFmpeg remuxing on ingest.
* **Night-Vision IR Grain:** Low-light black-and-white footage reduces fine object recognition; mitigate via user-drawn ROI masks and prompt tuning focused on macro actions (person/vehicle entry, parcel placement).

---

## 8. Release & Validation Milestones
* **Week 1:** Build local FastAPI + React MVP (`ingest.py`, `events.py`, `vlm_client.py`, `main.py`, `App.tsx`) and benchmark accuracy on daytime and nighttime driveway clips.
* **Week 2:** Run Concierge Validation Test in homeowner/DIY security groups with 20 early testers; collect accuracy feedback and willingness-to-pay signal.
* **Week 3:** Package one-click local installer (Tauri/PyInstaller) and launch v1.0 early-access build.