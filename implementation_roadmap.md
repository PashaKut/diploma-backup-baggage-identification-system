# Implementation Roadmap And Session Handoff

This file is a working handoff for continuing the diploma project in a new Codex session with cleared context.

It captures:

- the original implementation roadmap split into phases 1-8
- what has already been implemented
- what has already been prototyped offline but is not yet integrated into the Flask app
- the current project structure and behavior
- important technical decisions and known limitations
- the next recommended step

## 1. Project Goal

Build a web-based demonstration system for an airport baggage identification backup subsystem.

Core idea:

- simulate baggage processing from registration to identification result
- try direct QR reading first
- if direct QR fails and backup mode is enabled, route to the backup subsystem
- show live pipeline progress in the UI
- store all per-session results in SQLite for comparison and presentation

Primary source of requirements:

- `system_description.md`

## 2. Roadmap By Phase

### Phase 1. Project Scaffold

Target:

- create the base folder structure and files
- add config, app state, requirements, data/model folders

Status:

- Done

Implemented:

- `app.py`
- `config.py`
- `state.py`
- `database/`
- `subsystem/`
- `static/`
- `data/`
- `models/`
- `.gitignore`
- `requirements.txt`

### Phase 2. Backend Foundation

Target:

- Flask app setup
- Flask-SocketIO setup
- SQLite/SQLAlchemy setup
- database reset on app start
- session image directory cleanup

Status:

- Done

Implemented:

- Flask app and Socket.IO are initialized in `app.py`
- DB engine/session factory are in `database/connection.py`
- ORM schema is in `database/models.py`
- DB tables are recreated on app start
- `data/sessions/` is cleaned on app start

### Phase 3. Basic UI, REST API, And Background Session Flow

Target:

- main page and results page
- mode API
- session start API
- background processing thread
- live WebSocket block/counter updates

Status:

- Done

Implemented:

- `GET /`
- `GET /results`
- `GET /api/modes`
- `POST /api/set_mode`
- `POST /api/start`
- `GET /api/results/sessions`
- `GET /api/results/session/<session_id>`
- `GET /static/sessions/<path>`
- background sorting thread in `app.py`
- live pipeline UI in `static/index.html` + `static/app.js`

### Phase 4. Registration Import

Target:

- import `data/registrations.csv`
- validate CSV structure
- insert `baggage_registrations`
- update registration block in real time

Status:

- Done

Implemented:

- exact CSV header validation in `subsystem/registration.py`
- empty-cell validation
- insert into `baggage_registrations`
- registration block count update

Important detail:

- registration import runs once per app start because the DB is recreated on startup

### Phase 5. Direct QR Reading

Target:

- replace fake direct success path with real QR decoding
- validate decoded LPN format
- store success/failure in `sorting_results`

Status:

- Done

Implemented:

- real QR reading wired into `subsystem/sorting.py`
- QR reader logic in `read_qr.py`
- success/failure result recording in `subsystem/results.py`
- decoded LPN is treated as authoritative

Important detail:

- the current QR reader is designed for the generated tag images you have now
- it will likely need tuning later for real photographed printed tags

### Phase 6. Results Page Drill-Down

Target:

- session summary table
- expandable per-photo rows
- image preview support

Status:

- Done

Implemented:

- expandable session rows on `/results`
- registration-centric detail fetch on demand
- one main results row per CSV registration for each session
- grouped matched-image galleries for successful registrations
- separate unidentified-images table per session
- thumbnails for original images
- processed image support if a later phase writes them
- fullscreen image modal

### Phase 7. Backup Subsystem Skeleton And Routing

Target:

- if direct QR fails in `enhanced` or `full`, route into backup flow instead of immediately failing
- add detection/enhancement/recognition control flow
- handle missing model files gracefully

Status:

- Not done

Current state:

- unreadable direct QR ends as `failed`
- optional blocks in the UI exist
- Flask integration modules are still stubs:
  - `subsystem/detection.py`
  - `subsystem/enhancement.py`
  - `subsystem/recognition.py`
- however, offline preparation work for the real backup subsystem has already been done:
  - a YOLO tag detector was trained and works well on the cropped-tag use case
  - a separate YOLO zones detector was trained for the 2 LPN text regions on the tag:
    - `LPN_right`
    - `LPN_bottom`
  - standalone scripts exist to evaluate detector performance, crop detections, and test OCR flow on the detected LPN zones
  - none of that is wired into `subsystem/` or the web app yet

### Phase 8. Real Backup Subsystem Implementation

Target:

- YOLO tag detection
- enhanced QR attempts on cropped tag
- CRNN fallback with OCR
- processed image saving

Status:

- Not done

Important clarification:

- the research/prototyping part of Phase 8 has started outside the web app
- the integration part of Phase 8 is still not done

## 3. What The App Can Do Right Now

- load modes from `config.py`
- start a session from the UI
- import baggage registrations from CSV
- iterate over images in `data/photos`
- copy originals into `data/sessions/<session_id>/original/`
- try direct QR reading
- validate decoded LPNs against the DB
- create one session result row per CSV registration
- attach all successfully decoded images to their registration by decoded LPN
- keep unmatched/unreadable photos in a separate unidentified-images table
- show live counters on the pipeline page
- show session history and registration-centric results on the results page
- preview session images in the results modal

## 3.1 What Has Been Proven Outside The App

- a YOLO tag detector has been trained for tag-region detection
- a YOLO LPN-zones detector has been trained for the 2 text zones on the baggage tag
- the currently good offline model runs are:
  - tag detector: `runs/detect/runs/train/detector_20260506_012226/weights/best.pt`
  - LPN zones detector: `runs/detect/runs/train/lpn_zones_20260509_003420/weights/best.pt`
- standalone scripts were added for dataset preparation, evaluation, crop extraction, and OCR checking:
  - `train_YOLO.py`
  - `evaluate_yolo_photos.py`
  - `crop_yolo_detections.py`
  - `crop_lpn_zone_detections.py`
  - `check_lpn_zone_ocr.py`
- OCR on cropped LPN zones works well enough in standalone testing to move forward with integration
- current standalone OCR logic:
  - detect `LPN_right` and `LPN_bottom`
  - crop both zones
  - run OCR on each zone separately
  - normalize to the expected LPN format
  - if both zone readings match, accept as high-confidence
  - if only one zone is valid, still use it
  - if both are present but disagree, flag as conflict

## 4. Current Important Files

### Core App

- `app.py`: routes, session start, background thread, image serving
- `config.py`: paths, modes, constants
- `state.py`: in-memory runtime state

### Database

- `database/connection.py`: engine, session factory, DB init/reset
- `database/models.py`: `BaggageRegistration`, `SortingResult`, `MatchedImageResult`, `UnidentifiedImageResult`

### Implemented Subsystem Parts

- `subsystem/registration.py`: CSV import and validation
- `subsystem/sorting.py`: photo-first session loop and direct QR path
- `subsystem/results.py`: session-result initialization, matched-image writes, unidentified-image writes, WebSocket counter updates
- `read_qr.py`: QR decoding logic

### Offline Model And OCR Prototypes

- `train_YOLO.py`: robust dataset split + YOLO training/evaluation script
- `evaluate_yolo_photos.py`: standalone detector evaluation on a photo folder
- `crop_yolo_detections.py`: standalone tag crop extraction from detector output
- `crop_lpn_zone_detections.py`: standalone crop extraction for the 2 LPN zone classes
- `check_lpn_zone_ocr.py`: standalone end-to-end zone detection + OCR + comparison workflow

### Frontend

- `static/index.html`: pipeline page
- `static/results.html`: results page
- `static/app.js`: page logic, Socket.IO handling, results expansion, image modal
- `static/style.css`: shared styling

## 5. Current Data And Naming Rules

### CSV

Expected columns:

`lpn,from,to,flight,date,passenger,class,pieces,type`

Current implementation requires this exact order.

### Photos

Preferred long-term naming convention:

- replace the space in LPN with `_`
- example: `0014 123456` -> `0014_123456.png`

Current implementation:

- photo filenames no longer control registration mapping
- every image in `data/photos` is processed regardless of filename
- successful mapping is done only by decoded QR LPN
- multiple images can attach to the same registered LPN in one session
- images that do not decode to a registered LPN are shown in a separate unidentified-images table

## 6. Important Decisions Already Made

### 6.1 Build A Vertical Slice First

The project was intentionally built in working slices:

- infrastructure first
- real QR reading second
- backup subsystem later

This keeps the demo runnable even before YOLO/OCR is implemented.

### 6.2 Decoded LPN Is Authoritative

During sorting, the result uses the QR-decoded LPN, not the expected LPN inferred from filename.

That is important because:

- filenames may be arbitrary
- later real photographed tags should still be judged by what is actually decoded

### 6.3 Oversized Image Safety

A freeze happened earlier when one replacement image was very large:

- `0014_567890.png`
- about `4284x5712`
- about `32 MB`

The old QR reader tried `x4` upscaling on such inputs and effectively stalled the worker.

Fix already implemented in `read_qr.py`:

- large images are resized to a bounded working size first
- unsafe upscale attempts are skipped
- per-image exceptions in `subsystem/sorting.py` are caught and converted into `failed` results

### 6.4 Verification Constraint In Codex Sandbox

In previous sessions, the Codex sandbox could not reliably execute the local Python interpreter from the virtual environment.

Meaning:

- code changes were made in the workspace
- runtime verification had to be done by the user locally

Useful local run commands:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

### 6.5 Integration Boundary Is Now Clear

The project now has 2 layers of progress:

- integrated web-app functionality:
  - phases 1-6
  - direct QR flow
  - results UI
- offline backup-subsystem prototypes:
  - YOLO tag detection
  - YOLO LPN zone detection
  - OCR and two-zone reconciliation logic

The next work should focus on moving those offline pieces into `subsystem/detection.py`, `subsystem/enhancement.py`, and `subsystem/recognition.py` without breaking the existing QR-first demo flow.

## 7. Current Dependencies

Current `requirements.txt` includes:

- `flask`
- `flask-socketio`
- `sqlalchemy`
- `python-dotenv`
- `opencv-python-headless`

Used in standalone prototype scripts / model work:

- `ultralytics`
- `easyocr`

Potentially still optional depending on future implementation details:

- `pandas`
- `Pillow`
- `qrcode`

`ultralytics` and `easyocr` should be treated as real project dependencies now, because they are required for the trained-model and OCR workflow that will be integrated in Phases 7-8.

## 8. Known Limitations

- backup subsystem is not implemented yet
- `enhanced` and `full` modes do not perform real backup recovery yet
- if direct QR fails, the photo is stored as unidentified and the registration remains `failed`
- processed images are not generated yet
- the trained YOLO/OCR components currently live only in standalone scripts, not in the Flask request/session pipeline
- no app-level model loading lifecycle has been implemented yet for the trained YOLO detectors or OCR reader
- registration import uses Python `csv`, not `pandas`
  - this is acceptable for now
  - the system description mentioned `pandas`, but switching later is optional unless strictly required
- the results page can display processed images, but none are produced yet

## 9. Recommended Next Step

The next correct step is Phase 7.

Practical target for the next session:

1. Stop sending unreadable direct QR images straight to `failed` in `enhanced` and `full`.
2. Implement backup routing control flow in `subsystem/sorting.py`.
3. Replace the `subsystem/detection.py` stub with real YOLO tag detection using the trained detector weights.
4. Save the tag crop into the session `processed/` flow or another integration-safe intermediate path.
5. Route the tag crop into:
   - enhanced QR attempts first
   - then, in `full` mode, LPN zone detection + OCR fallback
6. Move the already working standalone LPN OCR logic into app-integrated modules:
   - detect `LPN_right` and `LPN_bottom`
   - OCR both zones
   - accept matching readings
   - allow single-zone fallback
   - flag zone conflicts as failure
7. Keep graceful failure behavior if the model files or OCR dependencies are missing.

The big picture is no longer “design Phase 7 from scratch”; it is “integrate the already proven offline YOLO/OCR pipeline into the existing web app and results flow.”

## 10. Suggested Prompt For A New Session

Use something close to this:

> Read `implementation_roadmap.md` and `system_description.md` first, then continue with Phase 7 and Phase 8 integration. Preserve the existing working QR flow and results page. The current code already implements phases 1-6 in the Flask app. Offline prototypes already exist for YOLO tag detection, YOLO LPN zone detection, and OCR on cropped LPN zones, but `subsystem/detection.py`, `subsystem/enhancement.py`, and `subsystem/recognition.py` are still not integrated into the app.
