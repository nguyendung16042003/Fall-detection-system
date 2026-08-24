# Tan Dung (AI)

AI / Computer-Vision module for the **Multi-Camera Fall Detection System on Edge Device with VLM
Verification** capstone project — owned by Nguyễn Tấn Dũng. Covers the three cross-camera
challenges the system solves on a Jetson Nano 4GB: **Re-Identification**, **Identity Association**,
and **Boundary Feature Fusion**.

This repo tracks source code and written documents only (`.py`, `.docx`, `.md`) — datasets, trained
weights, videos, and generated images are excluded via `.gitignore` to keep the shared repo light.
See the report/slide deck below for the full narrative, numbers, and figures.

## Where things live

| Folder | What it is |
|---|---|
| `Chạy xử lý toàn bộ hệ thống/Code/p1_reid/` | Re-Identification — OSNet x0.25 distillation training, MSINet baseline comparison, ONNX export, video-demo scripts |
| `Chạy xử lý toàn bộ hệ thống/Code/p2_homography/` | Identity Association — floor-plane homography calibration + Hungarian matching, fall-rule adapter |
| `Chạy xử lý toàn bộ hệ thống/Code/p3_cross_camera/` | Boundary Feature Fusion — cross-camera attention fusion, calibration, lightweight-classifier benchmark, ONNX export |
| `Chạy xử lý toàn bộ hệ thống/Code/_archive_khong_dung/` | Deprecated/experimental code, **not** part of the final deployed system |
| `Chạy xử lý toàn bộ hệ thống/Report/` | `Report chính.docx`, `Script thuyết trình.docx`, and `gen_diagrams.py` (regenerates the chart/diagram images used in the slides and report) |
| `Chạy xử lý toàn bộ hệ thống/Số liệu thống kê(- Copy)/` | Result/statistics dumps referenced by the report |
| `Chạy xử lý toàn bộ hệ thống/Video demo/`, `File Run Problem 1-3/`, `File Test đề phòng/` | Demo-recording and test-run scripts for each challenge |
| `Coding/` | Earlier/parallel training-and-eval workspace (Test, training, Evaluation, Pipeline, Benchmark) |
| `Datasets/` | Dataset-prep and download scripts (the actual data is gitignored) |
| `Handoff_for_Edge/` | Final deployment-ready pipeline code and calibration handed off to the Edge module — `1_Single_Camera_Pipeline/` and `2_Multi_Camera_Modules/` |
| `Papers/` | Notes on reference papers (LFD-YOLO, PIFR) |
| `docs/` | Full capstone report (docx) |

## Full documentation

- **Report**: `Chạy xử lý toàn bộ hệ thống/Report/Report chính.docx` — authoritative source for every
  number/claim in the slides.
- **Slides**: `Chạy xử lý toàn bộ hệ thống/Report/Fall Detection(Slide bản PowerPoint).pptx`
- **Deployed artifacts**: see `Handoff_for_Edge/` for what actually ships on the Jetson Nano.
