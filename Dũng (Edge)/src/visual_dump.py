#!/usr/bin/env python3
"""Xuất video MP4 CÓ ANNOTATE (khung + ID tracker + nhãn tư thế SGIE) từ 2 cam
RTSP, để XEM BẰNG MẮT pipeline đang hoạt động ra sao — không chạy mù như
pipeline.py (fakesink).

Khung ĐỎ nếu tư thế 'lie', XANH nếu khác. Text: "ID {id} {label} {conf:.2f}".
Dùng CHUNG config PGIE/tracker/SGIE với pipeline.py (qua cùng biến môi trường
PGIE_CONFIG/TRACKER_CONFIG_OVERRIDE/SGIE_CONFIG nếu cần override).

Dùng:
    export RTSP_CAM1=... RTSP_CAM2=...
    python3 src/visual_dump.py --seconds 150 --out ../output/visual/demo.mp4
"""
import argparse
import os
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PGIE_CONFIG = os.environ.get("PGIE_CONFIG", f"{PROJECT_ROOT}/configs/pgie_yolov8n.txt")
SGIE_CONFIG = os.environ.get("SGIE_CONFIG", f"{PROJECT_ROOT}/configs/sgie_yolov8n_cls.txt")
TRACKER_CONFIG = os.environ.get("TRACKER_CONFIG_OVERRIDE", f"{PROJECT_ROOT}/configs/tracker_nvdcf_ds6.yml")
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so"

MUXER_WIDTH = 1280
MUXER_HEIGHT = 720


def read_sgie_label(obj_meta):
    l_class = obj_meta.classifier_meta_list
    while l_class is not None:
        class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
        if class_meta.unique_component_id == 2:
            l_label = class_meta.label_info_list
            if l_label is not None:
                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                return label_info.result_label, label_info.result_prob
        try:
            l_class = l_class.next
        except StopIteration:
            break
    return None, 0.0


def annotate_probe(pad, info, u_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            label, conf = read_sgie_label(obj_meta)
            is_lie = label == "lie"

            obj_meta.rect_params.has_bg_color = 0
            obj_meta.rect_params.border_width = 4
            if is_lie:
                obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
            else:
                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)

            txt = f"ID{obj_meta.object_id} {label or '?'} {conf:.2f}"
            obj_meta.text_params.display_text = txt
            obj_meta.text_params.x_offset = max(0, int(obj_meta.rect_params.left))
            obj_meta.text_params.y_offset = max(0, int(obj_meta.rect_params.top) - 20)
            obj_meta.text_params.font_params.font_name = "Serif"
            obj_meta.text_params.font_params.font_size = 13
            obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            obj_meta.text_params.set_bg_clr = 1
            if is_lie:
                obj_meta.text_params.text_bg_clr.set(0.6, 0.0, 0.0, 1.0)
            else:
                obj_meta.text_params.text_bg_clr.set(0.0, 0.4, 0.0, 1.0)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--out", default=f"{PROJECT_ROOT}/output/visual/demo.mp4")
    ap.add_argument("--bitrate", type=int, default=900, help="kbps, giữ file nhỏ để xem qua trình duyệt")
    args = ap.parse_args()

    cam1 = os.environ.get("RTSP_CAM1")
    cam2 = os.environ.get("RTSP_CAM2")
    if not cam1 or not cam2:
        print("Thiếu RTSP_CAM1/RTSP_CAM2 trong biến môi trường", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    Gst.init(None)

    pipeline_str = (
        f"rtspsrc location={cam1} protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! m.sink_0 "
        f"rtspsrc location={cam2} protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! m.sink_1 "
        f"nvstreammux name=m width={MUXER_WIDTH} height={MUXER_HEIGHT} batch-size=2 "
        f"batched-push-timeout=40000 live-source=1 ! "
        f"nvinfer name=pgie config-file-path={PGIE_CONFIG} ! "
        f"nvtracker name=trk ll-lib-file={TRACKER_LIB} ll-config-file={TRACKER_CONFIG} "
        f"tracker-width=640 tracker-height=384 display-tracking-id=1 ! "
        f"nvinfer name=sgie config-file-path={SGIE_CONFIG} ! "
        f"nvmultistreamtiler name=tiler rows=1 columns=2 width={MUXER_WIDTH} height=640 ! "
        f"nvvideoconvert ! nvdsosd ! nvvideoconvert ! "
        f"video/x-raw,format=I420 ! "
        f"x264enc name=enc speed-preset=ultrafast tune=zerolatency bitrate={args.bitrate} key-int-max=30 ! "
        f"h264parse ! mp4mux fragment-duration=1000 streamable=true ! filesink location={args.out}"
    )
    print(f"[visual_dump] pipeline: {pipeline_str}", flush=True)
    pipeline = Gst.parse_launch(pipeline_str)

    sgie = pipeline.get_by_name("sgie")
    sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, annotate_probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_msg(_bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("[visual_dump] EOS nhận được, đóng file...", flush=True)
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[visual_dump][ERROR] {err} {dbg}", flush=True)
            loop.quit()
        return True

    bus.connect("message", on_msg)
    pipeline.set_state(Gst.State.PLAYING)

    def send_eos():
        # pipeline.send_event(EOS) KHÔNG đủ với rtspsrc (nguồn sống, tự bơm data
        # liên tục) -> phải gửi EOS TRỰC TIẾP vào từng rtspsrc để nó tự phát
        # EOS xuôi dòng, mp4mux mới đóng file đúng cách (nếu không file lớn
        # dần mãi, không bao giờ dừng).
        print("[visual_dump] Hết giờ, gửi EOS vào từng rtspsrc...", flush=True)
        it = pipeline.iterate_sources()
        while True:
            result, elem = it.next()
            if result != Gst.IteratorResult.OK:
                break
            print(f"[visual_dump]   EOS -> {elem.get_name()}", flush=True)
            elem.send_event(Gst.Event.new_eos())

        def force_stop():
            print("[visual_dump] EOS không về sau 20s -> dừng cứng (file có thể lỗi).", flush=True)
            loop.quit()
            return False

        GLib.timeout_add_seconds(20, force_stop)
        return False

    GLib.timeout_add_seconds(args.seconds, send_eos)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print(f"[visual_dump] Xong. File: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
