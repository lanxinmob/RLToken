#!/usr/bin/env python3
"""Interactive critical-phase annotation tool for a LeRobot teleop dataset.

Reads image datasets directly from parquet, or decodes videos with PyAV
(including AV1/libsvtav1). Two annotation modes, auto-selected:

1. GUI mode (default when a display is present; tkinter, no OpenCV GUI needed):
   displays the episode's cameras and marks intervals with the keyboard:
       s        mark interval START at the current frame
       e        mark interval END at the current frame (appends [start, end])
       u        undo the last interval
       space    pause / resume
       left/right   step one frame (auto-pause)
       Shift+left/right   jump ±30 frames
       n        skip to the next episode
       q        save annotations and quit
2. Contact-sheet mode (automatic fallback for headless/remote sessions):
   renders each episode into a montage PNG (thumbnail per frame with its frame
   number) into --sheet_dir, then collects intervals interactively on the
   terminal:
       100 300   add interval [100, 300]
       undo      remove the last interval
       skip      skip this episode (nothing saved)
       (blank)   finish this episode, continue with the next
       quit      save annotations and exit

Annotations are saved as JSON with inclusive, episode-local frame indices
(LeRobot frame_index semantics, 0-based per episode). The file is incremental:
already-annotated episodes are skipped, new intervals are appended.

Usage:
    python scripts/annotate_dataset.py \
        --dataset_path /path/to/my_dataset \
        --annotations_out annotations.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations_out", type=str, required=True,
                        help="JSON output path (incremental: existing intervals are kept).")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera feature to show (default: all cameras in GUI, first in contact sheets).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the dataset frame rate from meta/info.json.")
    parser.add_argument("--start_paused", action="store_true",
                        help="Pause at the first frame of each episode until space is pressed.")
    parser.add_argument("--force_sheet", action="store_true",
                        help="Force contact-sheet mode even if a display is available.")
    parser.add_argument("--sheet_dir", type=str, default="annotation_sheets",
                        help="Directory for contact-sheet PNGs (headless mode).")
    parser.add_argument("--sheet_cols", type=int, default=12,
                        help="Thumbnails per row in contact sheets.")
    parser.add_argument("--thumb_width", type=int, default=160,
                        help="Thumbnail width in contact sheets.")
    return parser.parse_args()


def load_existing(path: Path) -> dict[int, list[tuple[int, int]]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    annotations: dict[int, list[tuple[int, int]]] = {}
    for item in data.get("annotations", []):
        annotations.setdefault(int(item["episode"]), []).append(
            (int(item["start"]), int(item["end"]))
        )
    return annotations


def save_annotations(path: Path, annotations: dict[int, list[tuple[int, int]]]) -> None:
    payload = {
        "annotations": [
            {"episode": episode, "start": start, "end": end}
            for episode in sorted(annotations)
            for start, end in annotations[episode]
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.flush()
    tmp.replace(path)
    logger.info("Saved %d intervals across %d episodes to %s",
                len(payload["annotations"]), len(annotations), path)


def episode_frame_sources(
    dataset_path: Path,
    fps: float,
    cameras: tuple[str, ...],
) -> list[tuple[int, dict[str, tuple[Path, int]], int]]:
    """Return episode camera sources and lengths.

    Each source is (path, offset): a video and its first frame, or a parquet
    file and the episode index used to select its embedded image rows.

    Newer LeRobot metadata (0.5.x) stores per-episode video references as
    ``videos/<feature>/chunk_index`` + ``file_index``; several episodes share
    one video file, so the episode's position inside it is
    ``from_timestamp * fps`` (a continuous recording). The timestamps are
    RELATIVE TO EACH FILE (every file restarts at 0), and cameras can roll to
    a new file at different times (e.g. top every 688 s but wrist every 344 s),
    so each camera gets its OWN start_frame. Missing cameras (no parquet
    columns / missing file) are omitted from the dict.
    """
    import pyarrow.parquet as pq

    episodes_dir = dataset_path / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet under {episodes_dir}")
    episodes = pq.read_table(parquet_files).to_pylist()
    info = json.loads((dataset_path / "meta" / "info.json").read_text())

    videos_dir = dataset_path / "videos"
    result: list[tuple[int, dict[str, tuple[Path, int]], int]] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        paths: dict[str, tuple[Path, int]] = {}
        for cam in cameras:
            if info["features"][cam]["dtype"] == "image":
                data_path = dataset_path / info["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
                if not data_path.is_file():
                    raise FileNotFoundError(data_path)
                paths[cam] = (data_path, episode_index)
                continue
            try:
                chunk_index = int(row[f"videos/{cam}/chunk_index"])
                file_index = int(row[f"videos/{cam}/file_index"])
                from_timestamp = float(row[f"videos/{cam}/from_timestamp"])
            except (KeyError, TypeError):
                continue
            video_path = (
                videos_dir / cam / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            )
            if video_path.exists():
                start_frame = int(round(from_timestamp * fps))
                paths[cam] = (video_path, start_frame)
        if not paths:
            raise FileNotFoundError(
                f"Episode {episode_index}: no image or video sources found for cameras {cameras}"
            )
        result.append((episode_index, paths, length))
    return sorted(result, key=lambda item: item[0])



def _seek_to_frame(container, stream, target_frame: int, fps: float) -> None:
    """Seek to the key frame at or before ``target_frame`` (video-global index).

    PyAV's ``Container.seek`` interprets the offset in ``stream.time_base``
    units when a stream is given (NOT microseconds), so convert frame -> ticks.
    """
    offset = int(max(0, target_frame) * (1.0 / fps) / float(stream.time_base))
    container.seek(offset, stream=stream)


class EpisodeFrameReader:
    """Random-access frame reader for one episode's video segment (PyAV).

    Sequential playback decodes forward without reopening; jumps (backward or
    far forward) reopen the container and seek, so pausing and ±30-frame
    stepping stay fast.
    """

    def __init__(self, video_path: Path, start_frame: int, length: int, fps: float):
        self._video_path = str(video_path)
        self._start_frame = start_frame
        self._length = length
        self._fps = fps
        self._container = None
        self._stream = None
        self._decoder = None
        self._current_frame = -1  # last frame index handed out (video-global)
        self._last_image = None   # cached ndarray of _current_frame (for repeated reads)

    def _open(self, episode_frame: int) -> None:
        import av

        if self._container is not None:
            self._container.close()
        self._container = av.open(self._video_path)
        self._stream = self._container.streams.video[0]
        _seek_to_frame(
            self._container, self._stream, max(0, self._start_frame + episode_frame - 10), self._fps,
        )
        self._decoder = self._container.decode(self._stream)
        self._current_frame = self._start_frame + episode_frame - 11
        self._last_image = None

    def read(self, episode_frame: int):
        """Return bgr24 ndarray for the episode-local frame, or None past the end."""
        target = self._start_frame + episode_frame
        if target < 0 or target >= self._start_frame + self._length:
            return None
        if (
            self._container is None
            or target < self._current_frame
            or target - self._current_frame > 90
        ):
            self._open(episode_frame)
        frame = None
        while self._current_frame < target:
            try:
                frame = next(self._decoder)
            except StopIteration:
                return None
            if frame.pts is not None:
                frame_no = int(round(float(frame.pts * self._stream.time_base) * self._fps))
            else:
                frame_no = self._current_frame + 1
            self._current_frame = frame_no
        if self._current_frame == target:
            if frame is None:
                # Repeated read of the current frame (GUI tick + keypress both
                # refresh the same frame): the decoder already handed it out on
                # the first call, so serve the cached image.
                return self._last_image
            self._last_image = frame.to_ndarray(format="bgr24")
            return self._last_image
        return None

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


class EpisodeImageReader:
    """Read one episode's image column without loading other episodes or cameras."""

    def __init__(self, path: Path, episode: int, length: int, camera: str):
        import pyarrow.parquet as pq

        table = pq.read_table(
            path, columns=["frame_index", camera],
            filters=[("episode_index", "=", episode)],
        ).sort_by("frame_index")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"Episode {episode}: image frame indices do not match length {length}")
        self._images = table[camera].to_pylist()
        self._root = path.parents[2]
        self._camera = camera
        self._episode = episode
        self._current_frame = -1
        self._last_image = None

    def read(self, episode_frame: int):
        if not 0 <= episode_frame < len(self._images):
            return None
        if episode_frame == self._current_frame:
            return self._last_image
        entry = self._images[episode_frame]
        if entry.get("bytes") is not None:
            image = cv2.imdecode(np.frombuffer(entry["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            path = Path(entry["path"])
            if not path.is_absolute():
                candidate = self._root / path
                path = candidate if candidate.is_file() else (
                    self._root / "images" / self._camera / f"episode-{self._episode:06d}" / path
                )
            image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Episode {self._episode}: cannot decode {self._camera} frame {episode_frame}")
        self._current_frame = episode_frame
        self._last_image = image
        return image

    def close(self) -> None:
        self._images.clear()
        self._last_image = None


def open_frame_reader(path: Path, offset: int, length: int, fps: float, camera: str):
    if path.suffix == ".parquet":
        return EpisodeImageReader(path, offset, length, camera)
    return EpisodeFrameReader(path, offset, length, fps)


def gui_available() -> bool:
    """Tkinter-based GUI works whenever a display is present (no OpenCV GUI needed)."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


class _TkAnnotator:
    """tkinter playback window showing every camera side by side.

    Keys: space pause, s/e intervals, u undo, ←/→ step frames,
    Shift+←/→ jump ±30, n next episode, q quit.
    """

    DISPLAY_W = 480   # displayed width per camera (source 640x480)
    DISPLAY_H = 360

    def __init__(self, readers: dict[str, EpisodeFrameReader | EpisodeImageReader],
                 episode: int, length: int, *, start_paused: bool = False):
        import tkinter as tk

        from PIL import Image, ImageTk

        self.readers = readers
        self.cameras = list(readers.keys())
        self.episode = episode
        self.length = length
        self.frame = 0
        self.paused = start_paused
        self.start_mark: int | None = None
        self.intervals: list[tuple[int, int]] = []
        self.quit_all = False
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._Image = Image
        self._ImageTk = ImageTk

        n_cameras = max(1, len(self.cameras))
        width = n_cameras * self.DISPLAY_W
        height = self.DISPLAY_H + 24
        self.root = tk.Tk()
        self.root.title(
            f"annotate episode {episode} | cameras: {', '.join(self.cameras)} "
            f"(s=start e=end u=undo space=pause ←/→=frame Shift+←/→=±30 n=next q=quit)"
        )
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black")
        self.canvas.pack()
        self.status = tk.Label(self.root, text="", font=("TkDefaultFont", 12), anchor="w")
        self.status.pack(fill="x")
        self.root.bind("<KeyPress>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_frame()
        self.root.after(30, self._tick)

    def _show_frame(self) -> None:
        self.canvas.delete("all")
        for index, camera in enumerate(self.cameras):
            x0 = index * self.DISPLAY_W
            reader = self.readers[camera]
            bgr = reader.read(self.frame)
            if bgr is None:
                self.canvas.create_text(
                    x0 + self.DISPLAY_W // 2, self.DISPLAY_H // 2,
                    text=f"{camera}: no frame", fill="#888888",
                )
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = self._Image.fromarray(rgb).resize(
                    (self.DISPLAY_W, self.DISPLAY_H), self._Image.BILINEAR,
                )
                self._photos[camera] = self._ImageTk.PhotoImage(image)
                self.canvas.create_image(x0, 0, anchor="nw", image=self._photos[camera])
                for start, end in self.intervals:
                    if start <= self.frame <= end:
                        color = "#ff6666" if self.start_mark is None else "#ffcc66"
                        self.canvas.create_rectangle(
                            x0 + 2, 2, x0 + self.DISPLAY_W - 2, self.DISPLAY_H - 2,
                            outline=color, width=6,
                        )
                        break
                if self.start_mark is not None and self.frame >= self.start_mark:
                    self.canvas.create_rectangle(
                        x0 + 2, 2, x0 + self.DISPLAY_W - 2, self.DISPLAY_H - 2,
                        outline="#66ff66", width=3,
                    )
            self.canvas.create_text(
                x0 + 8, self.DISPLAY_H + 4, anchor="nw",
                text=camera.replace("observation.images.", ""),
                fill="#ffffff", font=("TkDefaultFont", 10),
            )
        status = "PAUSED" if self.paused else "PLAYING"
        self.status.config(
            text=f"ep {self.episode}  frame {self.frame}/{self.length}  [{status}]  "
                 f"intervals: {self.intervals}  (n = next episode, q = quit, "
                 f"auto-advances at the last frame)"
        )

    def _tick(self) -> None:
        if not self.paused and self.frame + 1 < self.length:
            self.frame += 1
            self._show_frame()
            self.root.after(30, self._tick)
        elif not self.paused:
            # Playback reached the last frame: auto-save and move to the next
            # episode (the main loop persists any marked intervals).
            print(f"  [ep {self.episode}] end of episode reached; moving to next episode")
            self.root.destroy()
        else:
            self.root.after(30, self._tick)

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.quit_all = True
            self.root.destroy()
        elif key == "space":
            self.paused = not self.paused
            self._show_frame()
        elif key == "n":
            self.root.destroy()
        elif key == "s":
            self.start_mark = self.frame
            print(f"  [ep {self.episode}] interval START at frame {self.frame}")
            self._show_frame()
        elif key == "e":
            if self.start_mark is None:
                print(f"  [ep {self.episode}] press 's' before 'e'")
            else:
                end_frame = max(self.start_mark, self.frame)
                self.intervals.append((self.start_mark, end_frame))
                print(f"  [ep {self.episode}] interval END at frame {self.frame} -> "
                      f"[{self.start_mark}, {end_frame}]")
                self.start_mark = None
            self._show_frame()
        elif key == "u":
            if self.intervals:
                removed = self.intervals.pop()
                print(f"  [ep {self.episode}] removed interval {removed}")
            else:
                print(f"  [ep {self.episode}] nothing to undo")
            self._show_frame()
        elif key in ("left", "right"):
            # Arrow keys step one frame (auto-pause); Shift+arrow jumps ±30.
            step = 30 if (event.state & 0x0001) else 1
            delta = step if key == "right" else -step
            self.paused = True
            self.frame = max(0, min(self.frame + delta, self.length - 1))
            self._show_frame()

    def _on_close(self) -> None:
        self.quit_all = True
        self.root.destroy()

    def run(self) -> tuple[list[tuple[int, int]], bool]:
        self.root.mainloop()
        return self.intervals, self.quit_all


def annotate_episode_gui(
    episode: int,
    video_infos: dict[str, tuple[Path, int]],
    length: int,
    fps: float,
    *,
    start_paused: bool = False,
) -> tuple[list[tuple[int, int]], bool]:
    """Play the episode in a tkinter window (all cameras side by side).

    Each camera has its own start_frame inside its own video file, because
    cameras can roll to new files at different times.
    """
    readers = {
        camera: open_frame_reader(path, start_frame, length, fps, camera)
        for camera, (path, start_frame) in video_infos.items()
    }
    print(f"\n=== Episode {episode} | {length} frames | cameras: {', '.join(video_infos)} ===")
    print("Keys: s=interval START  e=interval END  u=undo  space=pause  "
          "←/→=1 frame  Shift+←/→=±30  n=next  q=quit")
    try:
        annotator = _TkAnnotator(readers, episode, length, start_paused=start_paused)
        return annotator.run()
    finally:
        for reader in readers.values():
            reader.close()


def render_contact_sheet(
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    cols: int,
    thumb_width: int,
    camera: str,
) -> np.ndarray:
    """Render one episode as a montage of thumbnails labelled with frame numbers."""
    thumb_height = int(thumb_width * 3 / 4)
    label_height = 22
    cell_w, cell_h = thumb_width, thumb_height + label_height
    rows = (length + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    reader = open_frame_reader(video_path, start_frame, length, fps, camera)
    try:
        for episode_frame in range(length):
            image = reader.read(episode_frame)
            if image is None:
                raise ValueError(f"Cannot read {camera} frame {episode_frame} from {video_path}")
            col = episode_frame % cols
            row = episode_frame // cols
            thumb = cv2.resize(image, (thumb_width, thumb_height))
            y0 = row * cell_h + label_height
            x0 = col * cell_w
            sheet[y0:y0 + thumb_height, x0:x0 + thumb_width] = thumb
            cv2.putText(sheet, f"{episode_frame}", (x0 + 2, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    finally:
        reader.close()
    return sheet


def annotate_episode_sheet(
    episode: int,
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    sheet_dir: Path,
    cols: int,
    thumb_width: int,
    camera: str,
) -> tuple[list[tuple[int, int]], bool]:
    """Render a contact sheet, then collect intervals from terminal input.

    Returns (intervals, quit_all).
    """
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sheet_dir / f"episode_{episode:04d}.png"
    sheet = render_contact_sheet(video_path, start_frame, length, fps,
                                 cols=cols, thumb_width=thumb_width, camera=camera)
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\n=== Episode {episode} | {length} frames | sheet: {sheet_path} ===")
    print("Open the sheet image and enter critical-phase intervals.")
    print("  '<start> <end>'  add interval | 'undo' | 'skip' | 'quit' | blank = next episode")
    intervals: list[tuple[int, int]] = []
    while True:
        try:
            line = input(f"  ep {episode} interval: ").strip().lower()
        except EOFError:
            return intervals, True
        if not line:
            return intervals, False
        if line in ("q", "quit"):
            return intervals, True
        if line == "skip":
            return intervals, False
        if line == "undo":
            if intervals:
                removed = intervals.pop()
                print(f"  removed {removed}")
            else:
                print("  nothing to undo")
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                print("  invalid numbers")
                continue
            if not (0 <= start <= end < length):
                print(f"  out of range [0, {length - 1}]")
                continue
            intervals.append((start, end))
            print(f"  added [{start}, {end}] (total {len(intervals)})")
        else:
            print("  expected: '<start> <end>'")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    out_path = Path(args.annotations_out)
    annotations = load_existing(out_path)
    if annotations:
        logger.info("Loaded %d previously annotated episodes; they will be skipped.", len(annotations))

    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    fps = args.fps if args.fps is not None else float(info["fps"])
    if fps <= 0:
        raise ValueError("Frame rate must be positive")
    cameras = tuple(key for key, feature in info["features"].items()
                    if feature["dtype"] in ("image", "video"))
    if args.camera is not None:
        if args.camera not in cameras:
            raise ValueError(f"Unknown camera {args.camera!r}; available cameras: {cameras}")
        cameras = (args.camera,)
    if not cameras:
        raise ValueError("Dataset has no image or video features")
    episodes = episode_frame_sources(dataset_path, fps, cameras)
    logger.info("Found %d episodes in %s", len(episodes), dataset_path)

    use_gui = not args.force_sheet and gui_available()
    logger.info("Annotation mode: %s", "GUI playback (tkinter)" if use_gui else "contact-sheet (headless)")
    if not out_path.exists():
        save_annotations(out_path, annotations)

    try:
        for episode, video_infos, length in episodes:
            if episode in annotations:
                logger.info("Skipping episode %d (already annotated: %s)", episode, annotations[episode])
                continue
            if use_gui:
                intervals, should_quit = annotate_episode_gui(
                    episode, video_infos, length, fps, start_paused=args.start_paused,
                )
            else:
                camera = args.camera or cameras[0]
                sheet_info = video_infos.get(camera)
                if sheet_info is None:
                    logger.warning(
                        "Episode %d: camera %s unavailable (have %s); skipping.",
                        episode, camera, sorted(video_infos),
                    )
                    continue
                sheet_video, sheet_start = sheet_info
                intervals, should_quit = annotate_episode_sheet(
                    episode, sheet_video, sheet_start, length, fps,
                    sheet_dir=Path(args.sheet_dir),
                    cols=args.sheet_cols,
                    thumb_width=args.thumb_width,
                    camera=camera,
                )
            if intervals:
                annotations[episode] = intervals
                save_annotations(out_path, annotations)
            else:
                logger.info("Episode %d: no intervals marked, not saved.", episode)
            if should_quit:
                print("Quit requested.")
                break
    finally:
        if annotations:
            save_annotations(out_path, annotations)
    print(f"\nDone. Annotations: {out_path}")


if __name__ == "__main__":
    main()
