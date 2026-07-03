"""이미지 검사기 GUI (그래픽 화면).

명령줄 없이 창에서 버튼으로 이미지를 검사할 수 있는 프로그램입니다.
파이썬에 기본 포함된 Tkinter 를 사용하므로 별도 설치가 (거의) 필요 없습니다.

실행::

    python gui.py

사용 순서
  1) [1. 모델 학습]  또는  [모델 불러오기] 로 모델을 준비
  2) [2. 이미지 열기] 로 검사할 이미지를 선택
  3) 화면에 이미지와 판정 결과(양품/불량 + 확신도)가 표시됨
"""

from __future__ import annotations

import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from inspector import (
    ImageInspector,
    export_results,
    inspect_folder,
    save_csv,
    summarize,
)

# 화면에 이미지를 표시할 때의 최대 크기(px)
PREVIEW_SIZE = 360
DEFAULT_MODEL = "model.joblib"

# 라벨 이름별 색상(초록=양호, 빨강=불량). 없는 이름은 파란색.
GOOD_WORDS = {"ok", "good", "pass", "양품", "정상", "합격"}
BAD_WORDS = {"ng", "bad", "fail", "defect", "불량", "결함", "불합격"}


def color_for(label: str) -> str:
    low = label.lower()
    if low in GOOD_WORDS:
        return "#1a9850"  # 초록
    if low in BAD_WORDS:
        return "#d73027"  # 빨강
    return "#2c6fbb"      # 파랑


class InspectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.inspector: ImageInspector | None = None
        self.current_photo = None  # ImageTk 참조 유지용 (GC 방지)
        self.batch_results: list[dict] = []  # 마지막 일괄 검사 결과 (CSV 저장용)

        root.title("이미지 검사기")
        root.geometry("640x760")
        root.minsize(560, 680)

        self._build_ui()

        # 시작 시 model.joblib 이 있으면 자동으로 불러오기
        if Path(DEFAULT_MODEL).exists():
            self._load_model_from(DEFAULT_MODEL, silent=True)

    # ------------------------------------------------------------------ #
    # 화면 구성
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # 상단 버튼들
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Button(top, text="1. 모델 학습", command=self.on_train).pack(side="left", padx=4)
        ttk.Button(top, text="모델 불러오기", command=self.on_load_model).pack(side="left", padx=4)
        ttk.Button(top, text="2. 이미지 열기", command=self.on_open_image).pack(side="left", padx=4)
        ttk.Button(top, text="3. 폴더 일괄 검사", command=self.on_scan_folder).pack(side="left", padx=4)
        ttk.Button(top, text="4. 불량 모으기+히트맵", command=self.on_export).pack(side="left", padx=4)

        # 모델 상태 표시
        self.model_var = tk.StringVar(value="모델: 없음 (먼저 학습하거나 불러오세요)")
        ttk.Label(self.root, textvariable=self.model_var, foreground="#555").pack(anchor="w", padx=12)

        # 이미지 미리보기 영역
        self.image_label = tk.Label(
            self.root,
            text="\n\n검사할 이미지를 열어주세요\n\n",
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE // 20,
            relief="groove",
            bg="#f4f4f4",
            fg="#888",
        )
        self.image_label.pack(padx=12, pady=10)

        # 판정 결과 (큰 글씨)
        self.result_var = tk.StringVar(value="—")
        self.result_label = tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Arial", 28, "bold"),
            fg="#333",
        )
        self.result_label.pack(pady=(4, 2))

        # 확신도 막대들이 들어갈 영역
        self.conf_frame = ttk.Frame(self.root)
        self.conf_frame.pack(fill="x", padx=24, pady=6)

        # 폴더 일괄 검사 결과 영역
        batch = ttk.LabelFrame(self.root, text="폴더 일괄 검사 결과")
        batch.pack(fill="both", expand=True, padx=12, pady=6)

        bar = ttk.Frame(batch)
        bar.pack(fill="x", padx=6, pady=4)
        self.batch_summary_var = tk.StringVar(value="아직 검사한 폴더가 없습니다.")
        ttk.Label(bar, textvariable=self.batch_summary_var).pack(side="left")
        self.save_csv_btn = ttk.Button(bar, text="결과 CSV 저장", command=self.on_save_csv, state="disabled")
        self.save_csv_btn.pack(side="right")

        table_wrap = ttk.Frame(batch)
        table_wrap.pack(fill="both", expand=True, padx=6, pady=4)
        self.tree = ttk.Treeview(
            table_wrap, columns=("file", "label", "conf"), show="headings", height=8
        )
        self.tree.heading("file", text="파일명")
        self.tree.heading("label", text="판정")
        self.tree.heading("conf", text="확신도")
        self.tree.column("file", width=280)
        self.tree.column("label", width=80, anchor="center")
        self.tree.column("conf", width=80, anchor="center")
        scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        # 판정별 행 색상
        self.tree.tag_configure("good", foreground="#1a9850")
        self.tree.tag_configure("bad", foreground="#d73027")
        self.tree.tag_configure("err", foreground="#999999")

        # 하단 상태줄
        self.status_var = tk.StringVar(value="준비됨")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            side="bottom", fill="x"
        )

    # ------------------------------------------------------------------ #
    # 모델 학습
    # ------------------------------------------------------------------ #
    def on_train(self):
        folder = filedialog.askdirectory(title="학습 폴더 선택 (안에 분류별 하위 폴더가 있어야 함)")
        if not folder:
            return

        self.status_var.set("학습 중... 잠시 기다려 주세요")
        self.root.update_idletasks()

        def work():
            try:
                inspector = ImageInspector()
                report = inspector.train(folder)
                inspector.save(DEFAULT_MODEL)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._train_failed(exc))
                return
            self.root.after(0, lambda: self._train_done(inspector, report))

        threading.Thread(target=work, daemon=True).start()

    def _train_done(self, inspector: ImageInspector, report: dict):
        self.inspector = inspector
        names = ", ".join(report["class_names"])
        acc = report["cv_accuracy"]
        acc_txt = f"{acc * 100:.1f}%" if acc is not None else "(데이터 적음)"
        self.model_var.set(f"모델: {names}  |  학습 {report['num_images']}장  |  정확도 {acc_txt}")
        self.status_var.set(f"학습 완료 → {DEFAULT_MODEL} 저장됨")
        messagebox.showinfo(
            "학습 완료",
            f"분류 종류: {names}\n학습 이미지: {report['num_images']}장\n교차검증 정확도: {acc_txt}",
        )

    def _train_failed(self, exc: Exception):
        self.status_var.set("학습 실패")
        messagebox.showerror("학습 실패", str(exc))

    # ------------------------------------------------------------------ #
    # 모델 불러오기
    # ------------------------------------------------------------------ #
    def on_load_model(self):
        path = filedialog.askopenfilename(
            title="모델 파일 선택",
            filetypes=[("모델 파일", "*.joblib"), ("모든 파일", "*.*")],
        )
        if path:
            self._load_model_from(path)

    def _load_model_from(self, path: str, silent: bool = False):
        try:
            self.inspector = ImageInspector.load(path)
        except Exception as exc:  # noqa: BLE001
            if not silent:
                messagebox.showerror("불러오기 실패", str(exc))
            return
        names = ", ".join(self.inspector.class_names)
        self.model_var.set(f"모델: {names}  (불러옴: {Path(path).name})")
        self.status_var.set(f"모델 불러오기 완료: {path}")

    # ------------------------------------------------------------------ #
    # 이미지 열기 & 검사
    # ------------------------------------------------------------------ #
    def on_open_image(self):
        if self.inspector is None:
            messagebox.showwarning(
                "모델 없음",
                "먼저 [1. 모델 학습] 하거나 [모델 불러오기] 로 모델을 준비하세요.",
            )
            return

        path = filedialog.askopenfilename(
            title="검사할 이미지 선택",
            filetypes=[
                ("이미지", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return

        self._show_image(path)
        self._run_prediction(path)

    def _show_image(self, path: str):
        image = Image.open(path).convert("RGB")
        image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_photo, text="", width=0, height=0)

    def _run_prediction(self, path: str):
        try:
            label, confidences = self.inspector.predict(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("검사 실패", str(exc))
            return

        self.result_var.set(f"판정: {label}")
        self.result_label.config(fg=color_for(label))
        self._show_confidences(confidences, best=label)
        self.status_var.set(f"검사 완료: {Path(path).name}")

    # ------------------------------------------------------------------ #
    # 폴더 일괄 검사 & CSV 저장
    # ------------------------------------------------------------------ #
    def on_scan_folder(self):
        if self.inspector is None:
            messagebox.showwarning(
                "모델 없음",
                "먼저 [1. 모델 학습] 하거나 [모델 불러오기] 로 모델을 준비하세요.",
            )
            return

        folder = filedialog.askdirectory(title="일괄 검사할 이미지 폴더 선택")
        if not folder:
            return

        self.status_var.set("일괄 검사 중... 잠시 기다려 주세요")
        self.save_csv_btn.config(state="disabled")
        self.root.update_idletasks()

        def work():
            try:
                results = inspect_folder(self.inspector, folder)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._scan_failed(exc))
                return
            self.root.after(0, lambda: self._scan_done(results))

        threading.Thread(target=work, daemon=True).start()

    def _scan_done(self, results: list[dict]):
        self.batch_results = results

        # 표 채우기
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in results:
            if row["error"]:
                self.tree.insert("", "end", values=(Path(row["path"]).name, "오류", "-"), tags=("err",))
                continue
            conf = row["confidences"].get(row["label"], 0)
            tag = "good" if color_for(row["label"]) == "#1a9850" else (
                "bad" if color_for(row["label"]) == "#d73027" else ""
            )
            self.tree.insert(
                "", "end",
                values=(Path(row["path"]).name, row["label"], f"{conf * 100:.1f}%"),
                tags=(tag,) if tag else (),
            )

        if not results:
            self.batch_summary_var.set("검사할 이미지를 찾지 못했습니다.")
            self.status_var.set("일괄 검사: 이미지 없음")
            return

        stats = summarize(results)
        summary = ", ".join(f"{name} {count}장" for name, count in stats["counts"].items())
        extra = f"  (오류 {stats['errors']}장)" if stats["errors"] else ""
        self.batch_summary_var.set(f"총 {stats['total']}장 → {summary}{extra}")
        self.save_csv_btn.config(state="normal")
        self.status_var.set("일괄 검사 완료 — [결과 CSV 저장] 으로 저장할 수 있습니다")

    def _scan_failed(self, exc: Exception):
        self.status_var.set("일괄 검사 실패")
        messagebox.showerror("일괄 검사 실패", str(exc))

    def on_save_csv(self):
        if not self.batch_results:
            return
        path = filedialog.asksaveasfilename(
            title="결과 CSV 저장",
            defaultextension=".csv",
            initialfile="검사결과.csv",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        try:
            save_csv(self.batch_results, path, self.inspector.class_names)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("저장 실패", str(exc))
            return
        self.status_var.set(f"CSV 저장 완료: {path}")
        messagebox.showinfo("저장 완료", f"결과를 저장했습니다:\n{path}")

    # ------------------------------------------------------------------ #
    # 불량 모으기 + 히트맵 (결과 폴더로 내보내기)
    # ------------------------------------------------------------------ #
    def on_export(self):
        if self.inspector is None:
            messagebox.showwarning(
                "모델 없음",
                "먼저 [1. 모델 학습] 하거나 [모델 불러오기] 로 모델을 준비하세요.",
            )
            return

        folder = filedialog.askdirectory(title="검사할 이미지 폴더 선택")
        if not folder:
            return
        out_dir = filedialog.askdirectory(title="결과를 저장할 폴더 선택")
        if not out_dir:
            return

        self.status_var.set("검사 + 히트맵 생성 중... (이미지가 많으면 시간이 걸립니다)")
        self.root.update_idletasks()

        def report_progress(done, total, message):
            self.root.after(0, lambda: self.status_var.set(f"{message} ({done}/{total})"))

        def work():
            try:
                stats = export_results(
                    self.inspector, folder, out_dir, progress=report_progress
                )
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._export_failed(exc))
                return
            self.root.after(0, lambda: self._export_done(stats))

        threading.Thread(target=work, daemon=True).start()

    def _export_done(self, stats: dict):
        # 결과 표도 함께 채워 준다
        self._scan_done(stats["results"])

        defect = ", ".join(stats["defect_classes"]) or "(자동 판단 불가)"
        msg = (
            f"결과 폴더: {stats['out_dir']}\n\n"
            f"불량 분류: {defect}\n"
            f"불량 이미지: {stats['ng_copied']}장  →  {stats['ng_dir']}\n"
            f"히트맵: {stats['heatmaps']}장  →  {stats['heat_dir']}\n"
            f"CSV: {stats['csv']}"
        )
        self.status_var.set(
            f"완료 — 불량 {stats['ng_copied']}장, 히트맵 {stats['heatmaps']}장 저장됨"
        )
        messagebox.showinfo("완료", msg)

    def _export_failed(self, exc: Exception):
        self.status_var.set("불량 모으기 실패")
        messagebox.showerror("실패", str(exc))

    def _show_confidences(self, confidences: dict, best: str):
        # 기존 막대 지우기
        for child in self.conf_frame.winfo_children():
            child.destroy()

        for name, prob in sorted(confidences.items(), key=lambda kv: kv[1], reverse=True):
            row = ttk.Frame(self.conf_frame)
            row.pack(fill="x", pady=2)

            tag = "◀" if name == best else "  "
            ttk.Label(row, text=f"{name:<8} {prob * 100:5.1f}% {tag}", width=20).pack(side="left")

            bar = ttk.Progressbar(row, maximum=100, value=prob * 100, length=240)
            bar.pack(side="left", fill="x", expand=True)


def main():
    root = tk.Tk()
    InspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
