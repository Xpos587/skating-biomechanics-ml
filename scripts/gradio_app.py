#!/usr/bin/env python3
"""Gradio web UI for figure skating biomechanics analysis.

Provides a browser-based interface for:
- Video upload and preview
- Interactive person selection (click on image or radio buttons)
- Real-time analysis with configurable options
- Downloadable results (video, poses, CSV)
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

import cv2
import numpy as np

from src.gradio_helpers import (
    choice_to_person_click,
    match_click_to_person,
    persons_to_choices,
    process_video_pipeline,
    render_person_preview,
)
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
from src.types import PersonClick
from src.utils.video import get_video_meta


def _create_extractor(tracking: str) -> RTMPoseExtractor:
    """Create RTMPoseExtractor with GPU→CPU fallback."""
    try:
        return RTMPoseExtractor(
            mode="balanced",
            tracking_backend="rtmlib",
            tracking_mode=tracking,
            conf_threshold=0.3,
            output_format="normalized",
            device="cuda",
        )
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU")
        return RTMPoseExtractor(
            mode="balanced",
            tracking_backend="rtmlib",
            tracking_mode=tracking,
            conf_threshold=0.3,
            output_format="normalized",
            device="cpu",
        )


def _detect_persons(
    video_path: str,
    tracking: str,
) -> tuple[gr.Image, list[str], list[dict], str]:
    """Detect all persons in the video and show annotated preview.

    Args:
        video_path: Path to uploaded video file.
        tracking: Tracking mode ("rtmlib", "sports2d", "deepsort").

    Returns:
        (annotated_image, radio_choices, persons_state, status)
    """
    if not video_path:
        return None, gr.update(choices=[], value=None), [], "⚠️ Загрузите видео."

    try:
        extractor = _create_extractor(tracking)

        persons, preview_path = extractor.preview_persons(Path(video_path), num_frames=30)

        if not persons:
            return None, gr.update(choices=[], value=None), [], "⚠️ Люди не найдены. Попробуйте другое видео."

        # Load the preview frame (first frame with detections)
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, gr.update(choices=[], value=None), [], "⚠️ Не удалось прочитать кадр видео."

        # Render annotated preview with numbered bboxes
        annotated = render_person_preview(frame, persons, selected_idx=None)

        # Convert BGR to RGB for Gradio
        import numpy as np
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        choices = persons_to_choices(persons)
        status = f"✅ Обнаружено {len(persons)} чел. Нажмите на человека на изображении или выберите из списка."

        return annotated, gr.update(choices=choices, value=choices[0] if len(choices) == 1 else None), persons, status

    except Exception as e:
        return None, gr.update(choices=[], value=None), [], f"❌ Ошибка: {e}"


def _on_image_select(
    evt: gr.SelectData,
    persons_state: list[dict],
    video_path: str,
) -> tuple[str, gr.Image, PersonClick]:
    """Handle click on preview image to select a person.

    Args:
        evt: Gradio SelectData event with .index (pixel coords).
        persons_state: List of detected person dicts.
        video_path: Path to video for frame dimensions.

    Returns:
        (status_text, annotated_image, person_click)
    """
    if not persons_state:
        return "⚠️ Сначала обнаружьте людей в видео.", None, None

    # Get video dimensions for coordinate normalization
    meta = get_video_meta(Path(video_path))
    w, h = meta.width, meta.height

    # Convert pixel click to normalized coordinates
    x_norm = evt.index[0] / w
    y_norm = evt.index[1] / h

    # Find closest person
    matched = match_click_to_person(persons_state, x_norm, y_norm)

    if matched is None:
        return "⚠️ Нажатие мимо. Попробуйте попасть на человека.", None, None

    # Find the index of the matched person
    idx = persons_state.index(matched)

    # Re-render preview with green highlight
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "⚠️ Не удалось прочитать кадр видео.", None, None

    annotated = render_person_preview(frame, persons_state, selected_idx=idx)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Create PersonClick from mid_hip
    mid_hip = matched["mid_hip"]
    person_click = PersonClick(
        x=int(mid_hip[0] * w),
        y=int(mid_hip[1] * h),
    )

    status = f"✅ Выбран #{idx + 1} (трек {matched['track_id']}, {matched['hits']} кадров)"

    return status, annotated, person_click


def _on_person_select(
    choice: str,
    persons_state: list[dict],
    video_path: str,
) -> tuple[gr.Image, PersonClick]:
    """Handle selection from Radio dropdown.

    Args:
        choice: Selected choice string (e.g., "Person #1 (10 hits, track 0)").
        persons_state: List of detected person dicts.
        video_path: Path to video for frame dimensions.

    Returns:
        (annotated_image, person_click)
    """
    if not choice or not persons_state:
        return None, None

    meta = get_video_meta(Path(video_path))
    person_click = choice_to_person_click(choice, persons_state, meta.width, meta.height)

    # Find the index
    idx = int(choice.split("#")[1].split(" ")[0]) - 1

    # Re-render preview with green highlight
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    annotated = render_person_preview(frame, persons_state, selected_idx=idx)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated, person_click


def _run_pipeline(
    video_path: str,
    person_click_state: PersonClick | None,
    persons_state: list[dict],
    person_choice: str,
    frame_skip: int,
    layer: int,
    tracking: str,
    use_3d: bool,
    render_scale: float,
    export: bool,
    progress=gr.Progress(),
) -> tuple[str, str, str, str]:
    """Run the full analysis pipeline.

    Args:
        video_path: Path to input video.
        person_click_state: Selected person click (from image click).
        persons_state: List of detected persons.
        person_choice: Selected person from Radio (fallback).
        layer: HUD layer (0-3).
        tracking: Tracking mode.
        use_3d: Enable 3D-corrected overlay.
        render_scale: Output render scale (0.5, 0.33, 1.0).
        export: Export poses/CSV.
        progress: Gradio progress callback.

    Returns:
        (output_video_path, poses_path, csv_path, status_text)
    """
    if not video_path:
        return None, None, None, "⚠️ Загрузите видео."

    # Resolve PersonClick (prefer image click, fallback to radio)
    person_click = person_click_state
    if person_click is None and person_choice and persons_state:
        meta = get_video_meta(Path(video_path))
        person_click = choice_to_person_click(person_choice, persons_state, meta.width, meta.height)

    if person_click is None:
        return None, None, None, "⚠️ Выберите человека (нажмите на изображение или выберите из списка)."

    # Generate output path
    input_path = Path(video_path)
    output_dir = input_path.parent / "gradio_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_analyzed.mp4"

    try:
        result = process_video_pipeline(
            video_path=video_path,
            person_click=person_click,
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            use_3d=use_3d,
            render_scale=render_scale,
            blade_3d=False,  # Disabled for now
            export=export,
            output_path=str(output_path),
            progress_cb=lambda p, msg: progress(p, desc=msg),
        )

        stats = result["stats"]
        status = (
            f"✅ Анализ завершён!\n"
            f"   Разрешение: {stats['resolution']}\n"
            f"   Кадров: {stats['valid_frames']}/{stats['total_frames']} валидных\n"
            f"   FPS: {stats['fps']:.1f}\n"
            f"   Результат: {result['video_path']}"
        )

        return (
            result["video_path"],
            result["poses_path"],
            result["csv_path"],
            status,
        )

    except Exception as e:
        return None, None, None, f"❌ Ошибка обработки: {e}"


def build_app() -> gr.Blocks:
    """Build and return the Gradio app interface."""
    with gr.Blocks(title="AI Тренер по фигурному катанию") as app:
        gr.Markdown("# AI Тренер по фигурному катанию")
        gr.Markdown("Загрузите видео, выберите фигуриста и получите биомеханический анализ.")

        # State
        persons_state = gr.State()
        person_click_state = gr.State(None)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Загрузите видео",
                    sources=["upload"],
                )

                tracking_dropdown = gr.Dropdown(
                    label="Режим трекинга",
                    choices=["auto", "sports2d", "deepsort"],
                    value="auto",
                    info="Auto — DeepSORT при наличии, иначе Sports2D",
                )

                detect_btn = gr.Button("Обнаружить людей", variant="primary", size="lg")

                preview_image = gr.Image(
                    label="Превью (нажмите на фигуриста)",
                    interactive=False,
                    type="numpy",
                    height=400,
                )

                person_radio = gr.Radio(
                    label="Или выберите из списка",
                    choices=[],
                    interactive=True,
                )

                selection_status = gr.Textbox(
                    label="Статус выбора",
                    value="Загрузите видео и нажмите «Обнаружить людей»",
                    interactive=False,
                )

                with gr.Accordion("Расширенные настройки", open=False):
                    frame_skip_slider = gr.Slider(
                        label="Пропуск кадров",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4,
                        info="1=все кадры (медленно), 4=каждый 4-й (рекомендуется), 8=максимальная скорость",
                    )

                    layer_slider = gr.Slider(
                        label="Уровень HUD",
                        minimum=0,
                        maximum=3,
                        step=1,
                        value=3,
                        info="0=скелет, 1=скорость+следы+углы, 2=+ось, 3=полный HUD",
                    )

                    render_scale_slider = gr.Slider(
                        label="Масштаб рендера",
                        minimum=0.33,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        info="Меньше = быстрее (0.5 рекомендуется для 1080p+)",
                    )

                    use_3d_checkbox = gr.Checkbox(
                        label="3D-коррекция позы",
                        value=False,
                        info="CorrectiveLens для обработки окклюзий",
                    )

                    export_checkbox = gr.Checkbox(
                        label="Экспорт поз и CSV",
                        value=True,
                        info="Скачать данные поз и биомеханики",
                    )

                process_btn = gr.Button("Обработать видео", variant="primary", size="lg")

            # Right column: Outputs
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Результат анализа",
                    autoplay=True,
                )

                poses_download = gr.File(
                    label="Скачать позы (.npy)",
                    visible=False,
                )

                csv_download = gr.File(
                    label="Скачать биомеханику (.csv)",
                    visible=False,
                )

                output_status = gr.Textbox(
                    label="Статус",
                    value="Ожидание видео...",
                    lines=4,
                    interactive=False,
                )

        # Event wiring
        detect_btn.click(
            fn=_detect_persons,
            inputs=[video_input, tracking_dropdown],
            outputs=[preview_image, person_radio, persons_state, selection_status],
        )

        preview_image.select(
            fn=_on_image_select,
            inputs=[persons_state, video_input],
            outputs=[selection_status, preview_image, person_click_state],
        )

        person_radio.change(
            fn=_on_person_select,
            inputs=[person_radio, persons_state, video_input],
            outputs=[preview_image, person_click_state],
        )

        process_btn.click(
            fn=_run_pipeline,
            inputs=[
                video_input,
                person_click_state,
                persons_state,
                person_radio,
                frame_skip_slider,
                layer_slider,
                tracking_dropdown,
                use_3d_checkbox,
                render_scale_slider,
                export_checkbox,
            ],
            outputs=[output_video, poses_download, csv_download, output_status],
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    )


if __name__ == "__main__":
    main()
