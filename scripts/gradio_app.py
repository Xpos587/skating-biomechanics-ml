#!/usr/bin/env python3
"""Gradio web UI for figure skating biomechanics analysis.

Provides a browser-based interface for:
- Video upload and preview
- Interactive person selection (click on image or radio buttons)
- Real-time analysis with configurable options
- Downloadable results (video, poses, CSV)
"""

from __future__ import annotations

import gradio as gr
from pathlib import Path

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
        return None, gr.update(choices=[], value=None), [], "⚠️ Please upload a video first."

    try:
        extractor = RTMPoseExtractor(
            mode="balanced",
            tracking_backend="rtmlib",
            tracking_mode=tracking,
            conf_threshold=0.3,
            output_format="normalized",
            device="cuda",
        )

        persons, preview_path = extractor.preview_persons(video_path, num_frames=30)

        if not persons:
            return None, gr.update(choices=[], value=None), [], "⚠️ No persons detected. Try a different video."

        # Load the preview frame (first frame with detections)
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, gr.update(choices=[], value=None), [], "⚠️ Failed to read video frame."

        # Render annotated preview with numbered bboxes
        annotated = render_person_preview(frame, persons, selected_idx=None)

        # Convert BGR to RGB for Gradio
        import numpy as np
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        choices = persons_to_choices(persons)
        status = f"✅ Detected {len(persons)} person(s). Click on the image or select from the list."

        return annotated, gr.update(choices=choices, value=choices[0] if len(choices) == 1 else None), persons, status

    except Exception as e:
        return None, gr.update(choices=[], value=None), [], f"❌ Error: {e}"


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
        return "⚠️ No persons detected yet.", None, None

    # Get video dimensions for coordinate normalization
    meta = get_video_meta(video_path)
    w, h = meta.width, meta.height

    # Convert pixel click to normalized coordinates
    x_norm = evt.index[0] / w
    y_norm = evt.index[1] / h

    # Find closest person
    matched = match_click_to_person(persons_state, x_norm, y_norm)

    if matched is None:
        return "⚠️ Click didn't land on any person. Try again.", None, None

    # Find the index of the matched person
    idx = persons_state.index(matched)

    # Re-render preview with green highlight
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "⚠️ Failed to read video frame.", None, None

    annotated = render_person_preview(frame, persons_state, selected_idx=idx)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Create PersonClick from mid_hip
    mid_hip = matched["mid_hip"]
    person_click = PersonClick(
        x=int(mid_hip[0] * w),
        y=int(mid_hip[1] * h),
    )

    status = f"✅ Selected Person #{idx + 1} (track {matched['track_id']}, {matched['hits']} hits)"

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

    meta = get_video_meta(video_path)
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
        return None, None, None, "⚠️ Please upload a video first."

    # Resolve PersonClick (prefer image click, fallback to radio)
    person_click = person_click_state
    if person_click is None and person_choice and persons_state:
        meta = get_video_meta(video_path)
        person_click = choice_to_person_click(person_choice, persons_state, meta.width, meta.height)

    if person_click is None:
        return None, None, None, "⚠️ Please select a person first (click on image or use dropdown)."

    # Generate output path
    input_path = Path(video_path)
    output_dir = input_path.parent / "gradio_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_analyzed.mp4"

    try:
        result = process_video_pipeline(
            video_path=video_path,
            person_click=person_click,
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
            f"✅ Analysis complete!\n"
            f"   Resolution: {stats['resolution']}\n"
            f"   Frames: {stats['valid_frames']}/{stats['total_frames']} valid\n"
            f"   FPS: {stats['fps']:.1f}\n"
            f"   Output: {result['video_path']}"
        )

        return (
            result["video_path"],
            result["poses_path"],
            result["csv_path"],
            status,
        )

    except Exception as e:
        return None, None, None, f"❌ Pipeline error: {e}"


def build_app() -> gr.Blocks:
    """Build and return the Gradio app interface."""
    with gr.Blocks(
        title="AI Figure Skating Coach",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    ) as app:
        gr.Markdown("# 🏒 AI Figure Skating Coach")
        gr.Markdown("Upload a video, select a person, and analyze biomechanics.")

        # State
        persons_state = gr.State()
        person_click_state = gr.State(None)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                )

                tracking_dropdown = gr.Dropdown(
                    label="Tracking Mode",
                    choices=["auto", "sports2d", "deepsort"],
                    value="auto",
                    info="Auto uses DeepSORT if available, otherwise Sports2D",
                )

                detect_btn = gr.Button("Detect Persons", variant="primary", size="lg")

                preview_image = gr.Image(
                    label="Preview (click to select person)",
                    interactive=False,
                    type="numpy",
                    height=400,
                )

                person_radio = gr.Radio(
                    label="Or select from list",
                    choices=[],
                    interactive=True,
                )

                selection_status = gr.Textbox(
                    label="Selection Status",
                    value="Upload a video and click 'Detect Persons'",
                    interactive=False,
                )

                with gr.Accordion("Advanced Options", open=False):
                    layer_slider = gr.Slider(
                        label="HUD Layer",
                        minimum=0,
                        maximum=3,
                        step=1,
                        value=3,
                        info="0=skeleton, 1=velocity+trails+angles, 2=+axis, 3=full HUD",
                    )

                    render_scale_slider = gr.Slider(
                        label="Render Scale",
                        minimum=0.33,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        info="Lower = faster (0.5 recommended for 1080p+)",
                    )

                    use_3d_checkbox = gr.Checkbox(
                        label="Enable 3D-Corrected Overlay",
                        value=False,
                        info="Use CorrectiveLens for occlusion handling",
                    )

                    export_checkbox = gr.Checkbox(
                        label="Export Poses/CSV",
                        value=True,
                        info="Download pose data and biomechanics CSV",
                    )

                process_btn = gr.Button("Process Video", variant="primary", size="lg")

            # Right column: Outputs
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Analyzed Video",
                    autoplay=True,
                )

                poses_download = gr.File(
                    label="Download Poses (.npy)",
                    visible=False,
                )

                csv_download = gr.File(
                    label="Download Biomechanics (.csv)",
                    visible=False,
                )

                output_status = gr.Textbox(
                    label="Status",
                    value="Waiting for input...",
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
    )


if __name__ == "__main__":
    main()
