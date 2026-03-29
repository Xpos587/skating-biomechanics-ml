"""Главная компоновка страницы.

Main page layout composer for the Streamlit UI.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.components.export_dialog import export_dialog
from src.ui.components.sidebar import render_sidebar
from src.ui.components.video_player import (
    render_frame_slider,
    render_video_info,
)
from src.ui.core.config import UIConfig
from src.ui.core.events import EventBus
from src.ui.core.state import UIState
from src.ui.processors.layer_composer import LayerComposer
from src.ui.processors.pose_processor import PoseProcessor
from src.ui.types import LayerSettings
from src.video import extract_frames


class MainLayout:
    """Главная компоновка страницы.

    Main page layout composer that coordinates all UI components.
    """

    def __init__(
        self,
        state: UIState,
        config: UIConfig,
        events: EventBus,
    ) -> None:
        """Инициализация главной компоновки.

        Args:
            state: Менеджер состояния сессии.
            config: Менеджер настроек.
            events: Шина событий.
        """
        self._state = state
        self._config = config
        self._events = events
        self._pose_processor = PoseProcessor(events)
        self._layer_composer = LayerComposer()

    @classmethod
    def from_config(cls, config: UIConfig, events: EventBus) -> "MainLayout":
        """Создать из конфигурации.

        Args:
            config: Менеджер настроек.
            events: Шина событий.

        Returns:
            Экземпляр MainLayout.
        """
        state = UIState()
        return cls(state, config, events)

    def render(self) -> None:
        """Отрисовать главную страницу."""
        st.set_page_config(
            page_title="Анализ Фигурного Катания",
            page_icon="⛸️",
            layout="wide",
        )

        # Render sidebar and get settings
        settings, uploaded_file, export_clicked = render_sidebar(self._config, self._events)

        # Main content area
        if uploaded_file is not None:
            self._handle_uploaded_file(uploaded_file, settings, export_clicked)
        else:
            self._show_welcome()

    def _handle_uploaded_file(
        self,
        uploaded_file: Any,
        settings: LayerSettings,
        export_clicked: bool,
    ) -> None:
        """Обработать загруженный файл.

        Args:
            uploaded_file: Загруженный файл из Streamlit.
            settings: Настройки визуализации.
            export_clicked: Была ли нажата кнопка экспорта.
        """
        # Save uploaded file to temp
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = Path(tmp.name)

        # Check if video changed
        if self._state.video_path != video_path:
            self._state.video_path = video_path
            self._state.is_processed = False
            self._state.current_frame = 0

        # Process video if not done
        if not self._state.is_processed:
            st.info("🔄 Обработка видео...")
            try:
                poses = self._pose_processor.process(
                    video_path,
                    enable_3d=settings.enable_3d,
                    blade_3d=settings.blade_3d,
                    model_3d_type=settings.model_3d_type,
                )
                self._state.poses = poses
                self._state.is_processed = True
                st.success(f"✅ Обработка завершена: {len(poses.poses)} поз")
            except Exception as e:
                st.error(f"❌ Ошибка обработки: {e}")
                return

        # Get processed poses
        poses = self._state.poses
        if poses is None:
            return

        # Show video info
        render_video_info(
            poses.width,
            poses.height,
            poses.fps,
            poses.num_frames,
            poses.num_frames / poses.fps,
        )

        # Playback controls
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            is_playing = st.session_state.get("is_playing", False)
            if st.button("▶️ Play" if not is_playing else "⏸️ Pause", width="stretch"):
                st.session_state.is_playing = not is_playing
                st.rerun()

        with col2:
            if st.button("⏹️ Stop", width="stretch"):
                st.session_state.is_playing = False
                st.session_state.current_frame = 0
                self._state.current_frame = 0
                st.rerun()

        with col3:
            # Speed control
            playback_speed = st.select_slider(
                "Скорость",
                options=["0.5x", "1x", "2x", "5x"],
                value="1x",
                label_visibility="collapsed",
            )

        # Frame slider
        current_frame = render_frame_slider(poses.num_frames, self._state.current_frame)

        if current_frame != self._state.current_frame:
            self._state.current_frame = current_frame
            self._events.publish("frame:changed", current_frame)

        # Auto-play: advance frame and rerun
        if st.session_state.get("is_playing", False):
            import time

            speed_map = {"0.5x": 2.0, "1x": 1.0, "2x": 0.5, "5x": 0.2}
            delay = poses.fps / 30.0 * speed_map.get(playback_speed, 1.0)

            next_frame = min(current_frame + 1, poses.num_frames - 1)
            if next_frame != current_frame:
                st.session_state.current_frame = next_frame
                self._state.current_frame = next_frame
                time.sleep(max(0, delay))
                st.rerun()
            else:
                # End of video
                st.session_state.is_playing = False
                st.rerun()

        # Load frame and render with overlay
        try:
            from streamlit_overlay import overlay

            # Find the frame
            frame_iter = extract_frames(video_path)
            for i, frame in enumerate(frame_iter):
                if i == current_frame:
                    # Generate layer mask (independent from frame)
                    mask = self._layer_composer.compose_mask(
                        frame.shape,
                        current_frame,
                        poses,
                        settings,
                    )

                    # Use streamlit-overlay for true layer toggle (frontend only!)
                    overlay(
                        frame,
                        mask,
                        alpha=0.8,
                        toggle_label="🎨 Слои визуизации",
                        key=f"overlay_{current_frame}",
                    )
                    break
        except ImportError:
            st.error("❌ Установите streamlit-overlay: uv add streamlit-overlay")
        except Exception as e:
            st.error(f"❌ Ошибка отрисовки кадра: {e}")

        # Export dialog
        if export_clicked and poses is not None:
            with st.expander("💾 Экспорт видео", expanded=True):
                export_dialog(video_path, poses, settings, self._events)

    def _show_welcome(self) -> None:
        """Показать приветственное сообщение."""
        st.markdown(
            """
            # ⛸️ Анализ Фигурного Катания

            Загрузите видео для начала анализа.

            ## Возможности

            - **Визуализация скелета** - 17 ключевых точек H3.6M формата
            - **Вектора скорости** - цветовая индикация скорости суставов
            - **Траектории движения** - история движения с настраиваемой длиной
            - **Детекция ребра конька** - 3D определение ребра (inside/outside/flat)
            - **Траектория ЦТ** - центр масс для анализа прыжков

            ## Использование

            1. Загрузите видео через сайдбар
            2. Выберите слои визуализации
            3. Перематывайте кадры с помощью слайдера
            4. Экспортируйте результат с текущими настройками
            """
        )
