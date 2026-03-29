"""Сайдбар с элементами управления.

Sidebar widget with file upload and settings controls.
"""


import streamlit as st

from src.ui.core.config import UIConfig
from src.ui.core.events import EventBus
from src.ui.types import LayerSettings


def render_sidebar(config: UIConfig, events: EventBus | None = None) -> LayerSettings:
    """Отрисовать сайдбар с элементами управления.

    Args:
        config: Менеджер настроек.
        events: Шина событий.

    Returns:
        LayerSettings с текущими значениями из UI.
    """
    with st.sidebar:
        st.title("🎯 Анализ Катания")

        st.markdown("---")

        # File upload
        st.subheader("📤 Загрузить видео")
        uploaded_file = st.file_uploader(
            "Выберите видеофайл",
            type=["mp4", "webm", "mov", "avi"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Layer info (toggle is now in overlay component on frontend)
        st.caption("💡 Слои переключаются кнопкой под видео")

        st.markdown("---")

        # Advanced options
        st.subheader("🎚️ Дополнительно")
        advanced = config.get_advanced_settings()

        enable_3d = st.checkbox("3D поза", value=advanced.get("enable_3d", False), key="adv_3d")

        # 3D model selection (only show if 3D is enabled)
        model_3d_type = "motionagformer-s"
        if enable_3d:
            model_3d_type = st.selectbox(
                "3D модель",
                options=["motionagformer-s", "tcpformer"],
                index=0,
                format_func=lambda x: {
                    "motionagformer-s": "MotionAGFormer-S (59MB, быстрый)",
                    "tcpformer": "TCPFormer (422MB, точный)"
                }.get(x, x),
                key="adv_model_3d",
                help="MotionAGFormer-S: real-time, TCPFormer: higher accuracy"
            )

        blade_3d = st.checkbox("3D детекция ребра", value=advanced.get("blade_3d", False), key="adv_blade")
        com_trajectory = st.checkbox("Траектория ЦТ (CoM)", value=advanced.get("com_trajectory", False), key="adv_com")
        floor_mode = st.checkbox("Режим без коньков", value=advanced.get("floor_mode", False), key="adv_floor")

        # Disable blade_3d if 3d is off
        if blade_3d and not enable_3d:
            st.warning("3D детекция ребра требует включённой 3D позы")

        st.markdown("---")

        # Parameters
        st.subheader("⚙️ Параметры")
        params = config.get_parameters()

        trail_length = st.slider(
            "Длина следа",
            min_value=5,
            max_value=50,
            value=int(params.get("trail_length", 20)),
            step=1,
            help="Количество кадров для траектории движения",
        )

        d_3d_scale = st.slider(
            "3D масштаб",
            min_value=0.3,
            max_value=1.0,
            value=float(params.get("d_3d_scale", 0.6)),
            step=0.05,
            help="Масштаб 3D скелета в окне PIP",
        )

        font_size = st.slider(
            "Размер шрифта",
            min_value=16,
            max_value=48,
            value=int(params.get("font_size", 30)),
            step=2,
        )

        # Save parameters on change
        if st.session_state.get("params_changed", False):
            config.update_parameter("trail_length", trail_length)
            config.update_parameter("d_3d_scale", d_3d_scale)
            config.update_parameter("font_size", font_size)
            st.session_state.params_changed = False

        st.markdown("---")

        # Export button
        st.subheader("💾 Экспорт")
        export_clicked = st.button(
            "Экспортировать видео",
            width="stretch",
            type="primary",
        )

        # Build LayerSettings (layers always enabled, toggled via overlay)
        settings = LayerSettings(
            skeleton=True,
            velocity=True,
            trails=True,
            edge_indicators=True,
            subtitles=True,
            enable_3d=enable_3d,
            model_3d_type=model_3d_type if enable_3d else "motionagformer-s",
            blade_3d=blade_3d and enable_3d,  # Only if 3d enabled
            com_trajectory=com_trajectory and enable_3d,  # Only if 3d enabled
            floor_mode=floor_mode,
            trail_length=trail_length,
            d_3d_scale=d_3d_scale,
            font_size=font_size,
            no_3d_autoscale=False,
        )

        return settings, uploaded_file, export_clicked
