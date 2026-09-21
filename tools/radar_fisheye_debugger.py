#!/usr/bin/env python3
"""
MMAUD Radar -> Left Fisheye Interactive Debugger (v3)

Run:
    streamlit run radar_fisheye_debugger_v3.py

New in v3
---------
1. Multi-select Radar points from the point table.
2. Selected rows -> only selected Radar points are displayed.
3. No selection -> display all Radar points.
4. Optional "display selected only" switch.
5. Selection summary + selected-point table.
6. Interactive zoom / pan retained.
7. Zero removal / dedup / 6-DoF transform / optional GT retained.

GT is used ONLY for visual validation. It never participates in Radar projection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# MMAUD LEFT FISHEYE PARAMETERS
# ============================================================

LEFT_WIDTH = 1280
LEFT_HEIGHT = 960

XI = 2.9414247486244967
FU = 1844.9864629191054
FV = 1845.5803330648505
PU = 615.0248705672041
PV = 507.23461275157297

K1 = -0.35828592536315235
K2 = 0.3430453067389284
P1 = -0.0000797244897011588
P2 = -0.0006030718056190388


# ============================================================
# Geometry
# ============================================================

def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def radar_to_camera(
    radar_xyz: np.ndarray,
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
):
    R = build_rotation(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float64)
    camera_xyz = radar_xyz @ R.T + t
    return camera_xyz, R, t


def fisheye_project(camera_xyz: np.ndarray, require_in_image: bool = True):
    p = np.asarray(camera_xyz, dtype=np.float64)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    d = np.sqrt(X * X + Y * Y + Z * Z)
    denominator = Z + XI * d

    valid = (
        np.isfinite(p).all(axis=1)
        & np.isfinite(d)
        & (d > 1e-12)
        & np.isfinite(denominator)
        & (denominator > 1e-12)
    )

    x = np.full(len(p), np.nan, dtype=np.float64)
    y = np.full(len(p), np.nan, dtype=np.float64)

    x[valid] = X[valid] / denominator[valid]
    y[valid] = Y[valid] / denominator[valid]

    r2 = x * x + y * y
    radial = 1.0 + K1 * r2 + K2 * r2 * r2

    xd = (
        x * radial
        + 2.0 * P1 * x * y
        + P2 * (r2 + 2.0 * x * x)
    )

    yd = (
        y * radial
        + P1 * (r2 + 2.0 * y * y)
        + 2.0 * P2 * x * y
    )

    u = FU * xd + PU
    v = FV * yd + PV

    pixels = np.stack([u, v], axis=1)
    valid &= np.isfinite(pixels).all(axis=1)

    if require_in_image:
        valid &= (
            (u >= 0.0)
            & (u < LEFT_WIDTH)
            & (v >= 0.0)
            & (v < LEFT_HEIGHT)
        )

    return pixels, valid


# ============================================================
# Data IO
# ============================================================

@st.cache_data(show_spinner=False)
def load_radar_raw(path_str: str) -> np.ndarray:
    path = Path(path_str).expanduser()
    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Radar array must be (N, >=3), got {arr.shape}")

    xyz = np.asarray(arr[:, :3], dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


@st.cache_data(show_spinner=False)
def load_left_image(path_str: str) -> Image.Image:
    path = Path(path_str).expanduser()

    with Image.open(path) as im:
        im = im.convert("RGB")

        if im.size == (LEFT_WIDTH, LEFT_HEIGHT):
            return im.copy()

        if im.height == LEFT_HEIGHT and im.width >= 2560:
            return im.crop((0, 0, LEFT_WIDTH, LEFT_HEIGHT))

        raise ValueError(
            f"Expected 1280x960 left image or >=2560x960 stitched image, got {im.size}"
        )


def remove_zero_points(points: np.ndarray, eps: float):
    norms = np.linalg.norm(points, axis=1)
    keep = norms > eps
    return points[keep], keep


def deduplicate_xyz(points: np.ndarray, decimals: int):
    rounded = np.round(points, decimals=decimals)

    _, idx, counts = np.unique(
        rounded,
        axis=0,
        return_index=True,
        return_counts=True,
    )

    order = np.argsort(idx)
    idx = idx[order]
    counts = counts[order]

    return points[idx], counts


# ============================================================
# Optional GT validation
# ============================================================

def project_gt(gt_path: str, calibration_json_path: str):
    gt = np.load(Path(gt_path).expanduser(), allow_pickle=False)

    if gt.shape != (3,):
        raise ValueError(f"GT must have shape (3,), got {gt.shape}")

    payload = json.loads(
        Path(calibration_json_path).expanduser().read_text(encoding="utf-8")
    )
    left = payload["cameras"]["left"]

    R = np.asarray(
        left["rotation_camera_from_gt"],
        dtype=np.float64,
    ).reshape(3, 3)

    t = np.asarray(
        left["translation_camera_from_gt_m"],
        dtype=np.float64,
    ).reshape(3)

    gt_camera = gt.reshape(1, 3) @ R.T + t
    pixels, valid = fisheye_project(gt_camera, require_in_image=True)

    return gt, gt_camera[0], pixels[0], bool(valid[0])


# ============================================================
# Interactive figure
# ============================================================

def make_interactive_figure(
    image: Image.Image,
    pixels: np.ndarray,
    valid: np.ndarray,
    indices_to_show: list[int],
    multiplicity: np.ndarray,
    point_size: int,
    show_labels: bool,
    gt_pixel: np.ndarray | None,
):
    fig = go.Figure()

    fig.add_trace(go.Image(z=np.asarray(image)))

    show_valid = [i for i in indices_to_show if valid[i]]

    if show_valid:
        x = [float(pixels[i, 0]) for i in show_valid]
        y = [float(pixels[i, 1]) for i in show_valid]

        labels = [
            f"R{i} x{int(multiplicity[i])}"
            if multiplicity[i] > 1
            else f"R{i}"
            for i in show_valid
        ]

        hover = [
            (
                f"R{i}<br>"
                f"u={pixels[i,0]:.2f}, v={pixels[i,1]:.2f}<br>"
                f"multiplicity={int(multiplicity[i])}"
            )
            for i in show_valid
        ]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text" if show_labels else "markers",
                text=labels if show_labels else None,
                textposition="top right",
                marker=dict(
                    size=max(5, point_size * 2),
                    color="red",
                    line=dict(color="white", width=1),
                ),
                hovertext=hover,
                hoverinfo="text",
                name="Radar",
            )
        )

    if gt_pixel is not None:
        fig.add_trace(
            go.Scatter(
                x=[float(gt_pixel[0])],
                y=[float(gt_pixel[1])],
                mode="markers",
                marker=dict(
                    size=16,
                    symbol="x",
                    color="lime",
                    line=dict(width=3),
                ),
                name="GT",
                hovertext=f"GT ({gt_pixel[0]:.2f}, {gt_pixel[1]:.2f})",
                hoverinfo="text",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=32, b=0),
        height=720,
        dragmode="pan",
        legend=dict(orientation="h"),
        title="滚轮缩放 / 鼠标拖动 / 双击复位",
    )

    fig.update_xaxes(
        range=[0, LEFT_WIDTH],
        showgrid=False,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[LEFT_HEIGHT, 0],
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def make_3d_figure(
    points: np.ndarray,
    multiplicity: np.ndarray,
    point_size: int,
    color_mode: str,
    highlight_indices: list[int],
    frame_label: str,
):
    fig = go.Figure()

    n = len(points)

    if color_mode == "高度 Z":
        color = points[:, 2]
        cbar_title = "Z (m)"
    elif color_mode == "距离":
        color = np.linalg.norm(points, axis=1)
        cbar_title = "||XYZ|| (m)"
    else:
        color = np.arange(n)
        cbar_title = "索引"

    hover = [
        f"P{i}<br>x={points[i, 0]:.3f}<br>y={points[i, 1]:.3f}"
        f"<br>z={points[i, 2]:.3f}<br>multiplicity={int(multiplicity[i])}"
        for i in range(n)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=color,
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title=cbar_title),
            ),
            hovertext=hover,
            hoverinfo="text",
            name="Radar 点云",
        )
    )

    if highlight_indices:
        sel = np.asarray(
            [i for i in highlight_indices if 0 <= i < n], dtype=int
        )

        if len(sel) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=points[sel, 0],
                    y=points[sel, 1],
                    z=points[sel, 2],
                    mode="markers",
                    marker=dict(
                        size=max(point_size * 2, 6),
                        color="red",
                        symbol="diamond",
                        line=dict(color="white", width=1),
                    ),
                    hovertext=[hover[i] for i in sel],
                    hoverinfo="text",
                    name=f"已选点 ({len(sel)})",
                )
            )

    axis_len = float(np.max(np.abs(points))) * 1.1 if n else 1.0

    for axis, color, name in (
        ((axis_len, 0, 0), "red", "X"),
        ((0, axis_len, 0), "green", "Y"),
        ((0, 0, axis_len), "blue", "Z"),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, axis[0]],
                y=[0.0, axis[1]],
                z=[0.0, axis[2]],
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=32, b=0),
        height=640,
        title=f"3D 点云（{frame_label}）— 拖动旋转 / 滚轮缩放",
        legend=dict(orientation="h"),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )

    return fig


def render_save_image(
    image: Image.Image,
    pixels: np.ndarray,
    valid: np.ndarray,
    indices_to_show: list[int],
    point_radius: int,
    gt_pixel: np.ndarray | None,
):
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")

    for i in indices_to_show:
        if not valid[i]:
            continue

        u, v = pixels[i]
        r = point_radius

        draw.ellipse(
            (u-r, v-r, u+r, v+r),
            fill=(255, 0, 0, 190),
            outline=(255, 255, 255, 220),
            width=1,
        )
        draw.text(
            (u+r+2, v-r),
            f"R{i}",
            fill=(255, 255, 0, 255),
        )

    if gt_pixel is not None:
        u, v = gt_pixel
        draw.line((u-12, v, u+12, v), fill=(0,255,0,255), width=3)
        draw.line((u, v-12, u, v+12), fill=(0,255,0,255), width=3)

    return out


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="MMAUD Radar → Fisheye Debugger v3",
    layout="wide",
)

st.title("MMAUD Radar → 左鱼眼投影调试器 v3")
st.caption(
    "支持 Radar 多选：表格可同时选择多个点，图中只显示所选点。"
)

with st.sidebar:
    st.header("1. 输入文件")

    image_path = st.text_input(
        "鱼眼图像路径",
        value="/home/jasoncui/datasets/MMAUD/official/v1/Mavic3/image/1692846905.705314.png",
    )

    radar_path = st.text_input(
        "Radar NPY 路径",
        value="/home/jasoncui/datasets/MMAUD/official/v1/Mavic3/livox_avia/1692846905.715745.npy",
    )

    st.divider()
    st.header("2. 数据清洗")

    remove_zero = st.checkbox(
        "删除 0 / 近 0 点",
        value=True,
    )

    zero_eps = st.number_input(
        "零点阈值 ||XYZ|| ≤ eps",
        min_value=0.0,
        value=1e-6,
        step=1e-6,
        format="%.8f",
        disabled=not remove_zero,
    )

    dedup = st.checkbox(
        "Radar XYZ 去重",
        value=True,
    )

    dedup_decimals = st.number_input(
        "去重小数位",
        min_value=0,
        max_value=8,
        value=3,
        step=1,
        disabled=not dedup,
    )

    st.divider()
    st.header("3. Radar → Camera 参数")

    rx = st.number_input("rx (deg)", value=0.0, step=0.1, format="%.4f")
    ry = st.number_input("ry (deg)", value=0.0, step=0.1, format="%.4f")
    rz = st.number_input("rz (deg)", value=0.0, step=0.1, format="%.4f")

    tx = st.number_input("tx (m)", value=0.0, step=0.01, format="%.4f")
    ty = st.number_input("ty (m)", value=0.0, step=0.01, format="%.4f")
    tz = st.number_input("tz (m)", value=0.0, step=0.01, format="%.4f")

    st.divider()
    st.header("4. 显示")

    point_size = st.slider(
        "点大小",
        min_value=2,
        max_value=15,
        value=5,
    )

    show_labels = st.checkbox(
        "显示点编号",
        value=True,
    )

    selected_only = st.checkbox(
        "有选择时仅显示已选点",
        value=True,
        help="关闭后，即使表格有选择，也仍显示全部 Radar 点。",
    )

    st.caption("图像支持滚轮缩放、拖动和平移。")

    st.divider()
    st.header("5. 3D 点云")

    cloud_enabled = st.checkbox(
        "显示 3D 点云",
        value=True,
    )

    cloud_path = st.text_input(
        "3D 点云 NPY 路径",
        value=radar_path,
        help="输入任意 Radar NPY 文件名即可显示其 3D 点云；"
        "默认使用上方 Radar 文件。",
        disabled=not cloud_enabled,
    )

    cloud_frame = st.radio(
        "坐标系",
        options=["radar", "camera"],
        format_func={
            "radar": "Radar 坐标系",
            "camera": "Camera 坐标系（应用 6-DoF）",
        }.get,
        horizontal=True,
        index=0,
        disabled=not cloud_enabled,
    )

    cloud_point_size = st.slider(
        "3D 点大小",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not cloud_enabled,
    )

    cloud_color_mode = st.selectbox(
        "着色方式",
        options=["高度 Z", "距离", "索引"],
        disabled=not cloud_enabled,
    )

    st.caption("3D 视图同样应用第 2 节的去零 / 去重设置。")

    st.divider()
    st.header("6. GT 验证（可选）")

    enable_gt = st.checkbox(
        "显示 GT 参考位置",
        value=False,
    )

    gt_path = st.text_input(
        "GT NPY 路径",
        value="/home/jasoncui/datasets/MMAUD/official/v1/Mavic3/ground_truth/1692846903.933849.npy",
        disabled=not enable_gt,
    )

    gt_calibration_path = st.text_input(
        "GT → Camera calibration JSON",
        value="/home/jasoncui/projects/rdq-uav-experiment/calibration/official_left_fitted_calibration.json",
        disabled=not enable_gt,
    )

    st.divider()
    st.header("7. 保存")

    output_dir = st.text_input(
        "输出目录",
        value="/home/jasoncui/projects/rdq-uav-experiment/outputs/radar_fisheye_debugger_v3",
    )


# ============================================================
# Compute
# ============================================================

try:
    image = load_left_image(image_path)
    radar_raw = load_radar_raw(radar_path)

    if remove_zero:
        radar_nozero, _ = remove_zero_points(
            radar_raw,
            float(zero_eps),
        )
    else:
        radar_nozero = radar_raw

    zero_removed_count = len(radar_raw) - len(radar_nozero)

    if dedup:
        radar_display, multiplicity = deduplicate_xyz(
            radar_nozero,
            int(dedup_decimals),
        )
    else:
        radar_display = radar_nozero
        multiplicity = np.ones(
            len(radar_display),
            dtype=np.int64,
        )

    camera_xyz, R, t = radar_to_camera(
        radar_display,
        rx, ry, rz,
        tx, ty, tz,
    )

    pixels, valid = fisheye_project(
        camera_xyz,
        require_in_image=True,
    )

    gt_pixel = None
    gt_info = None

    if enable_gt:
        gt_xyz, gt_camera, gt_pixel_candidate, gt_valid = project_gt(
            gt_path,
            gt_calibration_path,
        )

        if gt_valid:
            gt_pixel = gt_pixel_candidate

        gt_info = {
            "gt_xyz": gt_xyz.tolist(),
            "gt_camera_xyz": gt_camera.tolist(),
            "gt_pixel": gt_pixel_candidate.tolist(),
            "gt_valid": bool(gt_valid),
        }

except Exception as exc:
    st.error(f"运行失败：{exc}")
    st.exception(exc)
    st.stop()


# ============================================================
# Dataframe
# ============================================================

df = pd.DataFrame({
    "index": np.arange(len(radar_display), dtype=int),
    "radar_x": radar_display[:, 0],
    "radar_y": radar_display[:, 1],
    "radar_z": radar_display[:, 2],
    "camera_x": camera_xyz[:, 0],
    "camera_y": camera_xyz[:, 1],
    "camera_z": camera_xyz[:, 2],
    "u": pixels[:, 0],
    "v": pixels[:, 1],
    "valid": valid,
    "multiplicity": multiplicity,
})


# ============================================================
# Multi-row selection
# ============================================================

st.subheader("投影点数据")
st.caption(
    "可多选：按 Ctrl / Command 逐个选择多行，或按 Shift 连续选择。"
    " 没有选择时显示全部点。"
)

selection_event = st.dataframe(
    df,
    key="radar_point_table_v3",
    on_select="rerun",
    selection_mode="multi-row",
    width="stretch",
    height=320,
)

try:
    selected_table_rows = list(selection_event.selection.rows)
except Exception:
    selected_table_rows = []

# ============================================================
# Safe multi-row selection
# ============================================================

try:
    raw_selected_rows = selection_event.selection.rows
except Exception:
    raw_selected_rows = []

# Streamlit 有时会返回 None，直接过滤掉
selected_table_rows = [
    int(row_idx)
    for row_idx in (raw_selected_rows or [])
    if row_idx is not None
    and isinstance(row_idx, (int, np.integer))
    and 0 <= int(row_idx) < len(df)
]

selected_indices = [
    int(df.iloc[row_idx]["index"])
    for row_idx in selected_table_rows
]

# 有有效选择 → 显示选择的点
# 没有选择 / 选择异常 → 自动恢复显示全部点
if selected_only and len(selected_indices) > 0:
    indices_to_show = selected_indices
else:
    indices_to_show = list(range(len(radar_display)))

if selected_only and selected_indices:
    indices_to_show = selected_indices
else:
    indices_to_show = list(range(len(radar_display)))


# ============================================================
# Interactive display
# ============================================================

st.subheader("实时投影结果")

fig = make_interactive_figure(
    image=image,
    pixels=pixels,
    valid=valid,
    indices_to_show=indices_to_show,
    multiplicity=multiplicity,
    point_size=int(point_size),
    show_labels=show_labels,
    gt_pixel=gt_pixel,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
    },
)

status_cols = st.columns(6)
status_cols[0].metric("原始点", len(radar_raw))
status_cols[1].metric("去零删除", zero_removed_count)
status_cols[2].metric("去零后", len(radar_nozero))
status_cols[3].metric("去重后", len(radar_display))
status_cols[4].metric("有效投影", int(valid.sum()))
status_cols[5].metric("当前选中", len(selected_indices))

# ============================================================
# 3D point cloud
# ============================================================

if cloud_enabled:
    st.subheader("Radar 3D 点云")

    try:
        cloud_raw = load_radar_raw(cloud_path)

        if remove_zero:
            cloud_nozero, _ = remove_zero_points(
                cloud_raw,
                float(zero_eps),
            )
        else:
            cloud_nozero = cloud_raw

        if dedup:
            cloud_pts, cloud_mult = deduplicate_xyz(
                cloud_nozero,
                int(dedup_decimals),
            )
        else:
            cloud_pts = cloud_nozero
            cloud_mult = np.ones(
                len(cloud_pts),
                dtype=np.int64,
            )

        if cloud_frame == "camera":
            cloud_pts, _, _ = radar_to_camera(
                cloud_pts,
                rx, ry, rz,
                tx, ty, tz,
            )
            frame_label = "Camera 坐标系"
        else:
            frame_label = "Radar 坐标系"

        highlight = (
            selected_indices if cloud_path == radar_path else []
        )

        fig_3d = make_3d_figure(
            points=cloud_pts,
            multiplicity=cloud_mult,
            point_size=int(cloud_point_size),
            color_mode=cloud_color_mode,
            highlight_indices=highlight,
            frame_label=frame_label,
        )

        st.plotly_chart(
            fig_3d,
            width="stretch",
            config={"displaylogo": False},
        )

        st.caption(
            f"文件：{cloud_path}　"
            f"原始 {len(cloud_raw)} 点 → 清洗后 {len(cloud_pts)} 点"
        )

    except Exception as exc:
        st.error(f"3D 点云加载失败：{exc}")

if selected_indices:
    st.success(
        "当前选择 Radar 点：" +
        ", ".join(f"R{i}" for i in selected_indices)
    )

    selected_df = df[df["index"].isin(selected_indices)].copy()

    st.write("**已选择点详情**")
    st.dataframe(
        selected_df,
        width="stretch",
        height=min(260, 38 * (len(selected_df) + 1)),
    )

    if gt_pixel is not None:
        selected_valid = [
            i for i in selected_indices
            if valid[i]
        ]

        if selected_valid:
            dist_rows = []

            for i in selected_valid:
                d = float(
                    np.linalg.norm(
                        pixels[i] - gt_pixel
                    )
                )
                dist_rows.append(
                    {
                        "index": i,
                        "distance_to_gt_px": d,
                    }
                )

            dist_df = pd.DataFrame(dist_rows).sort_values(
                "distance_to_gt_px"
            )

            st.write("**所选 Radar 点到 GT 的像素距离**")
            st.dataframe(
                dist_df,
                width="stretch",
                hide_index=True,
            )

else:
    st.info("当前未选择 Radar 点，因此显示全部点。")


# ============================================================
# Diagnostics
# ============================================================

with st.expander("当前参数与矩阵"):
    st.write(
        {
            "rx_deg": rx,
            "ry_deg": ry,
            "rz_deg": rz,
            "tx_m": tx,
            "ty_m": ty,
            "tz_m": tz,
            "rotation_convention": "R = Rz @ Ry @ Rx",
        }
    )

    st.code(
        "R =\n" +
        np.array2string(R, precision=6)
    )

    st.code(
        "t = " +
        np.array2string(t, precision=6)
    )


# ============================================================
# Save
# ============================================================

if st.button("保存当前结果", type="primary"):
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    overlay_path = out / "overlay_left.png"
    csv_path = out / "projection_points.csv"
    json_path = out / "parameters.json"

    save_image = render_save_image(
        image=image,
        pixels=pixels,
        valid=valid,
        indices_to_show=indices_to_show,
        point_radius=int(point_size),
        gt_pixel=gt_pixel,
    )

    save_image.save(overlay_path)
    df.to_csv(csv_path, index=False)

    payload = {
        "image": image_path,
        "radar": radar_path,
        "cleaning": {
            "remove_zero": remove_zero,
            "zero_eps": zero_eps,
            "zero_removed_count": zero_removed_count,
            "deduplicate": dedup,
            "deduplicate_decimals": (
                dedup_decimals if dedup else None
            ),
        },
        "transform": {
            "rx_deg": rx,
            "ry_deg": ry,
            "rz_deg": rz,
            "tx_m": tx,
            "ty_m": ty,
            "tz_m": tz,
            "rotation_convention": "R = Rz @ Ry @ Rx",
            "R_camera_from_radar": R.tolist(),
            "t_camera_from_radar_m": t.tolist(),
        },
        "counts": {
            "raw": len(radar_raw),
            "after_zero_removal": len(radar_nozero),
            "after_dedup": len(radar_display),
            "valid_projected": int(valid.sum()),
        },
        "selected_indices": selected_indices,
        "displayed_indices": indices_to_show,
        "gt_validation": gt_info,
    }

    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    st.success(
        f"已保存：\n\n"
        f"{overlay_path}\n\n"
        f"{csv_path}\n\n"
        f"{json_path}"
    )
