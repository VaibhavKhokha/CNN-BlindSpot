from functools import partial
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from PIL import Image, UnidentifiedImageError


def _sync_tensor_cell_selection(widget_key, selection_key, row_start, column_start):
    grid_state = st.session_state.get(widget_key)
    if grid_state is None:
        return

    selected_cells = grid_state.selection.cells
    if selected_cells:
        selected_cell = selected_cells[0]
        st.session_state[selection_key] = (
            row_start + selected_cell["row"],
            column_start + selected_cell["column"],
        )


def _tensor_cell_style(value):
    intensity = int(value)
    text_color = "#111827" if intensity >= 145 else "#f8fafc"
    return (
        f"background-color: rgb({intensity}, {intensity}, {intensity}); "
        f"color: {text_color};"
    )


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def _binary_cross_entropy(prediction, target):
    prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
    return -(
        target * np.log(prediction)
        + (1 - target) * np.log(1 - prediction)
    )


def _softmax(logits):
    shifted_logits = np.asarray(logits, dtype=np.float64) - np.max(logits)
    exponentials = np.exp(shifted_logits)
    return exponentials / exponentials.sum()


def _render_tensor_matrix(
    tensor_channel,
    channel_label,
    image_id,
    selection_key,
    region_size,
    row_start,
    column_start,
    zoom,
    show_values,
):
    height, width = tensor_channel.shape
    visible_rows = min(region_size, height - row_start)
    visible_columns = min(region_size, width - column_start)
    matrix = tensor_channel[
        row_start : row_start + visible_rows,
        column_start : column_start + visible_columns,
    ]
    matrix_frame = pd.DataFrame(
        matrix,
        index=np.arange(row_start, row_start + visible_rows),
        columns=np.arange(column_start, column_start + visible_columns),
    )
    matrix_frame.index.name = "row"
    matrix_frame.columns.name = "column"

    widget_key = (
        f"tensor_grid_{image_id}_{channel_label}_{region_size}_"
        f"{row_start}_{column_start}"
    )
    selected_row, selected_column = st.session_state.get(
        selection_key, (row_start, column_start)
    )
    selected_in_region = (
        row_start <= selected_row < row_start + visible_rows
        and column_start <= selected_column < column_start + visible_columns
    )
    cell_styles = pd.DataFrame("", index=matrix_frame.index, columns=matrix_frame.columns)
    if selected_in_region:
        cell_styles.loc[selected_row, selected_column] = (
            "border: 2px solid #facc15 !important; "
            "box-shadow: inset 0 0 0 2px #facc15;"
        )

    styled_matrix = (
        matrix_frame.style
        .map(_tensor_cell_style)
        .apply(lambda _: cell_styles, axis=None)
        .format(lambda value: str(int(value)) if show_values else "")
        .set_properties(
            **{
                "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
                "font-size": f"{round(12 * zoom)}px",
                "min-width": f"{round(44 * zoom)}px",
                "text-align": "center",
                "border": "1px solid #46505e",
                "padding": f"{round(4 * zoom)}px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#202733"),
                        ("color", "#e5e7eb"),
                        ("font-family", "ui-monospace, monospace"),
                        ("text-align", "center"),
                        ("border", "1px solid #46505e"),
                    ],
                }
            ]
        )
    )
    st.dataframe(
        styled_matrix,
        width="stretch",
        height=min(680, max(220, 90 + visible_rows * round(30 * zoom))),
        row_height=round(30 * zoom),
        hide_index=False,
        key=widget_key,
        on_select=partial(
            _sync_tensor_cell_selection,
            widget_key,
            selection_key,
            row_start,
            column_start,
        ),
        selection_mode="single-cell",
    )

    selected_row, selected_column = st.session_state.get(
        selection_key, (row_start, column_start)
    )
    if 0 <= selected_row < height and 0 <= selected_column < width:
        st.info(
            f"Selected {channel_label} cell — row **{selected_row}**, "
            f"column **{selected_column}**, intensity **"
            f"{int(tensor_channel[selected_row, selected_column])}**."
        )


def render_knowledge_check(key, question, options, correct_idx, explanation):
    st.markdown("---")
    st.write(f"**Knowledge Check:** {question}")
    state_key = f"quiz_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    choice = st.radio("Select an answer:", options, index=None, key=f"radio_{key}")
    
    if st.button("Submit", key=f"btn_{key}"):
        st.session_state[state_key] = choice
        
    if st.session_state[state_key] is not None:
        if st.session_state[state_key] == options[correct_idx]:
            st.success("Correct!")
        else:
            st.error("Incorrect. Review the explanation below.")
        st.info(f"**Explanation:** {explanation}")

def chapter_1():
    st.header("Chapter 1: How a Computer Sees an Image")
    st.markdown("""
    ### 1.1 The Digital Image & Tensors
    When humans look at a photograph, our brains instantly process shapes, context, and depth. A computer, however, only sees a giant grid of numbers. 
    
    An image is constructed from thousands of tiny squares called **pixels**. In a standard color image, every single pixel is represented by three overlapping color channels: **Red, Green, and Blue (RGB)**. The intensity of each color is measured on a scale from 0 (completely dark) to 255 (maximum brightness).
    
    In Deep Learning, we store this data in multi-dimensional matrices called **Tensors**. If you have a 224x224 pixel color image, it is loaded into memory as a 3D tensor with the shape `(224, 224, 3)`. That is exactly 150,528 individual numbers that the neural network must process!
    """)
    
    st.markdown("#### Interactive Demo: RGB Pixels")
    st.write("Adjust the sliders to see how combining three numerical intensities creates the colors we see on screen.")
    c1, c2, c3 = st.columns(3)
    r = c1.slider("Red (0-255)", 0, 255, 255)
    g = c2.slider("Green (0-255)", 0, 255, 128)
    b = c3.slider("Blue (0-255)", 0, 255, 0)
    color_hex = f"#{r:02x}{g:02x}{b:02x}"
    st.markdown(f'<div style="background-color: {color_hex}; height: 80px; border-radius: 8px; border: 1px solid #555;"></div>', unsafe_allow_html=True)

    st.markdown("#### Interactive Demo: Image to Tensor")
    st.write(
        "Upload an image to explore its pixel values in an interactive tensor "
        "matrix, with a synchronized grayscale heatmap or 3D RGB visualization."
    )
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="chapter_1_image",
    )
    tensor_mode = st.radio(
        "Tensor color mode",
        ["RGB", "Grayscale"],
        horizontal=True,
        key="chapter_1_tensor_mode",
    )

    if uploaded_image is not None:
        try:
            with Image.open(uploaded_image) as image:
                image_tensor = np.asarray(
                    image.convert("RGB" if tensor_mode == "RGB" else "L"),
                    dtype=np.uint8,
                )
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            st.error(f"Could not read the uploaded image: {exc}")
        else:
            height, width = image_tensor.shape[:2]
            image_id = uploaded_image.file_id
            selection_key = f"tensor_selected_{image_id}"
            st.session_state.setdefault(selection_key, (0, 0))
            selected_row, selected_column = st.session_state[selection_key]
            selected_row = min(max(selected_row, 0), height - 1)
            selected_column = min(max(selected_column, 0), width - 1)
            st.session_state[selection_key] = (selected_row, selected_column)

            st.markdown("##### Tensor metadata")
            metadata_columns = st.columns(3)
            metadata_columns[0].metric("Full tensor shape", str(image_tensor.shape))
            metadata_columns[1].metric("Dimensions", f"{height} × {width}")
            metadata_columns[2].metric("Data type", str(image_tensor.dtype))
            metadata_columns = st.columns(3)
            metadata_columns[0].metric("Elements", f"{image_tensor.size:,}")
            metadata_columns[1].metric(
                "Channels", "3 (R, G, B)" if tensor_mode == "RGB" else "1 (grayscale)"
            )
            metadata_columns[2].metric(
                "Intensity range",
                f"{int(image_tensor.min())}–{int(image_tensor.max())}",
            )
            st.caption(
                "Metadata describes the full uploaded tensor. The matrix below "
                "shows only the selected region."
            )

            matrix_controls = st.columns([1, 1, 1, 1])
            with matrix_controls[0]:
                region_size = st.selectbox(
                    "Matrix region",
                    [8, 16, 32],
                    index=1,
                    format_func=lambda size: f"{size} × {size}",
                    key=f"tensor_region_{image_id}",
                )
            with matrix_controls[1]:
                zoom = st.select_slider(
                    "Cell zoom",
                    options=[1.0, 1.25, 1.5, 2.0],
                    value=1.25,
                    key=f"tensor_zoom_{image_id}",
                )
            with matrix_controls[2]:
                show_values = st.toggle(
                    "Show values",
                    value=True,
                    key=f"tensor_values_{image_id}",
                )

            max_row_start = max(0, height - region_size)
            max_column_start = max(0, width - region_size)
            start_controls = st.columns(2)
            with start_controls[0]:
                if max_row_start:
                    row_start = st.slider(
                        "Starting row",
                        min_value=0,
                        max_value=max_row_start,
                        value=0,
                        key=f"tensor_row_start_{image_id}_{region_size}",
                    )
                else:
                    row_start = 0
                    st.caption("Starting row: 0")
            with start_controls[1]:
                if max_column_start:
                    column_start = st.slider(
                        "Starting column",
                        min_value=0,
                        max_value=max_column_start,
                        value=0,
                        key=f"tensor_column_start_{image_id}_{region_size}",
                    )
                else:
                    column_start = 0
                    st.caption("Starting column: 0")

            image_column, tensor_column = st.columns([0.9, 1.5], gap="large")
            with image_column:
                st.markdown("**Uploaded image**")
                preview_scale = min(1.0, 512 / max(height, width))
                preview_width = max(1, round(width * preview_scale))
                preview_height = max(1, round(height * preview_scale))
                preview_image = np.asarray(
                    Image.fromarray(image_tensor).resize(
                        (preview_width, preview_height),
                        Image.Resampling.BILINEAR,
                    )
                )
                image_figure = go.Figure(go.Image(z=preview_image))
                image_figure.add_trace(
                    go.Scatter(
                        x=[(selected_column + 0.5) * preview_width / width - 0.5],
                        y=[(selected_row + 0.5) * preview_height / height - 0.5],
                        mode="markers",
                        marker={
                            "symbol": "circle-open",
                            "size": 16,
                            "color": "#facc15",
                            "line": {"color": "#facc15", "width": 3},
                        },
                        hovertemplate=(
                            f"row: {selected_row}<br>column: {selected_column}"
                            "<extra>Selected pixel</extra>"
                        ),
                        showlegend=False,
                    )
                )
                image_figure.update_layout(
                    height=460,
                    margin={"l": 0, "r": 0, "t": 10, "b": 10},
                    xaxis={"title": "Column", "constrain": "domain"},
                    yaxis={
                        "title": "Row",
                        "autorange": "reversed",
                        "scaleanchor": "x",
                    },
                    uirevision=f"image-{image_id}",
                )
                st.plotly_chart(
                    image_figure,
                    width="stretch",
                    key=f"uploaded_preview_{image_id}",
                )
            with tensor_column:
                st.markdown("### Tensor matrix")
                visible_rows = min(region_size, height - row_start)
                visible_columns = min(region_size, width - column_start)
                st.caption(
                    f"Selected region: {visible_rows} × {visible_columns} pixels "
                    f"(rows {row_start}–{row_start + visible_rows - 1}, columns "
                    f"{column_start}–{column_start + visible_columns - 1}); "
                    f"full tensor shape: `{image_tensor.shape}`."
                )
                st.caption(
                    "Select any grid cell to highlight it in the heatmap and "
                    "uploaded image. Use the sliders to pan the matrix."
                )
                if tensor_mode == "Grayscale":
                    _render_tensor_matrix(
                        image_tensor,
                        "grayscale",
                        image_id,
                        selection_key,
                        region_size,
                        row_start,
                        column_start,
                        zoom,
                        show_values,
                    )
                else:
                    red_tab, green_tab, blue_tab = st.tabs(
                        ["Red (R)", "Green (G)", "Blue (B)"]
                    )
                    for tab, channel_index, channel_label in (
                        (red_tab, 0, "red"),
                        (green_tab, 1, "green"),
                        (blue_tab, 2, "blue"),
                    ):
                        with tab:
                            _render_tensor_matrix(
                                image_tensor[:, :, channel_index],
                                channel_label,
                                image_id,
                                selection_key,
                                region_size,
                                row_start,
                                column_start,
                                zoom,
                                show_values,
                            )

                st.markdown("### Tensor heatmap")
                if tensor_mode == "Grayscale":
                    heatmap_stride = max(
                        1, int(np.ceil(max(height, width) / 512))
                    )
                    heatmap = image_tensor[::heatmap_stride, ::heatmap_stride]
                    heatmap_rows = np.arange(0, height, heatmap_stride)
                    heatmap_columns = np.arange(0, width, heatmap_stride)
                    figure = go.Figure(
                        go.Heatmap(
                            z=heatmap,
                            x=heatmap_columns,
                            y=heatmap_rows,
                            colorscale="Gray",
                            zmin=0,
                            zmax=255,
                            colorbar={"title": "Intensity (0–255)"},
                            hovertemplate=(
                                "row: %{y}<br>column: %{x}"
                                "<br>intensity: %{z}<extra></extra>"
                            ),
                        )
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=[selected_column],
                            y=[selected_row],
                            mode="markers",
                            marker={
                                "symbol": "square-open",
                                "size": 14,
                                "color": "#facc15",
                                "line": {"color": "#facc15", "width": 3},
                            },
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    figure.update_layout(
                        xaxis_title="Column",
                        yaxis_title="Row",
                        yaxis_autorange="reversed",
                        height=560,
                        margin={"l": 20, "r": 20, "t": 15, "b": 30},
                        uirevision=f"gray-heatmap-{image_id}",
                    )
                else:
                    flat_tensor = image_tensor.reshape(-1, 3)
                    max_points = 5000
                    stride = max(1, int(np.ceil(len(flat_tensor) / max_points)))
                    sampled_pixels = flat_tensor[::stride]
                    pixel_indices = np.arange(0, len(flat_tensor), stride)
                    pixel_colors = [
                        f"rgb({red},{green},{blue})"
                        for red, green, blue in sampled_pixels
                    ]
                    figure = go.Figure(
                        go.Scatter3d(
                            x=sampled_pixels[:, 0],
                            y=sampled_pixels[:, 1],
                            z=sampled_pixels[:, 2],
                            mode="markers",
                            marker={"size": 3, "color": pixel_colors, "opacity": 0.8},
                            customdata=np.column_stack(
                                (pixel_indices // width, pixel_indices % width)
                            ),
                            hovertemplate=(
                                "row: %{customdata[0]}<br>column: %{customdata[1]}"
                                "<br>R: %{x}<br>G: %{y}<br>B: %{z}<extra></extra>"
                            ),
                        )
                    )
                    figure.add_trace(
                        go.Scatter3d(
                            x=[int(image_tensor[selected_row, selected_column, 0])],
                            y=[int(image_tensor[selected_row, selected_column, 1])],
                            z=[int(image_tensor[selected_row, selected_column, 2])],
                            mode="markers",
                            marker={
                                "symbol": "diamond",
                                "size": 9,
                                "color": "#facc15",
                                "line": {"color": "#111827", "width": 2},
                            },
                            hovertemplate=(
                                f"row: {selected_row}<br>column: {selected_column}"
                                "<br>Selected pixel<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )
                    figure.update_layout(
                        scene={
                            "xaxis_title": "Red",
                            "yaxis_title": "Green",
                            "zaxis_title": "Blue",
                            "xaxis_range": [0, 255],
                            "yaxis_range": [0, 255],
                            "zaxis_range": [0, 255],
                        },
                        height=560,
                        margin={"l": 0, "r": 0, "t": 15, "b": 10},
                        uirevision=f"rgb-heatmap-{image_id}",
                    )
                st.plotly_chart(
                    figure,
                    width="stretch",
                    key=f"tensor_heatmap_{image_id}_{tensor_mode}",
                )

    st.markdown("""
    ### 1.2 The Preprocessing Pipeline
    Neural networks are rigid mathematical functions. They cannot dynamically adjust to images of random sizes. Furthermore, feeding numbers as large as 255 into complex calculus equations causes a problem called "exploding gradients," where the math breaks down.
    
    Before an image reaches the neural network, it must pass through a strict **preprocessing pipeline**:
    1. **Resizing & Cropping:** The image is forced into a fixed square (e.g., 224x224 pixels).
    2. **Normalization:** The pixel values are mathematically scaled down from the `0 to 255` range into a smaller decimal range (usually between `-1.0` and `1.0`, or `0.0` and `1.0`). This keeps the network's calculations stable.
    """)
    render_knowledge_check("ch1_q1", "Why must we normalize pixel values before feeding them into a Neural Network?", 
        [
            "To remove the color channels and make the image grayscale.", 
            "To scale large numbers (0-255) down to small decimals to prevent the network's mathematical calculations from becoming unstable.", 
            "To artificially increase the resolution of the image."
        ], 1, 
        "Normalization keeps numerical values small and stable. Unnormalized inputs cause numerical overflow during deep matrix multiplications.")

def chapter_2(model=None, preprocess=None, categories=None):
    st.header("Chapter 2: What Is a Neural Network?")
    st.markdown("### 2.1 The Architecture of a Network")
    st.markdown("""
    A Neural Network is an algorithm loosely inspired by biological brains. It consists of layers of artificial **neurons** (nodes).
    * **Inputs:** The raw data (our image tensor) fed into the first layer.
    * **Weights:** The connections between neurons. You can think of weights as the network's *memory* or *learned rules*. They are just numbers. If a weight is high, the connection is strong.
    * **Activations:** Mathematical gates that decide if a neuron should "fire" and pass data to the next layer based on the sum of its inputs and weights.
    """)
    st.markdown("#### Neural Network Playground")
    st.write(
        "Change the two inputs, hidden-layer size, or connection weights. "
        "The diagram and sigmoid activations update together."
    )
    playground_controls = st.columns([1, 1, 1])
    with playground_controls[0]:
        input_values = np.array(
            [
                st.slider("Input x₁", 0.0, 1.0, 0.4, 0.05, key="ch2_play_input_0"),
                st.slider("Input x₂", 0.0, 1.0, 0.8, 0.05, key="ch2_play_input_1"),
            ]
        )
    with playground_controls[1]:
        hidden_count = st.slider(
            "Hidden neurons", 2, 5, 3, key="ch2_play_hidden_count"
        )
    with playground_controls[2]:
        st.markdown("**Output neurons**")
        st.caption("Two sigmoid output scores")

    input_to_hidden = np.zeros((2, hidden_count))
    hidden_to_output = np.zeros((hidden_count, 2))
    weight_controls = st.columns(2)
    with weight_controls[0]:
        with st.expander("Adjust input → hidden weights", expanded=False):
            for hidden_index in range(hidden_count):
                cols = st.columns(2)
                for input_index in range(2):
                    input_to_hidden[input_index, hidden_index] = cols[
                        input_index
                    ].slider(
                        f"x{input_index + 1} → h{hidden_index + 1}",
                        -3.0,
                        3.0,
                        0.6 if input_index == hidden_index % 2 else -0.4,
                        0.1,
                        key=f"ch2_play_w_ih_{input_index}_{hidden_index}",
                    )
    with weight_controls[1]:
        with st.expander("Adjust hidden → output weights", expanded=False):
            for hidden_index in range(hidden_count):
                cols = st.columns(2)
                for output_index in range(2):
                    hidden_to_output[hidden_index, output_index] = cols[
                        output_index
                    ].slider(
                        f"h{hidden_index + 1} → y{output_index + 1}",
                        -3.0,
                        3.0,
                        0.7 if output_index == hidden_index % 2 else -0.3,
                        0.1,
                        key=f"ch2_play_w_ho_{hidden_index}_{output_index}",
                    )

    hidden_activations = _sigmoid(input_values @ input_to_hidden)
    output_activations = _sigmoid(hidden_activations @ hidden_to_output)
    st.write(
        "**Hidden activations:** "
        + " · ".join(f"h{i + 1} = {value:.3f}" for i, value in enumerate(hidden_activations))
    )
    st.write(
        "**Output activations:** "
        + " · ".join(f"y{i + 1} = {value:.3f}" for i, value in enumerate(output_activations))
    )

    network_figure = go.Figure()
    node_layers = [
        (0, input_values, "Input", "#38bdf8"),
        (1, hidden_activations, "Hidden", "#a78bfa"),
        (2, output_activations, "Output", "#34d399"),
    ]
    for input_index in range(2):
        for hidden_index in range(hidden_count):
            weight = input_to_hidden[input_index, hidden_index]
            network_figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[
                        -(input_index - 0.5),
                        (hidden_count - 1) / 2 - hidden_index,
                    ],
                    mode="lines+markers",
                    line={
                        "color": "#38bdf8" if weight >= 0 else "#fb7185",
                        "width": 1 + abs(weight) * 1.5,
                    },
                    marker={"size": [0, 0]},
                    text=[f"Weight: {weight:+.1f}"] * 2,
                    hovertemplate="%{text}<extra>Input → Hidden</extra>",
                    showlegend=False,
                )
            )
    for hidden_index in range(hidden_count):
        for output_index in range(2):
            weight = hidden_to_output[hidden_index, output_index]
            network_figure.add_trace(
                go.Scatter(
                    x=[1, 2],
                    y=[
                        (hidden_count - 1) / 2 - hidden_index,
                        -(output_index - 0.5),
                    ],
                    mode="lines+markers",
                    line={
                        "color": "#38bdf8" if weight >= 0 else "#fb7185",
                        "width": 1 + abs(weight) * 1.5,
                    },
                    marker={"size": [0, 0]},
                    text=[f"Weight: {weight:+.1f}"] * 2,
                    hovertemplate="%{text}<extra>Hidden → Output</extra>",
                    showlegend=False,
                )
            )
    for x_position, activations, layer_name, color in node_layers:
        layer_y = (
            [-(index - 0.5) for index in range(2)]
            if layer_name != "Hidden"
            else [(hidden_count - 1) / 2 - index for index in range(hidden_count)]
        )
        network_figure.add_trace(
            go.Scatter(
                x=[x_position] * len(activations),
                y=layer_y,
                mode="markers+text",
                text=[f"{layer_name[0]}{index + 1}: {value:.2f}" for index, value in enumerate(activations)],
                textposition="middle right",
                marker={
                    "size": 36,
                    "color": color,
                    "line": {"color": "#e5e7eb", "width": 1},
                },
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    network_figure.update_layout(
        height=430,
        margin={"l": 30, "r": 80, "t": 35, "b": 20},
        xaxis={
            "tickvals": [0, 1, 2],
            "ticktext": ["Input", "Hidden", "Output"],
            "range": [-0.35, 2.5],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"visible": False, "range": [-max(hidden_count, 2), max(hidden_count, 2)]},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 250},
        showlegend=False,
    )
    st.plotly_chart(network_figure, width="stretch", key="ch2_network_playground")

    render_knowledge_check(
        "ch2_q1",
        "In this network, what does a connection weight control?",
        [
            "How strongly one neuron's activation influences the next layer.",
            "The number of pixels in the input image.",
            "Whether the model downloads its weights.",
        ],
        0,
        "A weight scales a signal as it travels between neurons; changing it changes the next layer's weighted sum and activation.",
    )

    st.markdown("### 2.2 Training vs. Inference")
    st.markdown("""
    * **Training (The Backward Pass):** The network starts completely ignorant. We feed it an image of a dog, and it guesses randomly. We calculate how wrong it was (the *Loss*), and use calculus (*Backpropagation*) to update the **weights** so it guesses slightly better next time. This requires millions of images and massive GPU compute power.
    * **Inference (The Forward Pass):** Once the network is highly accurate, we "freeze" its weights. It stops learning. We give it new images and it applies its frozen weights to make a prediction. This requires much less computing power.
    """)
    st.markdown("#### Training vs. Inference Simulator")
    simulator_mode = st.radio(
        "Network mode",
        ["Training", "Inference (frozen weights)"],
        horizontal=True,
        key="ch2_simulator_mode",
    )
    st.caption(
        "A tiny two-input, one-output sigmoid neuron keeps each forward and backward "
        "calculation visible; its weights are separate from the pretrained model."
    )
    st.session_state.setdefault("ch2_sim_weights", [0.2, -0.2])
    st.session_state.setdefault("ch2_sim_bias", 0.0)
    sim_controls = st.columns(3)
    with sim_controls[0]:
        sim_inputs = np.array(
            [
                st.slider("Feature x₁", 0.0, 1.0, 0.8, 0.05, key="ch2_sim_x1"),
                st.slider("Feature x₂", 0.0, 1.0, 0.3, 0.05, key="ch2_sim_x2"),
            ]
        )
    with sim_controls[1]:
        learning_rate = st.slider(
            "Learning rate",
            0.05,
            1.0,
            0.3,
            0.05,
            disabled=simulator_mode != "Training",
            key="ch2_sim_learning_rate",
        )
        if simulator_mode == "Training":
            target = st.selectbox(
                "Target label",
                [0, 1],
                format_func=lambda value: f"Class {value}",
                key="ch2_sim_target",
            )
        else:
            st.caption("Target is hidden during inference.")
            target = None
    with sim_controls[2]:
        st.write("**Current weights**")
        st.write(
            " · ".join(
                f"w{i + 1} = {weight:+.3f}"
                for i, weight in enumerate(st.session_state.ch2_sim_weights)
            )
            + f" · b = {st.session_state.ch2_sim_bias:+.3f}"
        )
        if st.button("Reset demo weights", key="ch2_sim_reset"):
            st.session_state.ch2_sim_weights = [0.2, -0.2]
            st.session_state.ch2_sim_bias = 0.0
            st.rerun()

    sim_weights = np.asarray(st.session_state.ch2_sim_weights, dtype=float)
    sim_bias = float(st.session_state.ch2_sim_bias)
    sim_logit = float(sim_inputs @ sim_weights + sim_bias)
    sim_prediction = float(_sigmoid(sim_logit))
    if simulator_mode == "Training":
        sim_loss = float(_binary_cross_entropy(sim_prediction, target))
        st.write(
            f"**Forward pass:** sigmoid({sim_logit:.3f}) = **{sim_prediction:.3f}**"
            f" &nbsp; · &nbsp; **Binary cross-entropy loss:** `{sim_loss:.4f}`"
        )
        if st.button("Backpropagate and update weights", type="primary", key="ch2_sim_train"):
            gradients = (sim_prediction - target) * sim_inputs
            bias_gradient = sim_prediction - target
            updated_weights = sim_weights - learning_rate * gradients
            updated_bias = sim_bias - learning_rate * bias_gradient
            updated_prediction = float(_sigmoid(float(sim_inputs @ updated_weights + updated_bias)))
            updated_loss = float(_binary_cross_entropy(updated_prediction, target))
            st.session_state.ch2_sim_weights = updated_weights.tolist()
            st.session_state.ch2_sim_bias = float(updated_bias)
            st.session_state.ch2_sim_last_update = {
                "before_prediction": sim_prediction,
                "after_prediction": updated_prediction,
                "before_loss": sim_loss,
                "after_loss": updated_loss,
                "weight_gradients": gradients.tolist(),
                "bias_gradient": bias_gradient,
            }
            st.rerun()
        if "ch2_sim_last_update" in st.session_state:
            update = st.session_state.ch2_sim_last_update
            st.success(
                f"Backpropagation updated weights. Prediction: "
                f"{update['before_prediction']:.3f} → "
                f"{update['after_prediction']:.3f}; loss: "
                f"{update['before_loss']:.4f} → {update['after_loss']:.4f}."
            )
            st.caption(
                "Gradients: "
                + ", ".join(
                    f"∂L/∂w{i + 1} = {gradient:+.4f}"
                    for i, gradient in enumerate(update["weight_gradients"])
                )
                + f", ∂L/∂b = {update['bias_gradient']:+.4f}."
            )
    else:
        if st.button("Run forward pass (weights frozen)", key="ch2_sim_infer"):
            st.session_state.ch2_sim_last_inference = {
                "prediction": sim_prediction,
                "weights": sim_weights.tolist(),
                "bias": sim_bias,
                "inputs": sim_inputs.tolist(),
            }
        st.info(
            f"**Forward pass:** sigmoid({sim_logit:.3f}) = "
            f"**{sim_prediction:.3f}**. Weights and bias are frozen; "
            "backpropagation is disabled in inference mode."
        )
        if "ch2_sim_last_inference" in st.session_state:
            st.caption(
                f"Last inference prediction: "
                f"{st.session_state.ch2_sim_last_inference['prediction']:.3f}. "
                "No parameters were updated."
            )

    render_knowledge_check(
        "ch2_q2",
        "Which operation changes weights during training?",
        [
            "Running a forward pass in inference mode.",
            "Backpropagating the loss gradient and applying a weight update.",
            "Displaying the network diagram.",
        ],
        1,
        "Training computes the loss gradient with respect to each weight and updates parameters; inference only uses the current frozen weights.",
    )

    st.markdown("### 2.3 Pretrained Models")
    st.markdown("""
    In this application, we use a **pretrained model**. We did not train it ourselves. We downloaded the frozen weights from PyTorch. Because we are only doing *inference*, the model runs easily on a standard CPU.
    """)
    st.markdown("#### Pretrained Model Lab")
    if model is None or preprocess is None or categories is None:
        st.info(
            "The pretrained model is available from the main application. "
            "Open Chapter 2 from the running app to try image inference."
        )
    else:
        sample_dir = Path(__file__).resolve().parent.parent / "assets" / "images"
        image_options = {
            "Dog (original)": sample_dir / "dog.jpg",
            "Dog (masked)": sample_dir / "dog_masked_test.jpg",
        }
        available_images = {
            label: path for label, path in image_options.items() if path.is_file()
        }
        if not available_images:
            st.warning("No sample images were found in assets/images.")
        else:
            selected_sample = st.selectbox(
                "Sample image",
                list(available_images),
                key="ch2_pretrained_sample",
            )
            sample_path = available_images[selected_sample]
            with Image.open(sample_path) as sample_image:
                sample_image = sample_image.convert("RGB")
            st.image(
                sample_image,
                caption=selected_sample,
                width=280,
            )
            st.markdown(
                "**Pipeline:** `sample image` → `preprocessed tensor` "
                "→ `pretrained MobileNetV2` → `top predictions`"
            )
            if st.button(
                "Run pretrained inference",
                type="primary",
                key="ch2_pretrained_run",
            ):
                input_tensor = preprocess(sample_image).unsqueeze(0)
                with torch.inference_mode():
                    logits = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
                    top_probabilities, top_indices = torch.topk(probabilities, 5)
                st.session_state.ch2_pretrained_result = {
                    "sample": selected_sample,
                    "tensor_shape": tuple(input_tensor.shape),
                    "predictions": [
                        (
                            categories[int(index)],
                            float(probability),
                        )
                        for probability, index in zip(
                            top_probabilities.tolist(), top_indices.tolist()
                        )
                    ],
                }
            result = st.session_state.get("ch2_pretrained_result")
            if result and result["sample"] == selected_sample:
                st.caption(
                    f"Tensor shape: `{result['tensor_shape']}` "
                    "(batch, channels, height, width)."
                )
                prediction_columns = st.columns(5)
                for index, (label, confidence) in enumerate(result["predictions"]):
                    prediction_columns[index].metric(
                        f"Top {index + 1}",
                        label,
                        f"{confidence:.1%}",
                    )
                    prediction_columns[index].progress(confidence)

    render_knowledge_check(
        "ch2_q3",
        "What does a pretrained model bring to this inference lab?",
        [
            "Weights already learned from a large training dataset.",
            "A guarantee that every prediction is correct.",
            "A new set of weights trained whenever an image is uploaded.",
        ],
        0,
        "The pretrained MobileNetV2 starts with learned weights; this lab runs inference and does not train or update that model.",
    )

    render_knowledge_check("ch2_occlusion_q", "When you mask part of the image in the lab, is the network learning to recognize the grey box?",
        [
            "Yes, the network's weights are updating to learn about occlusion.", 
            "No, the weights are frozen. We are only running inference to see how the already-trained model reacts."
        ], 1, 
        "Our backend strictly uses `model.eval()` and `torch.inference_mode()`. No training or weight updating occurs in this application.")

def chapter_3():
    st.header("Chapter 3: Understanding CNNs")
    st.markdown("""
    ### 3.1 Why standard Neural Networks fail on images
    If you feed a 2D image into a standard neural network, it flattens the image into one giant 1D line of numbers. It completely destroys the **spatial relationships**—it forgets that a pixel was next to another pixel!
    
    ### 3.2 Convolution: The Sliding Window
    A **Convolutional Neural Network (CNN)** preserves 2D space. It slides a small grid (a **kernel** or **filter**, usually 3x3 pixels) across the image. 
    
    As the filter slides, it performs **Element-wise Multiplication and Summation**. The filter is mathematically tuned to react to specific patterns (vertical lines, corners, color gradients). The resulting output map is called a **Feature Map**.
    """)
    
    st.markdown("#### Interactive Demo: The Convolution Operation")
    col1, col2, col3 = st.columns(3)
    with col1:
        img_pattern = st.selectbox("1. Choose Image Pattern:", ["Vertical Line", "Horizontal Line"])
        img = np.array([[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0],[0,0,10,0,0]]) if img_pattern == "Vertical Line" else np.array([[0,0,0,0,0],[0,0,0,0,0],[10,10,10,10,10],[0,0,0,0,0],[0,0,0,0,0]])
        fig1 = px.imshow(img, color_continuous_scale="gray", text_auto=True, title="Input (5x5)")
        fig1.update_coloraxes(showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        filter_type = st.selectbox("2. Choose Filter:", ["Vertical Edge", "Horizontal Edge"])
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) if filter_type == "Vertical Edge" else np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        fig2 = px.imshow(kernel, color_continuous_scale="RdBu", text_auto=True, title="Kernel (3x3)")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.write("3. Calculate Feature Map")
        if st.button("Apply Convolution"):
            output = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    output[i, j] = np.sum(img[i:i+3, j:j+3] * kernel)
            fig3 = px.imshow(output, color_continuous_scale="Viridis", text_auto=True, title="Output Map")
            fig3.update_coloraxes(showscale=False)
            st.plotly_chart(fig3, use_container_width=True)
            
    st.markdown("""
    ### 3.3 Advanced Architecture: Padding, Stride, ReLU, and Pooling
    Real CNNs do more than just convolve:
    * **Stride:** How many pixels the filter jumps when it slides. A stride of 2 halves the image size.
    * **Padding:** Adding fake 0-pixels around the edge of the image so the 3x3 filter doesn't fall off the edges, preserving spatial dimensions.
    * **ReLU (Rectified Linear Unit):** An activation function that simply changes all negative numbers in the feature map to zero. This introduces *non-linearity*, allowing the network to learn complex shapes.
    * **Pooling:** A downsampling layer (like Max Pooling) that takes the highest number in a 2x2 grid and throws the rest away. This saves memory and makes the network "spatially invariant" (it recognizes a dog ear whether it's shifted slightly left or right).
    
    When you mask an image in our lab, you are mathematically erasing the textures that the CNN's filters are desperately looking for!
    """)
    render_knowledge_check("ch3_q1", "What happens if you apply a Horizontal Edge filter to an image that only contains a Vertical Line?", 
        ["The output feature map shows a massive positive reaction.", "The output feature map will be mostly zeros because the filter didn't find its mathematical pattern.", "The filter automatically rotates to find the line."], 1, 
        "Filters are mathematically hardcoded matrices. If the pattern isn't there, the dot-product sum is near zero.")

def chapter_4():
    st.header("Chapter 4: How Image Classification Works")
    st.markdown("### 4.1 From Feature Maps to Logits")
    st.markdown("""
    After the image passes through dozens of convolutional and pooling layers, the final feature maps are flattened and passed to a **Classification Head** (standard dense layers). 
    
    This final layer outputs a raw list of numbers called **logits**. Our model outputs exactly 1,000 logits—one for every object category it knows.
    """)
    st.markdown("#### Interactive Classification Pipeline")
    st.write(
        "Select each stage to follow a compact, calculated example from feature "
        "activations to a predicted class."
    )
    pipeline_controls = st.columns(3)
    with pipeline_controls[0]:
        channel_count = st.slider(
            "Feature-map channels", 1, 3, 2, key="ch4_pipeline_channels"
        )
    with pipeline_controls[1]:
        feature_side = st.slider(
            "Feature-map width/height", 2, 4, 2, key="ch4_pipeline_side"
        )
    with pipeline_controls[2]:
        feature_scale = st.slider(
            "Input feature strength", 0.2, 1.0, 0.6, 0.1, key="ch4_pipeline_scale"
        )

    feature_base = np.arange(
        1, channel_count * feature_side * feature_side + 1, dtype=np.float64
    ).reshape(channel_count, feature_side, feature_side)
    feature_maps = feature_base * feature_scale / feature_base.max()
    flattened = feature_maps.reshape(-1)
    classifier_weights = np.linspace(
        -0.8, 0.9, flattened.size * 3, dtype=np.float64
    ).reshape(flattened.size, 3)
    classifier_bias = np.array([-0.15, 0.05, 0.1], dtype=np.float64)
    pipeline_logits = flattened @ classifier_weights + classifier_bias
    pipeline_probabilities = _softmax(pipeline_logits)
    pipeline_classes = ["Cat", "Dog", "Car"]
    pipeline_prediction_index = int(np.argmax(pipeline_probabilities))
    pipeline_stages = [
        "Feature Maps",
        "Flatten",
        "Classification Head",
        "Logits",
        "Softmax",
        "Prediction",
    ]
    stage_descriptions = {
        "Feature Maps": (
            "Convolutional layers produce one spatial activation map per channel.",
            feature_maps,
        ),
        "Flatten": (
            "The channel and spatial axes are reshaped into one feature vector.",
            flattened,
        ),
        "Classification Head": (
            "A learned linear layer multiplies features by weights and adds bias.",
            classifier_weights,
        ),
        "Logits": (
            "The classification head emits one unnormalized score per class.",
            pipeline_logits,
        ),
        "Softmax": (
            "Softmax converts the three logits into probabilities that sum to 1.",
            pipeline_probabilities,
        ),
        "Prediction": (
            "The class with the highest softmax probability is selected.",
            np.array([pipeline_classes[pipeline_prediction_index]]),
        ),
    }
    pipeline_columns = st.columns(6)
    for stage_column, stage in zip(pipeline_columns, pipeline_stages):
        if stage_column.button(
            stage,
            key=f"ch4_pipeline_stage_{stage}",
            type=(
                "primary"
                if st.session_state.get("ch4_pipeline_selected", "Feature Maps") == stage
                else "secondary"
            ),
            width="stretch",
        ):
            st.session_state.ch4_pipeline_selected = stage
    selected_stage = st.session_state.get("ch4_pipeline_selected", "Feature Maps")
    stage_description, stage_values = stage_descriptions[selected_stage]
    stage_shape = "class label" if selected_stage == "Prediction" else str(stage_values.shape)
    stage_preview = (
        str(stage_values[0])
        if selected_stage == "Prediction"
        else np.array2string(stage_values, precision=3, threshold=8, edgeitems=3)
    )
    st.markdown(f"**{selected_stage}** · tensor shape `{stage_shape}`")
    st.write(stage_description)
    st.code(stage_preview, language="text")
    st.progress(
        (pipeline_stages.index(selected_stage) + 1) / len(pipeline_stages),
        text=(
            f"Pipeline stage {pipeline_stages.index(selected_stage) + 1} "
            f"of {len(pipeline_stages)}"
        ),
    )
    render_knowledge_check(
        "ch4_pipeline_q",
        "What happens when feature maps are flattened?",
        [
            "Their values are reshaped into a vector for the classification head.",
            "All activations are converted to class probabilities.",
            "The model updates its learned weights.",
        ],
        0,
        "Flattening preserves activation values while arranging them in one vector that can be passed to a dense classification head.",
    )

    st.markdown("### 4.2 The Softmax Function")
    st.markdown("""
    Logits are just unbounded numbers (e.g., `14.5` or `-2.1`). To make them readable, we pass them through the **Softmax** formula:
    
    $$\sigma(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^K e^{z_j}}$$
    
    Softmax does two vital things:
    1. It squashes all 1,000 numbers so they fall between `0.0` and `1.0`.
    2. It forces all 1,000 numbers to add up to exactly `1.0` (or 100%).
    
    These are our **Probability Scores**. We usually only care about the **Top-K** (e.g., Top 3 highest scores).
    """)
    st.markdown("#### Softmax Playground")
    st.write("Adjust each raw logit and see how the probability distribution responds.")
    class_names = ["Cat", "Dog", "Car"]
    logits = []
    logit_columns = st.columns(3)
    for column, class_name, initial in zip(
        logit_columns, class_names, [1.2, 0.4, -0.3]
    ):
        logits.append(
            column.slider(
                f"{class_name} logit",
                -5.0,
                5.0,
                initial,
                0.1,
                key=f"ch4_logit_{class_name.lower()}",
            )
        )
    probabilities = _softmax(logits)
    prediction_index = int(np.argmax(probabilities))
    probability_percentages = probabilities * 100
    st.caption(
        f"Probability total: **{probability_percentages.sum():.2f}%** "
        "(displayed total may differ slightly due to rounding)."
    )
    probability_figure = go.Figure(
        go.Bar(
            x=probability_percentages,
            y=class_names,
            orientation="h",
            marker={
                "color": [
                    "#34d399" if index == prediction_index else "#38bdf8"
                    for index in range(len(class_names))
                ],
                "line": {
                    "color": [
                        "#facc15" if index == prediction_index else "#334155"
                        for index in range(len(class_names))
                    ],
                    "width": [
                        3 if index == prediction_index else 1
                        for index in range(len(class_names))
                    ],
                },
            },
            text=[f"{value:.2f}%" for value in probability_percentages],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
        )
    )
    probability_figure.update_layout(
        xaxis={"title": "Probability", "range": [0, 110], "ticksuffix": "%"},
        yaxis={"title": None, "autorange": "reversed"},
        height=300,
        margin={"l": 10, "r": 60, "t": 15, "b": 40},
        transition={"duration": 350},
        uirevision="ch4-softmax",
    )
    st.plotly_chart(
        probability_figure,
        width="stretch",
        key="ch4_softmax_probabilities",
    )
    st.success(
        f"Predicted class: **{class_names[prediction_index]}** "
        f"({probability_percentages[prediction_index]:.2f}%)"
    )
    st.markdown("#### Top-K Prediction Explorer")
    top_k_mode = st.radio(
        "Show predictions",
        ["Top-1", "Top-3"],
        horizontal=True,
        key="ch4_top_k_mode",
    )
    ranked_indices = np.argsort(probabilities)[::-1][
        : 1 if top_k_mode == "Top-1" else 3
    ]
    prediction_columns = st.columns(len(ranked_indices))
    for rank, (column, class_index) in enumerate(
        zip(prediction_columns, ranked_indices), start=1
    ):
        column.metric(
            f"Top {rank}: {class_names[class_index]}",
            f"{probability_percentages[class_index]:.2f}%",
            "Predicted" if rank == 1 else None,
        )
        column.progress(
            float(probabilities[class_index]),
            text=f"Confidence {probability_percentages[class_index]:.2f}%",
        )
    render_knowledge_check(
        "ch4_softmax_q",
        "What does softmax guarantee about the probabilities?",
        [
            "They sum to 100%, but the top class is not guaranteed to be correct.",
            "The top class is always the true class.",
            "Every class receives the same probability.",
        ],
        0,
        "Softmax creates a distribution summing to 100%; it does not establish whether a prediction is correct.",
    )

    st.markdown("### 4.3 Confidence vs. Correctness")
    st.markdown("""
    If the model outputs 99% for 'Samoyed', it means the model's math is highly *confident*. **Confidence is not accuracy.** If you feed the model a picture of a spaceship (which it was never trained on), it might output 99% confidence for "Car" simply because it has wheels!
    """)
    st.markdown("#### Confidence vs. Correctness Challenge")
    challenge_examples = {
        "A dog photo predicted as “Dog”": {
            "prediction": "Dog",
            "confidence": 0.72,
            "truth": "Dog",
            "explanation": "This prediction happens to match the image label. The 72% score describes model confidence; correctness comes from comparing against the actual label.",
        },
        "A spaceship photo predicted as “Car”": {
            "prediction": "Car",
            "confidence": 0.99,
            "truth": "Spaceship",
            "explanation": "The model is highly confident but wrong: confidence is the model's score, not proof that the class is correct.",
        },
        "A cat photo predicted as “Dog”": {
            "prediction": "Dog",
            "confidence": 0.54,
            "truth": "Cat",
            "explanation": "The prediction does not match the cat image label. A confidence percentage alone cannot tell us whether it is correct.",
        },
    }
    challenge_case = st.selectbox(
        "Choose a sample prediction",
        list(challenge_examples),
        key="ch4_challenge_case",
    )
    example = challenge_examples[challenge_case]
    st.info(
        f"Model prediction: **{example['prediction']}** "
        f"({example['confidence']:.0%} confidence). Is it correct?"
    )
    answer = st.radio(
        "Your answer",
        ["Correct", "Incorrect"],
        horizontal=True,
        index=None,
        key="ch4_challenge_answer",
    )
    if st.button("Check answer", key="ch4_challenge_submit", disabled=answer is None):
        is_correct = example["prediction"] == example["truth"]
        expected_answer = "Correct" if is_correct else "Incorrect"
        if answer == expected_answer:
            st.success("That's right! " + example["explanation"])
        else:
            st.error(
                f"Not quite—the correct answer is **{expected_answer}**. "
                + example["explanation"]
            )

    render_knowledge_check("ch4_q1", "What does the Softmax function guarantee about the output scores?", 
        ["It guarantees the top prediction is factually correct.", "It guarantees all the output probabilities sum to exactly 100%.", "It removes all predictions below 50%."], 1, 
        "Softmax turns raw logits into a standardized probability distribution summing to 1.0.")

def _render_mobilenet_architecture_explorer(model):
    st.markdown("#### Interactive Architecture Explorer")
    if model is None or not hasattr(model, "features"):
        st.info("The shared pretrained MobileNetV2 is not available in this view.")
        return

    modules = dict(model.named_modules())
    feature_layers = [f"features.{index}" for index in range(len(model.features))]
    layer_names = feature_layers + (["classifier"] if "classifier" in modules else [])
    layer_labels = {
        name: (
            f"Feature stage {name.split('.')[-1]}"
            if name.startswith("features.")
            else "Classifier"
        )
        for name in layer_names
    }
    selected_layer = st.selectbox(
        "Select a MobileNetV2 stage",
        layer_names,
        format_func=lambda name: f"{layer_labels[name]} · {name}",
        key="ch5_arch_layer",
    )

    layer = modules[selected_layer]
    details = []
    for attribute, label in (
        ("in_channels", "Input channels"),
        ("out_channels", "Output channels"),
        ("kernel_size", "Kernel"),
        ("stride", "Stride"),
        ("groups", "Groups"),
    ):
        value = getattr(layer, attribute, None)
        if value is not None:
            details.append(f"**{label}:** `{value}`")
    if details:
        st.caption(" · ".join(details))
    st.caption(
        "Inspect runs one zero-image probe through the existing model to capture "
        "the selected layer's real output shape and activations."
    )
    if st.button(
        "Inspect selected stage",
        type="primary",
        key="ch5_arch_inspect",
    ):
        captured = {}

        def capture_output(_module, inputs, output):
            activation = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(activation, torch.Tensor):
                raise TypeError(
                    f"{selected_layer} returned {type(activation).__name__}, "
                    "not a tensor."
                )
            captured["input_shape"] = tuple(inputs[0].shape)
            captured["output_shape"] = tuple(activation.shape)
            values = activation.detach()
            if values.ndim == 4:
                captured["feature_map"] = (
                    values[0, 0, :32, :32].float().cpu().numpy()
                )
                captured["channel_means"] = (
                    values[0].float().mean(dim=(-2, -1))[:32].cpu().numpy()
                )
            else:
                captured["feature_map"] = None
                captured["channel_means"] = values.flatten()[:32].float().cpu().numpy()

        hook = layer.register_forward_hook(capture_output)
        parameter = next(model.parameters())
        try:
            probe = torch.zeros(
                (1, 3, 224, 224),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            with torch.inference_mode():
                model(probe)
        finally:
            hook.remove()
        captured["layer"] = selected_layer
        st.session_state["ch5_arch_capture"] = captured

    captured = st.session_state.get("ch5_arch_capture")
    if captured and captured["layer"] == selected_layer:
        st.write(
            f"Probe input: `{captured['input_shape']}` → "
            f"selected output: `{captured['output_shape']}` "
            "(batch, channels, height, width where applicable)."
        )
        map_data = captured["feature_map"]
        if map_data is not None:
            map_column, channel_column = st.columns((3, 2))
            with map_column:
                st.caption("First output channel (cropped to at most 32×32)")
                st.plotly_chart(
                    px.imshow(
                        map_data,
                        color_continuous_scale="Viridis",
                        labels={"x": "Column", "y": "Row", "color": "Activation"},
                    ),
                    width="stretch",
                    key="ch5_arch_feature_map",
                )
            with channel_column:
                st.caption("Mean activation for the first output channels")
                st.bar_chart(
                    pd.DataFrame(
                        {"Mean activation": captured["channel_means"]},
                        index=[
                            f"Channel {index}"
                            for index in range(len(captured["channel_means"]))
                        ],
                    )
                )


def _render_convolution_simulator():
    st.markdown("#### Depthwise vs. Standard Convolution Simulator")
    control_columns = st.columns(4)
    with control_columns[0]:
        input_channels = st.slider(
            "Input channels", 1, 64, 16, key="ch5_conv_input_channels"
        )
    with control_columns[1]:
        output_channels = st.slider(
            "Output channels", 1, 128, 32, key="ch5_conv_output_channels"
        )
    with control_columns[2]:
        kernel_size = st.select_slider(
            "Kernel size", options=[1, 3, 5, 7], value=3, key="ch5_conv_kernel"
        )
    with control_columns[3]:
        image_side = st.select_slider(
            "Feature-map side", options=[14, 28, 56, 112], value=28,
            key="ch5_conv_image_side",
        )

    locations = image_side * image_side
    standard_parameters = kernel_size**2 * input_channels * output_channels
    depthwise_parameters = kernel_size**2 * input_channels
    pointwise_parameters = input_channels * output_channels
    standard_macs = locations * standard_parameters
    depthwise_macs = locations * depthwise_parameters
    pointwise_macs = locations * pointwise_parameters
    separable_parameters = depthwise_parameters + pointwise_parameters
    separable_macs = depthwise_macs + pointwise_macs

    metrics = st.columns(3)
    metrics[0].metric("Standard convolution parameters", f"{standard_parameters:,}")
    metrics[1].metric(
        "Depthwise + pointwise parameters", f"{separable_parameters:,}"
    )
    metrics[2].metric(
        "Parameter reduction",
        f"{(1 - separable_parameters / standard_parameters):.1%}"
        if standard_parameters
        else "0%",
    )
    st.write(
        f"At **{image_side}×{image_side}** output locations, standard convolution "
        f"uses **{standard_macs:,} MACs**; depthwise + pointwise uses "
        f"**{separable_macs:,} MACs** "
        f"({(1 - separable_macs / standard_macs):.1%} fewer)."
    )
    st.caption(
        "Theoretical multiply-accumulate (MAC) counts assume stride 1, same-size "
        "output, no bias, and one output channel count as configured. These are "
        "formula-based operation counts, not measured latency."
    )
    st.code(
        "Standard: K² × Cin × Cout\n"
        "Depthwise: K² × Cin\n"
        "Pointwise: Cin × Cout\n"
        "MACs: output locations × parameters",
        language="text",
    )
    stage = st.select_slider(
        "Convolution animation stage",
        options=[
            "Input",
            "Standard convolution",
            "Depthwise convolution",
            "Pointwise channel mixing",
            "Output",
        ],
        value="Input",
        key="ch5_conv_stage",
    )
    stage_details = {
        "Input": (
            f"{image_side}×{image_side}×{input_channels}",
            "Feature map entering the layer",
        ),
        "Standard convolution": (
            f"{standard_parameters:,} parameters",
            f"{standard_macs:,} MACs · spatial and channel mixing together",
        ),
        "Depthwise convolution": (
            f"{depthwise_parameters:,} parameters",
            f"{depthwise_macs:,} MACs · one spatial filter per input channel",
        ),
        "Pointwise channel mixing": (
            f"{pointwise_parameters:,} parameters",
            f"{pointwise_macs:,} MACs · 1×1 filters mix channels",
        ),
        "Output": (
            f"{image_side}×{image_side}×{output_channels}",
            "Feature map leaving the layer",
        ),
    }
    stage_columns = st.columns(5)
    for column, (label, (shape_or_count, description)) in zip(
        stage_columns, stage_details.items()
    ):
        active_class = "ch5-flow-active" if stage == label else ""
        column.markdown(
            f"<div class='ch5-flow-stage {active_class}'>"
            f"<b>{label}</b><br>{shape_or_count}<br>{description}</div>",
            unsafe_allow_html=True,
        )


def _render_inverted_residual_visualizer():
    st.markdown("#### Inverted Residual Block Visualizer")
    controls = st.columns(4)
    with controls[0]:
        input_channels = st.slider(
            "Block input channels", 8, 128, 32, step=8,
            key="ch5_block_input_channels",
        )
    with controls[1]:
        output_channels = st.slider(
            "Block output channels", 8, 128, 32, step=8,
            key="ch5_block_output_channels",
        )
    with controls[2]:
        expansion = st.select_slider(
            "Expansion factor", options=[1, 2, 3, 4, 6], value=6,
            key="ch5_block_expansion",
        )
    with controls[3]:
        spatial_side = st.select_slider(
            "Input feature-map side", options=[14, 28, 56], value=28,
            key="ch5_block_spatial",
        )

    expanded_channels = input_channels * expansion
    residual = input_channels == output_channels
    stages = [
        ("Expansion (1×1)", input_channels, expanded_channels),
        ("Depthwise (3×3)", expanded_channels, expanded_channels),
        ("Linear projection (1×1)", expanded_channels, output_channels),
    ]
    selected_stage = st.select_slider(
        "Step through the block",
        options=[stage[0] for stage in stages] + ["Residual add"],
        value="Expansion (1×1)",
        key="ch5_block_stage",
    )
    active_channels = input_channels
    if selected_stage != "Residual add":
        _, _, active_channels = next(
            stage for stage in stages if stage[0] == selected_stage
        )
    else:
        active_channels = output_channels

    stage_columns = st.columns(4)
    for index, (label, channels_in, channels_out) in enumerate(stages):
        active = selected_stage == label
        stage_columns[index].markdown(
            f"<div class='ch5-flow-stage {'ch5-flow-active' if active else ''}'>"
            f"<b>{label}</b><br>{spatial_side}×{spatial_side}×{channels_in}"
            f" → {spatial_side}×{spatial_side}×{channels_out}</div>",
            unsafe_allow_html=True,
        )
    stage_columns[3].markdown(
        "<div class='ch5-flow-stage "
        + ("ch5-flow-active" if selected_stage == "Residual add" else "")
        + "'><b>Residual add</b><br>"
        + (
            f"Input + projection → {spatial_side}×{spatial_side}×{output_channels}"
            if residual
            else "Skipped: input and output shapes differ"
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Selected stage operates on **{spatial_side}×{spatial_side}×"
        f"{active_channels}** activations. The hidden expansion is "
        f"{input_channels}×{expansion} = {expanded_channels} channels. "
        + (
            "A residual add is shape-compatible because input and output channels match."
            if residual
            else "No residual add: this block's input and output channel counts differ."
        )
    )


def _render_mobilenet_inference_lab(model, preprocess, categories):
    st.markdown("#### ImageNet Classification Lab")
    if model is None or preprocess is None or categories is None:
        st.warning(
            "ImageNet inference needs the shared pretrained MobileNetV2, its "
            "preprocessing transform, and ImageNet categories."
        )
        return

    sample_directory = Path(__file__).resolve().parent.parent / "assets" / "images"
    sample_options = {
        label: sample_directory / filename
        for label, filename in (
            ("Dog", "dog.jpg"),
            ("Masked dog", "dog_masked_test.jpg"),
        )
        if (sample_directory / filename).is_file()
    }
    if not sample_options:
        st.warning("No sample images were found in assets/images.")
        return

    source_columns = st.columns((2, 3))
    with source_columns[0]:
        selected_sample = st.selectbox(
            "Sample image",
            list(sample_options),
            key="ch5_inference_sample",
        )
        uploaded_file = st.file_uploader(
            "Or upload an image",
            type=["png", "jpg", "jpeg", "webp"],
            key="ch5_inference_upload",
        )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB")
        image_label = uploaded_file.name
    else:
        image_path = sample_options[selected_sample]
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as sample_image:
            image = sample_image.convert("RGB")
        image_label = selected_sample

    with source_columns[1]:
        st.image(image, caption=image_label, width="stretch")
        st.markdown(
            "**Pipeline:** image → pretrained transform → "
            "MobileNetV2 → ImageNet class scores"
        )
        st.caption(
            f"Original image: {image.width}×{image.height} RGB. "
            "The shared pretrained model is used without training or weight updates."
        )

    import hashlib

    image_key = hashlib.sha256(image_bytes).hexdigest()
    if st.button(
        "Run MobileNetV2 inference",
        type="primary",
        key="ch5_inference_run",
    ):
        input_tensor = preprocess(image).unsqueeze(0)
        parameter = next(model.parameters())
        input_tensor = input_tensor.to(device=parameter.device, dtype=parameter.dtype)
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            top_probabilities, top_indices = torch.topk(probabilities, 5)
        st.session_state["ch5_inference_result"] = {
            "image_key": image_key,
            "tensor_shape": tuple(input_tensor.shape),
            "predictions": [
                (categories[int(index)], float(probability))
                for probability, index in zip(
                    top_probabilities.tolist(), top_indices.tolist()
                )
            ],
        }

    result = st.session_state.get("ch5_inference_result")
    if result and result["image_key"] == image_key:
        st.caption(
            f"Preprocessed tensor shape: `{result['tensor_shape']}` "
            "(batch, channels, height, width)."
        )
        for rank, (label, confidence) in enumerate(result["predictions"], start=1):
            prediction_columns = st.columns((1, 4))
            prediction_columns[0].write(f"**{rank}. {label}**")
            prediction_columns[1].progress(confidence, text=f"{confidence:.2%}")


def _render_model_comparison(model):
    st.markdown("#### Lightweight Model Comparison")
    from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights

    mobile_parameters = (
        sum(parameter.numel() for parameter in model.parameters())
        if model is not None
        else MobileNet_V2_Weights.DEFAULT.meta["num_params"]
    )
    resnet_parameters = ResNet50_Weights.DEFAULT.meta["num_params"]
    comparison = pd.DataFrame(
        [
            {
                "Model": "MobileNetV2",
                "Parameters": mobile_parameters,
                "Published ops (GFLOPs)": MobileNet_V2_Weights.DEFAULT.meta["_ops"],
                "Published checkpoint (MB)": MobileNet_V2_Weights.DEFAULT.meta[
                    "_file_size"
                ],
            },
            {
                "Model": "ResNet50",
                "Parameters": resnet_parameters,
                "Published ops (GFLOPs)": ResNet50_Weights.DEFAULT.meta["_ops"],
                "Published checkpoint (MB)": ResNet50_Weights.DEFAULT.meta[
                    "_file_size"
                ],
            },
        ]
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        "Parameter counts are read from the live shared MobileNetV2 and "
        "Torchvision's ResNet50 metadata. GFLOPs and checkpoint sizes are "
        "published Torchvision weight metadata (224×224 input), not measurements "
        "from this session."
    )

    if model is None:
        st.info("Local latency comparison needs the shared MobileNetV2.")
        return
    if st.button(
        "Measure local inference latency",
        key="ch5_latency_run",
    ):
        import time
        from torchvision.models import resnet50

        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        probe = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
        resnet = resnet50(weights=None).to(device=device, dtype=dtype).eval()
        model.eval()

        def measure_latency(target_model):
            with torch.inference_mode():
                target_model(probe)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations = []
                for _ in range(3):
                    started = time.perf_counter()
                    target_model(probe)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    durations.append((time.perf_counter() - started) * 1000)
            return float(np.median(durations))

        latency_results = {
            "MobileNetV2": measure_latency(model),
            "ResNet50 (untrained weights)": measure_latency(resnet),
            "device": str(device),
        }
        st.session_state["ch5_latency_results"] = latency_results
        del resnet

    latency_results = st.session_state.get("ch5_latency_results")
    if latency_results:
        measured_columns = st.columns(2)
        measured_columns[0].metric(
            "MobileNetV2 measured median",
            f"{latency_results['MobileNetV2']:.2f} ms",
        )
        measured_columns[1].metric(
            "ResNet50 measured median",
            f"{latency_results['ResNet50 (untrained weights)']:.2f} ms",
        )
        st.caption(
            f"Measured locally on {latency_results['device']} with batch size 1, "
            "224×224 input, one warm-up pass, and the median of three timed "
            "forward passes. ResNet50 uses untrained weights; this compares "
            "execution cost, not prediction quality. Latency depends on hardware "
            "and system load."
        )


def chapter_5(model=None, preprocess=None, categories=None):
    st.header("Chapter 5: Meet MobileNetV2")
    st.markdown(
        """
        <style>
        .ch5-flow-stage {
            min-height: 76px;
            padding: 12px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 10px;
            background: rgba(30, 41, 59, 0.58);
            transition: border-color 180ms ease, background-color 180ms ease,
                        transform 180ms ease;
        }
        .ch5-flow-active {
            border-color: #38bdf8;
            background: rgba(14, 116, 144, 0.28);
            transform: translateY(-2px);
        }
        @media (max-width: 720px) {
            .ch5-flow-stage { min-height: 64px; padding: 9px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
    ### 5.1 Why MobileNetV2?
    Many deep learning models (like ResNet or VGG) are massive, requiring gigabytes of RAM and powerful GPUs to run. **MobileNetV2** was specifically designed by Google to run on mobile phones and lightweight CPUs (like the one running this application).
    """)
    _render_mobilenet_architecture_explorer(model)
    _render_model_comparison(model)
    _render_inverted_residual_visualizer()
    render_knowledge_check(
        "ch5_q_architecture",
        "Why does an inverted residual block expand channels before depthwise filtering?",
        [
            "To create a higher-dimensional space for efficient feature filtering.",
            "To make every feature map larger in width and height.",
            "To remove the need for learned weights.",
        ],
        0,
        "The expansion layer temporarily increases channels so depthwise filtering can process richer features before a compact linear projection.",
    )

    st.markdown("""
    ### 5.2 Depthwise Separable Convolutions
    Standard convolution is computationally expensive because it mixes spatial data (pixels) and channel data (RGB) all at once. MobileNetV2 uses a trick:
    1. **Depthwise Convolution:** It filters each color channel separately.
    2. **Pointwise Convolution:** It then uses a 1x1 filter to mix the channels together.
    This drastic reduction in multiplication operations is what makes the model so fast!
    """)
    _render_convolution_simulator()
    render_knowledge_check(
        "ch5_q1",
        "Why is MobileNetV2 able to run so fast on a standard CPU?",
        [
            "It uses low-resolution black and white images only.",
            "It uses Depthwise Separable Convolutions to drastically reduce the number of mathematical operations.",
            "It only knows 10 object classes instead of 1000.",
        ],
        1,
        "By separating spatial filtering from channel mixing, MobileNetV2 achieves high accuracy with a fraction of the compute cost.",
    )

    st.markdown("""
    ### 5.3 The ImageNet Dataset
    MobileNetV2 was trained on **ImageNet**, a famous dataset containing over 1.2 million images mapped to exactly 1,000 specific classes (like 'Samoyed', 'Keyboard', or 'Tench fish'). Our model can *only* predict these 1,000 things. If you upload a picture of a Pikachu, the model physically cannot output "Pikachu"—it will guess whatever ImageNet class looks most similar.
    """)
    _render_mobilenet_inference_lab(model, preprocess, categories)
    render_knowledge_check(
        "ch5_q_imagenet",
        "What happens if you provide a pretrained ImageNet model with an object outside its 1,000 classes?",
        [
            "It adds a new class automatically.",
            "It returns the closest-scoring class from the classes it learned.",
            "It retrains itself on the uploaded image.",
        ],
        1,
        "The classifier can only score the 1,000 classes in its ImageNet label set; uploading an image does not add classes or retrain the model.",
    )

def chapter_6():
    st.header("Chapter 6: How Masking Changes a Prediction")
    st.markdown("""
    ### 6.1 The Image Occlusion Experiment
    If a neural network relies on a specific visual feature (like a dog's snout) to make a prediction, what happens if we delete that feature?
    
    By drawing a grey box over a region of the image (**occlusion**), we mathematically overwrite that information. When we run the modified image through the model again, the Probability Score shifts.
    
    * If the score **drops heavily**, the masked region contained critical information.
    * If the score **increases**, the region actually contained distracting noise!
    
    ### 6.2 The Artifact Problem (Out of Distribution)
    There is a major catch to our experiment. When we draw a sharp grey square on an image, we introduce artificial, straight edges that weren't there before. 
    
    Because the model was never trained on images with floating grey boxes, this is an **Out-of-Distribution** input. We must be careful: did the score drop because we hid the dog, or because the model's filters got completely confused by the unnatural straight grey lines?
    """)
    render_knowledge_check("ch6_q1", "If you mask a region and the model's confidence goes UP, what does this imply?", 
        ["The model broke.", "The masked region likely contained distracting background noise, so removing it helped the model focus.", "The masked region contained the dog's face."], 1, 
        "Removing noisy backgrounds (like complex grass textures) can sometimes make the core pattern easier for the CNN to classify.")

def chapter_7():
    st.header("Chapter 7: Understanding the AI X-Ray (Heatmap)")
    st.markdown("""
    ### 7.1 The Occlusion Sensitivity Formula
    In the Lab, when you click "Generate Heatmap", the computer runs a specific mathematical formula to build the **AI X-Ray**.
    
    For a selected target class $c$ (like 'Samoyed'):
    1. $S_{original}(c) =$ The probability score of the unedited image.
    2. $S_{masked}(c, i) =$ The probability score when grid region $i$ is covered.
    
    The **Signed Sensitivity Delta (D)** for that region is calculated as:
    $$D(i) = S_{original}(c) - S_{masked}(c, i)$$
    
    ### 7.2 Interpreting the Math
    * You mask the dog's face. The score drops from 80% to 35%. 
      $D = 0.80 - 0.35 = 0.45$ (A large positive effect).
    * You mask the grass. The score rises from 80% to 82%. 
      $D = 0.80 - 0.82 = -0.02$ (A slight negative effect).
      
    ### 7.3 Absolute Effect
    Sometimes we only care about *magnitude*—how much did the region matter, regardless of whether it helped or hurt? We calculate the **Absolute Effect (A)**:          
    **A(i) = | D(i) |**
    
    We map these numbers to colors (Red for high drop, Blue for negative drop) to create the Heatmap matrix.
    """)
    render_knowledge_check("ch7_q1", "In our formula, what does a large positive D(i) value mean?", 
        ["Masking that region helped the model.", "Masking that region hurt the model's confidence significantly.", "The region is completely black."], 1, 
        "A positive D(i) means Original - Masked was a positive number, meaning the Masked score was much lower.")

def chapter_8():
    st.header("Chapter 8: Conducting Valid Experiments")
    st.markdown("""
    ### 8.1 The Scientific Method in the Lab
    When you go to the Interactive Lab, you are not just playing a game; you are conducting empirical experiments on a black-box AI.
    
    1. **Form a Hypothesis:** Before clicking, make a prediction. *"I hypothesize that masking the dog's ear will reduce the score more than masking its tail."*
    2. **Control the Variables:** Only change one grid cell at a time. If you mask the whole image at once, you can't isolate which feature caused the score to drop.
    3. **Observe & Record:** The Experiment Lab table at the bottom of the app will automatically log the independent variable (the cell you clicked) and the dependent variable (the new score).
    4. **Conclude Cautiously:** Look at the data before deciding if your hypothesis was right.
    """)
    render_knowledge_check("ch8_q1", "Why is it important to test one grid cell at a time when forming a hypothesis?", 
        ["Because the computer will crash if you click two.", "To isolate variables; changing multiple cells makes it impossible to know which specific region caused the score change.", "It's a rule to make the game harder."], 1, 
        "Controlling variables is fundamental to science. You must isolate the input to understand the output.")

def chapter_9():
    st.header("Chapter 9: Limitations, Bias & Responsible Interpretation")
    st.markdown("""
    ### 9.1 The Illusion of Understanding
    A heatmap is incredibly useful, but it has severe limitations. **It does not prove human reasoning.** 
    
    If the heatmap highlights a dog's face, it does not mean the CNN "understands" what a face is. It just means the raw pixel values in that area heavily activated the math in the deeper layers.
    
    ### 9.2 Dataset Bias (Spurious Correlations)
    Imagine a network was trained on thousands of images where cows are always standing on green grass. The network might learn that "Green Grass = Cow". 
    
    If you show it a cow on a beach, it might fail. In the lab, masking the grass might drastically lower the score, even though grass is not part of a cow! The model learned the *context*, not just the object.
    
    You must interpret heatmaps responsibly. They show you what the math relies on, not what the AI "thinks."
    """)
    render_knowledge_check("ch9_q1", "If the AI X-Ray is entirely red over a patch of snow behind a Husky, what does this prove?", 
        ["The model thinks snow is a dog.", "The model relies heavily on the background context (snow) to classify the image as a Husky.", "The heatmap calculation is broken."], 1, 
        "This is a classic dataset bias! If Huskies are usually photographed in snow, the network learns a spurious correlation: 'White Snow = High probability of Husky'.")

def chapter_10():
    st.header("Chapter 10: Putting It All Together")
    st.markdown("""
    ### The CNN Blind Spot Pipeline
    You are now ready for the lab. Here is exactly what happens under the hood when you use it:
    
    1. **Input:** You view the standardized image.
    2. **Preprocess:** The app resizes and normalizes the image tensor.
    3. **Baseline Inference:** MobileNetV2 calculates the original Softmax probabilities.
    4. **Interact:** You click a grid button to mask a region.
    5. **Re-evaluate:** The newly masked image goes through the network.
    6. **X-Ray Math:** The app calculates the sensitivity delta $D(i)$ for all regions by running inference 16 times in the background.
    7. **Visualize:** Plotly renders the results as a color-coded Heatmap.
    8. **Log:** Your actions and score changes are permanently recorded in the Experiment Lab CSV.
    """)
    st.success("You have finished the core curriculum! You may proceed to the Interactive Lab via the sidebar.")

def render_academy_page(model=None, preprocess=None, categories=None):
    st.title("Learning Academy")
    st.markdown("A structured, interactive textbook to master Deep Learning concepts.")
    
    chapters = {
        "1. How a Computer Sees an Image": chapter_1,
        "2. What Is a Neural Network?": lambda: chapter_2(
            model=model,
            preprocess=preprocess,
            categories=categories,
        ),
        "3. Understanding CNNs": chapter_3,
        "4. Image Classification": chapter_4,
        "5. Meet MobileNetV2": lambda: chapter_5(
            model=model,
            preprocess=preprocess,
            categories=categories,
        ),
        "6. Masking & Predictions": chapter_6,
        "7. The AI X-Ray Math": chapter_7,
        "8. Conducting Valid Experiments": chapter_8,
        "9. Limitations & Bias": chapter_9,
        "10. The Full Pipeline": chapter_10
    }
    
    selection = st.sidebar.radio("Select Chapter:", list(chapters.keys()))
    st.divider()
    chapters[selection]()
