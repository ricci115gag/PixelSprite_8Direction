from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch

from pixel_sprite.config import load_config, resolve_project_path
from pixel_sprite.controller.infer import DIRECTIONS, load_model, predict_file


def create_app():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try loading the latest V2 model initially, but handle cases where checkpoints are empty
    try:
        model = load_model(config, device, version=2)
        initial_status = "Default (Latest V2) model loaded successfully."
        current_ts = "latest"
        initial_version = 2
    except Exception as e_v2:
        # Fallback to V1 if V2 has no checkpoints yet
        try:
            model = load_model(config, device, version=1)
            initial_status = "V2 has no checkpoints. Loaded latest V1 model successfully."
            current_ts = "latest"
            initial_version = 1
        except Exception as e_v1:
            model = None
            initial_status = f"No model loaded. V1 error: {e_v1}. V2 error: {e_v2}"
            current_ts = "none"
            initial_version = 2

    model_cache = {
        "config": config,
        "device": device,
        "model": model,
        "status": initial_status,
        "current_timestamp": current_ts,
        "current_version": initial_version
    }

    def get_model_choices(version: int) -> list[str]:
        checkpoints_path = config["misc"].get("checkpoints_path", "./checkpoints")
        checkpoints_dir = resolve_project_path(checkpoints_path) / f"v{version}"
        choices = ["latest"]
        if checkpoints_dir.exists():
            gen_files = list(checkpoints_dir.glob("generator_*.safetensors"))
            timestamps = [f.stem.replace("generator_", "") for f in gen_files]
            # Newest first
            timestamps.sort(reverse=True)
            choices.extend(timestamps)
        return choices

    def run(image_path: str, source_direction: str, target_direction: str) -> str:
        if model_cache["model"] is None:
            return "Error: No model is currently loaded. Go to 'Model Settings' to load a checkpoint."
        path = Path(image_path)
        if not path.exists():
            return f"File does not exist: {path}"
        try:
            output_path = predict_file(
                path, 
                source_direction, 
                target_direction, 
                model_cache["model"], 
                model_cache["device"]
            )
            return f"Saved: {output_path} (Using Model version: {model_cache['current_version']}, Checkpoint: {model_cache['current_timestamp']})"
        except Exception as e:
            return f"Prediction failed: {e}"

    def change_model(selected_timestamp: str, version_str: str) -> str:
        try:
            version = int(version_str)
            ts = None if selected_timestamp == "latest" else selected_timestamp
            new_model = load_model(model_cache["config"], model_cache["device"], timestamp=ts, version=version)
            model_cache["model"] = new_model
            model_cache["current_timestamp"] = selected_timestamp
            model_cache["current_version"] = version
            msg = f"Successfully loaded Model version {version} checkpoint: {selected_timestamp}"
            model_cache["status"] = msg
            return msg
        except Exception as e:
            return f"Failed to load model checkpoint: {e}"

    def on_version_change(version_str: str):
        version = int(version_str)
        choices = get_model_choices(version)
        try:
            new_model = load_model(model_cache["config"], model_cache["device"], timestamp=None, version=version)
            model_cache["model"] = new_model
            model_cache["current_timestamp"] = "latest"
            model_cache["current_version"] = version
            status_msg = f"Successfully loaded latest V{version} model checkpoint."
        except Exception as e:
            model_cache["model"] = None
            model_cache["current_timestamp"] = "none"
            model_cache["current_version"] = version
            status_msg = f"No V{version} checkpoints found. (Error: {e})"
        
        model_cache["status"] = status_msg
        return gr.Dropdown(choices=choices, value="latest"), status_msg

    def refresh_choices(version_str: str):
        version = int(version_str)
        return gr.Dropdown(choices=get_model_choices(version), value="latest")

    directions = list(DIRECTIONS)
    
    with gr.Blocks(title="Pixel Sprite Direction Transfer") as app:
        gr.Markdown("# Pixel Sprite Direction Transfer Control Panel")
        
        with gr.Tabs():
            with gr.Tab("Direction Transfer"):
                with gr.Row():
                    with gr.Column():
                        image_path = gr.Textbox(label="Input Image Path (RGBA)")
                        source_direction = gr.Dropdown(directions, value="down", label="Source direction")
                        target_direction = gr.Dropdown(directions, value="left", label="Target direction")
                        convert_btn = gr.Button("Convert", variant="primary")
                    status = gr.Textbox(label="Conversion Status", interactive=False)
                convert_btn.click(run, [image_path, source_direction, target_direction], status)
                
            with gr.Tab("Model Settings"):
                gr.Markdown("### Load and Manage Model Checkpoints")
                with gr.Row():
                    with gr.Column():
                        version_dropdown = gr.Dropdown(
                            choices=["1", "2"], 
                            value=str(model_cache["current_version"]), 
                            label="Model Architecture Version"
                        )
                        model_dropdown = gr.Dropdown(
                            choices=get_model_choices(model_cache["current_version"]), 
                            value=model_cache["current_timestamp"], 
                            label="Select Checkpoint Version (Timestamp)"
                        )
                        with gr.Row():
                            load_btn = gr.Button("Load Selected Model", variant="primary")
                            refresh_btn = gr.Button("Refresh Model List")
                    current_model_info = gr.Textbox(
                        label="Active Model Load Status", 
                        value=model_cache["status"], 
                        interactive=False
                    )
                
                version_dropdown.change(on_version_change, inputs=[version_dropdown], outputs=[model_dropdown, current_model_info])
                load_btn.click(change_model, inputs=[model_dropdown, version_dropdown], outputs=[current_model_info])
                refresh_btn.click(refresh_choices, inputs=[version_dropdown], outputs=[model_dropdown])

    return app
