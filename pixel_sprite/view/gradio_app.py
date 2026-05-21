from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch

from pixel_sprite.config import load_config, resolve_project_path
from pixel_sprite.controller.infer import DIRECTIONS, load_model, predict_file


def create_app():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try loading the latest model initially, but handle cases where checkpoints are empty
    try:
        model = load_model(config, device)
        initial_status = "Default (Latest) model loaded successfully."
        current_ts = "latest"
    except Exception as e:
        model = None
        initial_status = f"No model loaded. Please check checkpoints or train the model. (Error: {e})"
        current_ts = "none"

    model_cache = {
        "config": config,
        "device": device,
        "model": model,
        "status": initial_status,
        "current_timestamp": current_ts
    }

    def get_model_choices() -> list[str]:
        checkpoints_path = config["misc"].get("checkpoints_path", "./checkpoints")
        checkpoints_dir = resolve_project_path(checkpoints_path)
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
            return f"Saved: {output_path} (Using version: {model_cache['current_timestamp']})"
        except Exception as e:
            return f"Prediction failed: {e}"

    def change_model(selected_timestamp: str) -> str:
        try:
            ts = None if selected_timestamp == "latest" else selected_timestamp
            new_model = load_model(model_cache["config"], model_cache["device"], timestamp=ts)
            model_cache["model"] = new_model
            model_cache["current_timestamp"] = selected_timestamp
            msg = f"Successfully loaded model version: {selected_timestamp}"
            model_cache["status"] = msg
            return msg
        except Exception as e:
            return f"Failed to load model checkpoint: {e}"

    def refresh_choices():
        return gr.Dropdown(choices=get_model_choices(), value="latest")

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
                        model_dropdown = gr.Dropdown(
                            choices=get_model_choices(), 
                            value="latest", 
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
                load_btn.click(change_model, inputs=[model_dropdown], outputs=[current_model_info])
                refresh_btn.click(refresh_choices, inputs=[], outputs=[model_dropdown])

    return app
