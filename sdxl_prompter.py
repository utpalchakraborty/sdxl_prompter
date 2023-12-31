import datetime
import os

import gradio as gr
from gradio import Blocks
from loguru import logger

from helper_functions import (
    select_random_line,
    parse_comma_separated_strings,
    read_file_to_list,
    get_random_element,
    generate_unique_filename,
)
from llm import (
    generate_from_llm,
    SYSTEM_PROMPT,
    default_top_p,
    default_top_k,
    default_temperature,
)
from sdxl_generate import generate_image
from sdxl_prompt_constants import (
    styles_of_photo_str,
    framing_of_photo_str,
    lighting_of_photo_str,
    photographers_list_str,
    prompt_directory,
)

styles_of_photo = parse_comma_separated_strings(styles_of_photo_str)
framing_of_photo = parse_comma_separated_strings(framing_of_photo_str)
lighting_of_photo = parse_comma_separated_strings(lighting_of_photo_str)
photographers_list = parse_comma_separated_strings(photographers_list_str)
settings = read_file_to_list("photography_settings.txt")


# photographers = parse_comma_separated_strings(photographers_list.lower())
# photographers = sorted(photographers)
# print(",".join(photographers))


def add_element_to_prompt(prompt: str, element: str) -> str:
    element = element.strip()
    if element:
        prompt = f"{prompt}, {element}"
    return prompt


def generate_prompt(
    style: str,
    subject: str,
    important_feature: str,
    more_details: str,
    framing: str,
    apply_setting: bool,
    lighting: str,
    photographer: str,
    additional_specifiers: str,
) -> str:
    # [STYLE OF PHOTO]
    # photo of a [SUBJECT], [IMPORTANT FEATURE], [MORE DETAILS], [POSE OR ACTION],
    # [FRAMING], [SETTING / BACKGROUND], [LIGHTING],
    # [CAMERA ANGLE], [CAMERA PROPERTIES], in style of[PHOTOGRAPHER],
    prompt = f"a {style.strip()} photo of {subject.strip()}"
    prompt = add_element_to_prompt(prompt, important_feature)
    prompt = add_element_to_prompt(prompt, more_details)
    prompt = add_element_to_prompt(prompt, framing)

    if apply_setting:
        prompt = add_element_to_prompt(prompt, get_random_element(settings))

    prompt = add_element_to_prompt(prompt, lighting)
    photographer = photographer.strip()
    if photographer:
        prompt = f"{prompt}, in style of {photographer}"
    prompt = add_element_to_prompt(prompt, additional_specifiers)
    return prompt


def randomize_prompt(
    subject: str, important_feature: str, more_details: str, additional_specifiers: str
) -> str:
    return generate_prompt(
        get_random_element(styles_of_photo),
        subject,
        important_feature,
        more_details,
        get_random_element(framing_of_photo),
        True,
        get_random_element(lighting_of_photo),
        get_random_element(photographers_list),
        additional_specifiers,
    )


current_directory = os.getcwd()  # Get the current working directory
log_directory = os.path.join(current_directory, "prompt_logs")  # get the logs directory
log_file_name = os.path.join(log_directory, generate_unique_filename())


def append_to_log(prompt: str) -> str:
    timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log_file_name, "a") as f:
        f.write(f"------------{timestamp_str}\n")
        f.write(f"{prompt}\n")
    return f"Appended to logfile {log_file_name} at {timestamp_str}"


def change_tab_to_llm(prompt: str):
    return gr.Tabs(selected=2), prompt


def init_ui() -> Blocks:
    with gr.Blocks(title="SDXL Prompter", theme="gstaff/xkcd") as demo:
        gr.Markdown("Welcome to the SDXL Prompt & Image Generator!")
        with gr.Tabs() as tabs:
            with gr.TabItem("Generate Prompt", id=0):
                # set up the main prompt data
                subject_textbox = gr.Textbox(
                    label="Subject", value="a very beautiful woman"
                )
                important_feature_textbox = gr.Textbox(label="Important Feature")
                more_details_textbox = gr.Textbox(label="More Details")

                # now the additional specifiers like style, framing, lighting, photographer
                with gr.Row():
                    with gr.Column(scale=1, variant="compact"):
                        style_dropdown = gr.Dropdown(
                            styles_of_photo, label="Style", value=""
                        )
                    with gr.Column(scale=1, variant="compact"):
                        framing_dropdown = gr.Dropdown(
                            framing_of_photo, label="Framing", value=""
                        )
                with gr.Row():
                    with gr.Column(scale=1, variant="compact"):
                        lighting_dropdown = gr.Dropdown(
                            lighting_of_photo, label="Lighting", value=""
                        )
                    with gr.Column(scale=1, variant="compact"):
                        photographer_dropdown = gr.Dropdown(
                            photographers_list, label="Photographer", value=""
                        )

                # apply a setting?
                apply_setting_checkbox = gr.Checkbox(
                    label="Apply a setting", value=True
                )

                # any additional specifiers for the prompt
                additional_specifiers_textbox = gr.Textbox(
                    label="Additional Specifiers",
                    value="8k, (sharp:1.2), masterpiece, (intricate details:1.12)",
                )

                # final prompt output here
                output_prompt_textbox = gr.Textbox(
                    label="Prompt", lines=10, show_copy_button=True
                )
                with gr.Row():
                    with gr.Column(scale=1, variant="compact"):
                        generate_btn = gr.Button("Generate")
                    with gr.Column(scale=1, variant="compact"):
                        randomize_btn = gr.Button("Randomize")
                    with gr.Column(scale=1, variant="compact"):
                        log_button_1 = gr.Button("Log Prompt")
                    with gr.Column(scale=1, variant="compact"):
                        send_to_llm_btn = gr.Button("Send to LLM")
                log_label = gr.Markdown()
                log_button_1.click(
                    append_to_log,
                    inputs=[output_prompt_textbox],
                    outputs=[log_label],
                    api_name="Log_Prompt",
                )
                generate_btn.click(
                    fn=generate_prompt,
                    inputs=[
                        style_dropdown,
                        subject_textbox,
                        important_feature_textbox,
                        more_details_textbox,
                        framing_dropdown,
                        apply_setting_checkbox,
                        lighting_dropdown,
                        photographer_dropdown,
                        additional_specifiers_textbox,
                    ],
                    outputs=output_prompt_textbox,
                    api_name="Generate_Prompt",
                )
                randomize_btn.click(
                    fn=randomize_prompt,
                    inputs=[
                        subject_textbox,
                        important_feature_textbox,
                        more_details_textbox,
                        additional_specifiers_textbox,
                    ],
                    outputs=output_prompt_textbox,
                    api_name="Randomize_Prompt",
                )
            with gr.TabItem("Cinematic Prompt", id=1):
                files = []
                for file in os.listdir(prompt_directory):
                    if file.endswith(".csv"):
                        files.append(file)
                cinematic_prompt_filename = gr.Dropdown(files, label="Prompt File")
                cinematic_prompt_line_number = gr.Textbox(label="Line Number")
                cinematic_prompt = gr.Textbox(
                    label="Cinematic Prompt", lines=10, show_copy_button=True
                )
                with gr.Row():
                    with gr.Column(scale=1, variant="compact"):
                        cinematic_prompt_btn = gr.Button("Generate")
                    with gr.Column(scale=1, variant="compact"):
                        cinematic_prompt_sendto_llm_btn = gr.Button("Send to LLM")
                    cinematic_prompt_btn.click(
                        select_random_line,
                        inputs=[cinematic_prompt_filename],
                        outputs=[cinematic_prompt_line_number, cinematic_prompt],
                        api_name="Cinematic_Prompt",
                    )
            with gr.TabItem("LLM Enhance", id=2):
                with gr.Row():
                    with gr.Column(scale=3, variant="compact"):
                        llm_system_prompt = gr.Textbox(
                            label="LLM System Prompt", lines=9, value=SYSTEM_PROMPT
                        )
                    with gr.Column(min_width=150, scale=1):
                        with gr.Tab(label="Generation Parameters"):
                            # gr.Markdown("# Parameters")
                            top_p_slider = gr.Slider(
                                minimum=0.05,
                                maximum=1.0,
                                value=default_top_p,
                                step=0.05,
                                interactive=True,
                                label="Top-p (nucleus sampling)",
                            )
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=default_top_k,
                                step=1,
                                interactive=True,
                                label="Top-k",
                            )
                            temperature_slider = gr.Slider(
                                minimum=default_temperature,
                                maximum=0.001,
                                value=default_temperature,
                                step=0.0001,
                                interactive=True,
                                label="Temperature",
                            )
                llm_prompt = gr.Textbox(label="LLM Prompt", lines=5)
                llm_prompt_btn = gr.Button("Enhance")
                llm_output = gr.Textbox(
                    label="LLM Output", lines=15, show_copy_button=True
                )
                llm_prompt_btn.click(
                    generate_from_llm,
                    inputs=[
                        llm_prompt,
                        llm_system_prompt,
                        top_p_slider,
                        top_k_slider,
                        temperature_slider,
                    ],
                    outputs=[llm_output],
                    api_name="LLM_Prompt",
                )
                log_button_2 = gr.Button("Log Prompt")
                log_label = gr.Markdown()
                log_button_2.click(
                    append_to_log,
                    inputs=[llm_output],
                    outputs=[log_label],
                    api_name="Log_Prompt",
                )
            with gr.TabItem("Generate Image", id=3):
                with gr.Row():
                    with gr.Column(scale=3, variant="compact"):
                        final_image = gr.Gallery(
                            label="Final Image",
                            show_label=True,
                            visible=True,
                            height=1024,
                            object_fit="contain",
                            show_download_button=True,
                        )
                    with gr.Column(scale=1, variant="compact"):
                        with gr.Group():
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=15,
                                    step=0.01,
                                    value=3.5,
                                    info="Higher value means style is cleaner, vivider, and more artistic.",
                                )
                            with gr.Row():
                                steps_scale = gr.Slider(
                                    label="Steps",
                                    minimum=1.0,
                                    maximum=60,
                                    step=1,
                                    value=60,
                                    info="Higher value means more detailed image, but also more artifacts.",
                                )
                            with gr.Row():
                                sharpen_scale = gr.Slider(
                                    label="Sharpen",
                                    minimum=0.0,
                                    maximum=2,
                                    step=0.1,
                                    value=1.1,
                                    info="Higher value means more sharpened image. 1 = Original Image",
                                )
                            with gr.Row():
                                contrast_scale = gr.Slider(
                                    label="Contrast",
                                    minimum=0.0,
                                    maximum=2,
                                    step=0.1,
                                    value=1.1,
                                    info="Higher value means more image contrast. 1 = Original Image",
                                )
                            with gr.Row():
                                upscale_by = gr.Slider(
                                    label="Upscale",
                                    minimum=1,
                                    maximum=4,
                                    step=0.5,
                                    value=1.5,
                                    info="Higher value means larger image.",
                                )
                            with gr.Row():
                                seed_textbox = gr.Number(
                                    label="Seed", value=-1, precision=0
                                )
                            with gr.Row():
                                with gr.Group():
                                    use_refiner_checkbox = gr.Checkbox(
                                        label="Use Refiner", value=False, interactive=True
                                    )
                                    refiner_switch_at = gr.Slider(
                                        label="Refiner Switch at",
                                        minimum=0.5,
                                        maximum=1.0,
                                        step=0.1,
                                        value=0.8,
                                    )
                            with gr.Row():
                                face_restore_checkbox = gr.Checkbox(
                                    label="Face Restore", value=False, interactive=True
                                )
                            with gr.Row():
                                image_data_output = gr.Textbox(
                                    label="Image Data",
                                    lines=10,
                                    show_copy_button=True,
                                    interactive=False,
                                )

                with gr.Group():
                    with gr.Row():
                        image_prompt = gr.Textbox(
                            label="Image Prompt",
                            lines=3,
                            show_copy_button=True,
                        )
                    with gr.Row():
                        image_prompt_negative = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            show_copy_button=True,
                            value="ugly, deformed",
                        )

                image_generate_btn = gr.Button("Generate")
                image_generate_btn.click(
                    fn=generate_image,
                    inputs=[
                        image_prompt,
                        image_prompt_negative,
                        guidance_scale,
                        steps_scale,
                        seed_textbox,
                        use_refiner_checkbox,
                        sharpen_scale,
                        contrast_scale,
                        upscale_by,
                        face_restore_checkbox,
                        refiner_switch_at
                    ],
                    outputs=[final_image, image_data_output],
                    api_name="Image_Generate",
                )
        send_to_llm_btn.click(
            fn=change_tab_to_llm,
            inputs=[output_prompt_textbox],
            outputs=[tabs, llm_prompt],
            api_name="Change_Tab",
        )
        cinematic_prompt_sendto_llm_btn.click(
            fn=change_tab_to_llm,
            inputs=[cinematic_prompt],
            outputs=[tabs, llm_prompt],
            api_name="Change_Tab",
        )
    return demo


if __name__ == "__main__":
    logger.info("Initializing UI & launching")
    init_ui().launch()
