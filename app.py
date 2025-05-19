import gradio as gr
import spaces
import torch
import os
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline

# Load model
model_name = os.environ.get('MODEL_NAME', 'UnfilteredAI/NSFW-gen-v2.1')
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
pipe.to('cuda')

# Compel setup
compel = Compel(
  tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
  text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)

# Default negative prompt
default_negative_prompt = "(low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn, (deformed | distorted | disfigured:1.3), bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers:1.4, disconnected limbs, blurry, amputation."

# Example prompts
example_prompts = [
    ["a gorgeous nude model posing seductively in a luxurious bedroom, perfect lighting, detailed skin texture, professional photography, 8k", default_negative_prompt, 40, 7.5, 1024, 1024, 4],
    ["two fully nude naked asian female fitness models doing sit-ups in the gym, filmed from front, realistic, professional photography, 8k", default_negative_prompt, 45, 7.0, 1024, 1024, 4],
]

# Image generation function
@spaces.GPU(duration=120)
def generate(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, num_samples, progress=gr.Progress()):
    progress(0, desc="Preparing")
    embeds, pooled = compel(prompt)
    neg_embeds, neg_pooled = compel(negative_prompt)
    
    progress(0.1, desc="Generating images")
    
    # Define proper callback for step end
    def callback_on_step_end(pipe, i, t, callback_kwargs):
        progress((i + 1) / num_inference_steps)
        return callback_kwargs
    
    images = pipe(
        prompt_embeds=embeds,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=neg_embeds,
        negative_pooled_prompt_embeds=neg_pooled,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        num_images_per_prompt=num_samples,
        callback_on_step_end=callback_on_step_end
    ).images
    
    return images

# CSS styles
css = """
.gallery-item {
    transition: transform 0.2s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 10px;
}
.gallery-item:hover {
    transform: scale(1.03);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.container {
    max-width: 1200px;
    margin: auto;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(90deg, rgba(76,0,161,0.8) 0%, rgba(28,110,164,0.8) 100%);
    border-radius: 10px;
    color: white;
}
.slider-container {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.prompt-container {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    border: 1px solid #d0e8ff;
}
.examples-header {
    background: linear-gradient(90deg, rgba(41,128,185,0.7) 0%, rgba(142,68,173,0.7) 100%);
    color: white;
    padding: 0.5rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 0.5rem;
}
"""

# Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="header">
        <h1>🎨 Unfiltered AI NSFW Image Generator</h1>
        <p>Enter creative prompts and generate high-quality images.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group(elem_classes="prompt-container"):
                prompt = gr.Textbox(label="Prompt", placeholder="Describe your desired image...", lines=3)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    value=default_negative_prompt,
                    lines=3
                )
            
            with gr.Group(elem_classes="slider-container"):
                with gr.Row():
                    with gr.Column():
                        steps = gr.Slider(minimum=20, maximum=100, value=60, step=1, label="Inference Steps (Quality)", info="Higher values improve quality (longer generation time)")
                        guidance = gr.Slider(minimum=1, maximum=15, value=7, step=0.1, label="Guidance Scale (Creativity)", info="Lower values create more creative results")
                    
                    with gr.Column():
                        with gr.Row():
                            width = gr.Slider(minimum=512, maximum=1536, value=1024, step=128, label="Width")
                            height = gr.Slider(minimum=512, maximum=1536, value=1024, step=128, label="Height")
                        
                        num_samples = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Number of Images", info="Number of images to generate at once")
            
            generate_btn = gr.Button("🚀 Generate Images", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Generated Images", elem_classes="gallery-item", columns=2, object_fit="contain", height=650)
    
    gr.HTML("""<div class="examples-header"><h3>✨ Example Prompts</h3></div>""")
    gr.Examples(
        examples=example_prompts,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, num_samples],
        outputs=output_gallery,
        fn=generate,
        cache_examples=True,
    )
    
    # Event connections
    generate_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, steps, guidance, width, height, num_samples],
        outputs=output_gallery
    )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 10px;">
        <p>💡 Tip: For high-quality images, use detailed prompts and higher inference steps.</p>
        <p>Example: Add quality terms like "professional photography, 8k, highly detailed, sharp focus, HDR" to your prompts.</p>
    </div>
    """)

demo.launch()