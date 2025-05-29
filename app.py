import gradio as gr
import torch
from run import run_st


# Function to handle style transfer with custom epochs
def style_transfer_app(content, style, epochs=300):
    if content is None or style is None:
        return None, "Please upload both content and style images."

    try:
        output = run_st(content, style, epochs=int(epochs))
        return output, f"✅ Style transfer completed successfully with {epochs} epochs!"
    except Exception as e:
        error_msg = f"❌ Error during style transfer: {str(e)}"
        print(error_msg)
        return None, error_msg


# Create the enhanced interface
with gr.Blocks(
    title="🎨 Cnnvas - Neural Style Transfer",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    .contain {
        margin: 0 auto !important;
        text-align: center !important;
    }
    .block {
        margin: 0 auto !important;
    }
    """,
) as iface:

    # Header
    with gr.Column():
        gr.Markdown(
            """
            <div style="text-align: center;">
            
            # 🎨 Cnnvas - Neural Style Transfer
            
            Transform your photos with artistic styles using a VGG-16 based neural network!
            </div>
            """
        )

        gr.Markdown(
            """       
            <div style="max-width: 600px; margin: 0 auto;">
            
            ### How to use:
            
            </div>
            """,
            elem_classes="instructions-header",
        )

        gr.Markdown(
            """
            <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            
            1. **Upload your content image** (the photo you want to stylize)
            2. **Upload your style image** (the artwork whose style you want to apply)  
            3. **Adjust training epochs** (more epochs = better quality but slower processing)
            4. **Click "Generate Stylized Image"** and wait for the magic! ✨
            
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input Images")

            content_input = gr.Image(
                label="📷 Content Image",
                type="pil",
                height=250,
                sources=["upload", "clipboard"],
                show_label=True,
            )

            style_input = gr.Image(
                label="🎨 Style Image",
                type="pil",
                height=250,
                sources=["upload", "clipboard"],
                show_label=True,
            )

            # Training controls
            gr.Markdown("### ⚙️ Training Settings")
            epochs_slider = gr.Slider(
                minimum=100,
                maximum=2000,
                value=500,
                step=100,
                label="Training Epochs",
                info="More epochs = higher quality but slower processing",
                interactive=True,
            )

            generate_btn = gr.Button("🎨 Generate Stylized Image", variant="primary", size="lg", scale=1)
            device_info = (f"🖥️ **Device:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
            gr.Markdown(device_info)

        # Right column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Result")

            output_image = gr.Image(
                label="Stylized Result",
                type="pil",
                height=250,
                show_label=False,
                interactive=False,
            )

            # Status message
            status_message = gr.Textbox(
                label="Status",
                value="Ready to generate! Upload your images and click the button.",
                interactive=False,
                lines=2,
            )

    gr.Markdown(
        """
        <div style="max-width: 600px; margin: 0 auto;">
        
        ### 💡 Tips for best results:
        
        </div>
        """
    )

    gr.Markdown(
        """
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
        
        - **Content images**: Works best with clear subjects and good lighting
        - **Style images**: Try famous paintings, abstract art, or textures
        - **Epochs**: Start with 500 epochs, increase for better quality
        - **Processing time**: Expect 2-5 minutes depending on your hardware
        
        </div>
        """
    )

    # Progress tracking
    generate_btn.click(
        fn=style_transfer_app,
        inputs=[content_input, style_input, epochs_slider],
        outputs=[output_image, status_message],
        show_progress=True,
        scroll_to_output=True,
    )

# Launch configuration
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
