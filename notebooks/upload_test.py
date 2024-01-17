import gradio as gr

def test(x):
    return x.name
    
with gr.Blocks() as demo:
    inp = gr.File(label="Input")
    vis_out=gr.File(label="output")
    with gr.Row():
        btn = gr.Button("Run")
    btn.click(fn=test, inputs=[inp], outputs=[vis_out])

demo.launch(server_port=8000) 