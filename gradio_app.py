import gradio as gr
from model import get_model

MODEL_NAME = 'resnet'
model = get_model(MODEL_NAME)


def find_similar_images(inp):
    pass


inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=3)
gr.Interface(fn=find_similar_images, inputs=inputs,
             outputs=outputs).launch()
