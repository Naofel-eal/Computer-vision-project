import gradio as gr
from interfaces.image_interface import ImageInterface
from interfaces.video_interface import VideoInterface

image_tab = ImageInterface().image_tab
video_tab = VideoInterface().video_tab

iface = gr.TabbedInterface([image_tab, video_tab], ["Image", "Video"])

if __name__ == "__main__":
    iface.launch()