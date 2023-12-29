import gradio as gr
from interfaces.image_interface import ImageInterface
from interfaces.video_interface import VideoInterface

image_tab = ImageInterface().image_tab
video_tab = VideoInterface().video_tab

iface = gr.TabbedInterface([video_tab, image_tab], ["Video processing", "Image processing"], theme=gr.themes.Soft(), css=".part2-video, .part3-video, .part2-image, .part3-image {display: none !important;}")

if __name__ == "__main__":
    iface.launch(show_api=False)