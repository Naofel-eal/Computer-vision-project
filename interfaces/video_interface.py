import gradio as gr

class VideoInterface:
    
    def __init__(self):
        self.video_tab = gr.Interface(
            fn=self.process_video,
            inputs=["video"],
            outputs=["video"]
        )

    def process_video(self, video):
        # code de traitement de la vidéo ici
        return video