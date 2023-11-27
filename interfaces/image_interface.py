import gradio as gr

class ImageInterface:
    
    def __init__(self):
        self.image_tab = gr.Interface(
            fn=self.process_image,
            inputs=["image"],
            outputs=["image"]
        )

    def process_image(self, image):
        # code de traitement de la vidéo ici
        return image