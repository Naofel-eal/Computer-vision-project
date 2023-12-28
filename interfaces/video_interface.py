import gradio as gr

from services.medias.video_processor import VideoProcessor
from models.medias.video import Video

class VideoInterface:
    
    def __init__(self):
        self.video_tab = gr.Blocks()
        self.video_processor = VideoProcessor()
        self.personsDTO = []
        
        with self.video_tab:  
            gr.Markdown(
                """
                    # Import your video
                    Import the video you want to edit from the interface below.
                """
            )
            video_input = gr.Video(label="Input video")
            analyse_button = gr.Button("Analyze the video")
                
            gr.Markdown(
                """
                    # Settings
                    Here you can choose your settings before making changes to the video.
                """
            )
            persons_faces_output = gr.Gallery(label="Detected faces of people", show_label=True, preview=True)
            checkboxes = gr.CheckboxGroup(label="Select the people you want to blur", info="You can select as many people as you want.")
            gradient_blur_checkbox = gr.Checkbox(label="Gradient blur", info="Check the box bellow if you want to apply a gradient circular blur rather than a raw rectangular blur. (Note that the video processing will take longer with gradient blur)")
            blur_button = gr.Button("Video processing")
            
            gr.Markdown(
                """
                    # Output video
                    You can download the output video by clicking the download icon on the upper right corner of the video.
                """
            )
            video_output = gr.Video(label="Output video")
            
            analyse_button.click(self.analyse_video, inputs=video_input, outputs=[persons_faces_output, checkboxes])
            checkboxes.change(self.update_persons_should_be_blurred, inputs=checkboxes)
            blur_button.click(self.apply_blur, inputs=[video_input, gradient_blur_checkbox], outputs=video_output)
            video_input.change(self.reset, outputs=[persons_faces_output, checkboxes, video_output])
            

    def analyse_video(self, video_path):
        self.personsDTO = self.video_processor.get_persons(video=Video(video_path))
        persons_labels = []
        persons_faces = []
        for i, person in enumerate (self.personsDTO):
            persons_labels.append(f"Person {i}")
            persons_faces.append((person.face, persons_labels[i]))
        checkboxes = gr.CheckboxGroup(choices=persons_labels, label="Select the persons you want to blur", info="You can select as many people as you want.")
        return persons_faces, checkboxes

    def update_persons_should_be_blurred(self, checkboxes):
        if self.personsDTO:
            for personDTO in self.personsDTO:
                personDTO.should_be_blurred = False
        
        if checkboxes:
            for label in checkboxes:
                index = int(label.replace("Person ", ""))
                self.personsDTO[index].should_be_blurred = True
            
    def apply_blur(self, video_path, gradient_blur_checkbox_value):
        blurred_video_path = self.video_processor.save(video=Video(video_path), personsDTO=self.personsDTO, gradual=gradient_blur_checkbox_value)
        return blurred_video_path
    
    def reset(self):
        self.video_processor.reset()
        self.personsDTO = []
        checkboxes = gr.CheckboxGroup(choices=[], value=[], label="Select the persons you want to blur", info="You can select as many people as you want.")
        return None, checkboxes, None
