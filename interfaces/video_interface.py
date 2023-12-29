import gradio as gr
from models.medias.video import Video
from services.medias.video_processor import VideoProcessor

class VideoInterface:
    
    def __init__(self):
        self.video_tab = gr.Blocks()
        self.video_processor = VideoProcessor()
        self.personsDTO = []
        
        with self.video_tab:
            with gr.Column(elem_classes="part1") as self.part1_container:
                gr.Markdown(
                    """
                        # Import your video
                        Import the video you want to edit from the interface below.
                    """
                )
                video_input = gr.Video(label="Input video")
                analyse_button = gr.Button("Analyze the video")

            part2_container = gr.Column(elem_classes="part2")
            with part2_container:
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

            part3_container = gr.Column(elem_classes="part3")
            with part3_container:
                gr.Markdown(
                    """
                        # Output video
                        You can download the output video by clicking the download icon on the upper right corner of the video.
                    """
                )
                video_output = gr.Video(label="Output video")
            
            analyse_button.click(self.analyse_video, inputs=video_input, outputs=[persons_faces_output, checkboxes], js="(video_input) => {document.querySelector('.part2').setAttribute('style', 'display: block !important;');return video_input;}")
            checkboxes.change(self.update_persons_should_be_blurred, inputs=checkboxes)
            blur_button.click(self.apply_blur, inputs=gradient_blur_checkbox, outputs=[video_output], js="(gradient_blur_checkbox) => {document.querySelector('.part3').setAttribute('style', 'display: block !important;');return gradient_blur_checkbox;}")
            video_input.change(self.reset, outputs=[persons_faces_output, checkboxes, video_output], js="() => {document.querySelector('.part2').setAttribute('style', 'display: none !important;');document.querySelector('.part3').setAttribute('style', 'display: none !important;')}")
            

    def analyse_video(self, video_path):
        self.video = Video(video_path)
        self.personsDTO = self.video_processor.get_persons(self.video)
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
            
    def apply_blur(self, gradient_blur_checkbox_value):
        blurred_video_path = self.video_processor.save(video=self.video, personsDTO=self.personsDTO, gradual=gradient_blur_checkbox_value)
        return blurred_video_path
    
    def reset(self):
        self.video_processor.reset()
        self.personsDTO = []
        checkboxes = gr.CheckboxGroup(choices=[], value=[], label="Select the persons you want to blur", info="You can select as many people as you want.")
        return None, checkboxes, None
