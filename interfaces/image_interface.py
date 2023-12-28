import gradio as gr

from services.medias.image_processor import ImageProcessor

class ImageInterface:
    
    def __init__(self):
        self.image_tab = gr.Blocks()
        self.image_processor = ImageProcessor()
        self.personsDTO = []
        
        with self.image_tab:  
            gr.Markdown(
                """
                    # Import your image
                    Import the image you want to edit from the interface below.
                """
            )
            image_input = gr.Image(label="Input image")
            analyse_button = gr.Button("Analyze the image")
                
            gr.Markdown(
                """
                    # Settings
                    Here you can choose your settings before making changes to the image.
                """
            )
            persons_faces_output = gr.Gallery(label="Detected faces of people", show_label=True, preview=True)
            checkboxes = gr.CheckboxGroup(label="Select the people you want to blur", info="You can select as many people as you want.")
            gradient_blur_checkbox = gr.Checkbox(label="Gradient blur", info="Check the box bellow if you want to apply a gradient circular blur rather than a raw rectangular blur. (Note that the image processing will take longer with gradient blur)")
            blur_button = gr.Button("Image processing")
            
            gr.Markdown(
                """
                    # Output image
                    You can download the output image by clicking the download icon on the upper right corner of the image.
                """
            )
            image_output = gr.Image(label="Output image")
            
            analyse_button.click(self.analyse_image, inputs=image_input, outputs=[persons_faces_output, checkboxes])
            checkboxes.change(self.update_persons_should_be_blurred, inputs=checkboxes)
            blur_button.click(self.apply_blur, inputs=[image_input, gradient_blur_checkbox], outputs=image_output)
            image_input.change(self.reset, outputs=[persons_faces_output, checkboxes, image_output])
            

    def analyse_image(self, image):
        self.personsDTO = self.image_processor.get_persons(frame=image)
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
            
    def apply_blur(self, image, gradient_blur_checkbox_value):
        image = self.image_processor.apply_blur(frame=image, personsDTO=self.personsDTO, gradual=gradient_blur_checkbox_value)
        return image
    
    def reset(self):
        self.image_processor.reset()
        self.personsDTO = []
        checkboxes = gr.CheckboxGroup(choices=[], value=[], label="Select the persons you want to blur", info="You can select as many people as you want.")
        return None, checkboxes, None
    