import streamlit as st
import pandas as pd
import numpy as np

import time
import cv2


st.set_page_config("NLP App", layout="wide")

with st.sidebar:
    st.title("NLP Project")
    st.page_link("app.py", label="App", icon="❌")
    st.page_link("./pages/text_detection_model.py", label="Text detection model", icon="❌")
    st.page_link("./pages/ocr_model.py", label="OCR model", icon="✔️")
    st.page_link("./pages/context_extraction_model.py", label="Context extraction", icon="✔️")

# ----------------------------------------------------------------------------------------------------

input_shape = (1024, 1024, 3)

# ----------------------------------------------------------------------------------------------------

def get_input_model_image(imported_image):
    output_image = []

    for image in imported_image:
        try:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
        except:
            pass
        finally:
            #Resize the image to a 1024x0124 square, keeping the aspect_ratio
            #img = tf.image.resize(
            #                        images=image,
            #                        size=input_shape[:2],
            #                        method=tf.image.ResizeMethod.BILINEAR,
            #                        preserve_aspect_ratio=True,
            #                        antialias=False,
            #                        name=None
            #                        )
            
            #Resize the image to a 1024x0124 square
            img = cv2.resize(image, input_shape[:2])

            new_shape = img.shape # retrieve the new shape of the image

            canvas = np.zeros(input_shape) # Create an 1024x1024 image filled with 0s
            canvas[:new_shape[0], :new_shape[1]] = img # Add the image into the 1024x1024 0s square


            cv2.imwrite(f"./ressources/images/app_output_1.jpg", np.array(canvas))
            output_image.append(cv2.imread("./ressources/images/app_output_1.jpg"))

    return output_image

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# ---------------------------------------------------------------------------------------------------

st.title("Context extraction from scanned documents")


# -------------------------------------- Files importation ---------------------------------------------

importFrame = st.container()
outputFrame = st.container()

imported_file = importFrame.file_uploader("Import your file(s)", [".jpg", ".png"], True)

col1, col2 = importFrame.columns([1,8])
testFile = col1.button("Test Image", use_container_width=True)
process = col2.button("Process", use_container_width=True)

if process and imported_file or testFile:
    if imported_file == [] or testFile:
        imported_file = [None]
        imported_file[0] = cv2.imread("./ressources/images/20240523_170445.jpg")
        imported_image_channel = 'BGR'
    else:
        imported_image_channel = 'RGB'

    statusBar = importFrame.status("Processing...", expanded=True)

    statusBar.write("Loading data into the pipeline...")

    input_images = get_input_model_image(imported_file)

    statusBar.write("Scanning images for words...")

    # Text detection model
    output_image = cv2.imread("./ressources/images/process_error.jpg")

    statusBar.write("Retriving words...")

    # OCR model
    time.sleep(1)

    statusBar.write("Extraction of the context...")

    # Context extraction model
    time.sleep(1)

    text_output = '''The text detection model could not be trained, the pipeline could not be set up.'''

    statusBar.update(label="Process completed!", state="complete", expanded=True)

    # ------------- Output the results ------------------

    outputFrame.header("Results")

    outputFrame.subheader("First image example")

    col1, col2, col3 = outputFrame.columns(3)

    col1.image(imported_file[0], channels=imported_image_channel, caption="Imported image")

    col2.image(input_images[0], channels='BGR', caption="Image inpputed into the model (zero-padded)")

    col3.image(output_image, channels='BGR', caption="Bounding boxed image")

    chat_output = outputFrame.chat_message("ai")
    chat_output.write_stream(stream_data(text_output))

    outputFrame.divider()

    outputFrame.subheader("Full results")

    # Tableau avec le context pour chaque fichier, statut reussit ou raté

    df_output = pd.DataFrame([])

    outputFrame.dataframe(df_output, hide_index=True, use_container_width=True)

elif process:
    importFrame.error("You must import at least one file!")


