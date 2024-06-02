<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np

import cv2

st.set_page_config("NLP App - Text detection model", layout="wide")

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
            img = cv2.resize(image, input_shape[:2])

            new_shape = img.shape # retrieve the new shape of the image

            canvas = np.zeros(input_shape) # Create an 1024x1024 image filled with 0s
            canvas[:new_shape[0], :new_shape[1]] = img # Add the image into the 1024x1024 0s square


            cv2.imwrite(f"./ressources/images/app_output_1.jpg", np.array(canvas))
            output_image.append(cv2.imread("./ressources/images/app_output_1.jpg"))

    return output_image

# ---------------------------------------------------------------------------------------------------

st.title("Text Detection Model")

# -------------------------------------- Testing the model ---------------------------------------------

st.header("Testing the model")

importFrame = st.container()
outputFrame = st.container()
imported_file = importFrame.file_uploader("Import your file(s)", [".jpg", ".png"], True)

col1, col2 = importFrame.columns([1,8])
testFile = col1.button("Test Image", use_container_width=True)
process = col2.button("Process", use_container_width=True)


if process and imported_file or testFile:
    if imported_file == [] or testFile:
        imported_file = [0]
        imported_file[0] = cv2.imread("./ressources/images/20240523_170445.jpg")
        imported_image_channel = 'BGR'
    else:
        imported_image_channel = 'RGB'
    
    statusBar = importFrame.status("Processing...", expanded=False)

    input_images = get_input_model_image(imported_file)

    # Model output

    output_image = cv2.imread("./ressources/images/process_error.jpg")

    statusBar.update(label="Process completed!", state="complete", expanded=True)

    outputFrame.header("Result")
    outputFrame.subheader("First image example")

    # Output the results

    col1, col2, col3 = outputFrame.columns(3)

    col1.image(imported_file[0], channels=imported_image_channel, caption="Imported image")

    col2.image(input_images[0], channels='BGR', caption="Image inpputed into the model")

    col3.image(output_image, channels='BGR', caption="Model output")

elif process:
    importFrame.error("You must import at least one file!")

# ------------------------------------------- Dataset ---------------------------------------------------

datasetFrame = st.container()

datasetFrame.divider()

datasetFrame.header("Dataset")

datasetFrame.markdown(
    '''<div class="jss148 jss187 jss215"><h6 class="jss58 jss76 jss80"><b>General Information</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">Data is available under <a class="jss58 jss88 jss93 jss53 jss55" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> license.</h6></li><li><h6 class="jss58 jss76 jss80">Numbers in the papers should be reported on v0.1 test set. We will soon host a challenge on that.</h6></li><li><h6 class="jss58 jss76 jss80">Reach us out at <a class="jss58 jss88 jss93 jss53 jss55" href="mailto:textvqa@fb.com">textvqa@fb.com</a> for any questions, suggestions and feedback.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Images</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">Images for training and validation set are from OpenImages train set while images for test set are from OpenImages test set.</h6></li><li><h6 class="jss58 jss76 jss80">Validation set's images are contained in the zip for training set's images. The OpenImages dataset can be downloaded from <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/web/download.html">here</a>.</h6></li><li><h6 class="jss58 jss76 jss80"><b>Note:</b> Some of the images in OpenImages are rotated, please make sure to check the <b>Rotation</b> field in the Image IDs files for <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv">train</a> and <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv">test</a>.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Dataset Format</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">The json format mostly follows COCO-Text v2, except the "mask" field in "anns" is named as "points" for the polygon annotation.</h6></li><li><h6 class="jss58 jss76 jss80">The "points" field is a list of 2D coordinates like [x1, y1, x2, y2, ...]. Note that (x1,y1) is always the top-left corner of the text (in its own orientation), and order of the points is clockwise (for example, for horizontal text, (x1, y1) and (x2, y2) will form the top line).</h6></li><li><h6 class="jss58 jss76 jss80">The "bbox" field contains horizontal box converted from "points" for convenience, and "area" is computed based on width and height of "bbox". For any conversion to other formats such as rotated boxes or quadrilaterals, "points" should be used as the source of truth.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Annotation Details</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">In cases when the text is illegible or not in English, polygon is annotated normally but word will be annotated as a single "." symbol.</h6></li><li><h6 class="jss58 jss76 jss80">Word annotations are case-sensitive, and can contain punctuations too.</h6></li><li><h6 class="jss58 jss76 jss80">The annotators were instructed to draw exactly 4 points (quadrilaterals) whenever possible, and only draw more than 4 points when necessary (for cases like curved text).</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Relationship with TextVQA/TextCaps</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">The image ids in TextOCR match the ids in TextVQA.</h6></li><li><h6 class="jss58 jss76 jss80">train/val/test splits are the same as TextVQA/TextCaps. However due to privacy reasons, we removed 274 images from TextVQA while creating TextOCR.</h6></li></ul></div>''',
    unsafe_allow_html=True
    )

datasetFrame.page_link("https://textvqa.org/textocr/dataset/", label="Download the dataset")

# -------------------------------------- Fine tuning the model ---------------------------------------------

modelFrame = st.container()

modelFrame.divider()

modelFrame.header("Fine tuning the model")

modelFrame.subheader("Import the librairies")

modelFrame.markdown('''<p>We start by importing the necessary libraries, including TensorFlow and Keras for model building, and other utilities for data handling and visualization.</p>''',
                    unsafe_allow_html=True)

code = '''
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tqdm import tqdm

from datetime import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import concurrent.futures
import json
import cv2
import os
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Configure GPU (Optional)")

modelFrame.markdown('''<p>Ensure GPU memory growth is set if needed.</p>''',
                    unsafe_allow_html=True)

code = '''# Ensure GPU memory growth is set if needed
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define Paths and Constants")

modelFrame.markdown('''<p>Set the paths for the dataset and model saving directory. Initialize other constants like batch size.</p>''',
                    unsafe_allow_html=True)

code = '''MODEL_PATH = f"./Models/{dt.now().strftime('%Y%m%d%H%M%S')}/"
json_dir = './Dataset/TextOCR_0.1_train.json'
base_dir = './Dataset/'
batch_size = 4

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Load and Preprocess Data")

modelFrame.markdown('''Load the dataset from a JSON file.''',
                    unsafe_allow_html=True)

code = '''with open(json_dir, 'r') as f:
    data = json.load(f)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Data Generator Class")

modelFrame.markdown('''Define a data generator class to handle data loading and preprocessing.''',
                    unsafe_allow_html=True)

code = '''class TextOCRDataGenerator:
    def __init__(self, data, base_dir, batch_size=4, image_size=(224, 224), set_type='train'):
        self.data = data
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.set_type = set_type
        self.image_ids = [img_id for img_id, img_info in data['imgs'].items() if img_info['set'] == set_type and 'file_name' in img_info]
        self.indexes = np.arange(len(self.image_ids))
        print(f"Initialized with {len(self.image_ids)} images for set '{set_type}'")

    def __len__(self):
        length = len(self.image_ids) // self.batch_size
        print(f"Data generator length (number of batches): {length}")
        return length

    def __data_generation(self, batch_image_ids):
        X = np.empty((self.batch_size, *self.image_size, 3))
        grid_size = 7 # Grid size
        num_boxes = 64 # Number of anchor boxes
        num_coords = 4 # (x1, y1, x2, y2)
        num_classes = 2 # Background and text
        y = np.zeros((self.batch_size, grid_size * grid_size * num_boxes, num_coords + num_classes))

        for i, img_id in enumerate(batch_image_ids):
            img_info = self.data['imgs'][img_id]
            img_path = os.path.join(self.base_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image not found: {img_path}")
                continue
            img = cv2.resize(img, self.image_size)
            img = img / 255.0
            X[i, ] = img

            if img_id in self.data['imgToAnns']:
                for ann_id in self.data['imgToAnns'][img_id]:
                    ann = self.data['anns'][ann_id]
                    x1, y1, x2, y2 = ann['bbox']
                    x1 = x1 / img_info['width']
                    y1 = y1 / img_info['height']
                    x2 = x2 / img_info['width']
                    y2 = y2 / img_info['height']
                    
                    # Check for NaNs and infinities
                    if any(np.isnan([x1, y1, x2, y2])) or any(np.isinf([x1, y1, x2, y2])):
                        print(f"Found NaNs or infinities in the annotation for image {img_id}. Skipping this annotation.")
                        continue
                    
                    # Calculate the cell position
                    cell_x = int((x1 + x2) / 2 * grid_size)
                    cell_y = int((y1 + y2) / 2 * grid_size)
                    cell_index = cell_y * grid_size + cell_x

                    for b in range(num_boxes): # 64 anchor boxes
                        y[i, cell_index * num_boxes + b, :num_coords] = [x1, y1, x2, y2] # bbox coordinates
                        y[i, cell_index * num_boxes + b, num_coords:] = [0, 1] # one-hot class (text)

        return X, y

    def __call__(self):
        for start_idx in range(0, len(self.image_ids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.image_ids))
            batch_image_ids = self.image_ids[start_idx:end_idx]
            yield self.__data_generation(batch_image_ids)

    def clean_data_parallel(self):
        valid_image_ids = []

        def process_image(img_id):
            try:
                X, y = self.__data_generation([img_id])
                if not (np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any()):
                    return img_id
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
            return None

        print("Checking Dataset...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_image, img_id): img_id for img_id in self.image_ids}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    valid_image_ids.append(result)

        return valid_image_ids
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the Model")

modelFrame.markdown('''Create a function to define the model architecture using VGG16 as the base.''',
                    unsafe_allow_html=True)

code = '''def create_model(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x)
    x = Dropout(0.5)(x) # Added dropout for regularization
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x)
    x = Dropout(0.5)(x) # Added dropout for regularization
    
    num_classes = 2  # Background and text
    num_boxes = 64  # 64 anchor boxes
    num_coords = 4  # (x1, y1, x2, y2)
    grid_size = 7  # 7x7 grid

    # Ensure the Conv2D filters match the desired reshape dimensions
    bbox_output = Conv2D(num_boxes * num_coords, (1, 1), activation='linear', name='bbox_output')(x)
    class_output = Conv2D(num_boxes * num_classes, (1, 1), activation='softmax', name='class_output')(x)
    
    # Flatten the outputs before reshaping them
    bbox_output = Flatten()(bbox_output)
    class_output = Flatten()(class_output)

    # Reshape to the desired output shape
    bbox_output = Reshape((grid_size * grid_size * num_boxes, num_coords), name='bbox_output_reshape')(bbox_output)
    class_output = Reshape((grid_size * grid_size * num_boxes, num_classes), name='class_output_reshape')(class_output)

    outputs = Concatenate(axis=-1, name='outputs')([bbox_output, class_output])

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Custom Callback for NaN/Inf Monitoring")

modelFrame.markdown('''Create a custom callback to monitor NaNs/Infs in loss.''',
                    unsafe_allow_html=True)

code = '''class NaNInfMonitor(Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            if np.isnan(logs.get('loss')) or np.isinf(logs.get('loss')):
                print(f"NaN or Inf found in loss at batch {batch}")
                self.model.stop_training = True
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Clean the Data")

modelFrame.markdown('''Instantiate the data generator and clean the data.''',
                    unsafe_allow_html=True)

code = '''data_generator = TextOCRDataGenerator(data, base_dir, batch_size=batch_size, image_size=(224, 224), set_type='train')
clean_image_ids = data_generator.clean_data_parallel()

data_generator.image_ids = clean_image_ids
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Compile the Model")

modelFrame.markdown('''Compile the model with a custom optimizer and loss function.''',
                    unsafe_allow_html=True)

code = '''model = create_model()
optimizer = Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Prepare Data for TensorFlow Dataset")

modelFrame.markdown('''Convert the data generator to a TensorFlow dataset for efficient training.''',
                    unsafe_allow_html=True)

code = '''output_signature = (
    tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(batch_size, 7 * 7 * 64, 6), dtype=tf.float32)
)

tf_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
).repeat().prefetch(tf.data.AUTOTUNE)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Custom Training Loop")

modelFrame.markdown('''Define and execute a custom training loop with NaN/Inf handling.''',
                    unsafe_allow_html=True)

code = '''def custom_training_loop(model, dataset, epochs, steps_per_epoch):
    optimizer = model.optimizer
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    # Initialize lists to store loss and accuracy history
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        
        epoch_loss = []
        epoch_accuracy = []

        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            if step >= steps_per_epoch:
                break
            
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                
                # Check for NaNs and infinities in the loss
                if tf.reduce_any(tf.math.is_nan(loss_value)) or tf.reduce_any(tf.math.is_inf(loss_value)):
                    print(f"NaN or Inf loss detected at step {step} in epoch {epoch}. Skipping this batch.")
                    continue

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch_train, logits)

            # Check for NaNs and infinities in model parameters
            nan_inf_detected = False
            for weight in model.trainable_weights:
                if tf.reduce_any(tf.math.is_nan(weight)) or tf.reduce_any(tf.math.is_inf(weight)):
                    print(f"NaN or Inf detected in model parameters after step {step} in epoch {epoch}.")
                    nan_inf_detected = True
                    break
            
            if nan_inf_detected:
                print(f"Stopping training due to NaN or Inf in model parameters at step {step} in epoch {epoch}.")
                return history

            epoch_loss.append(loss_value.numpy())
            epoch_accuracy.append(train_acc_metric.result().numpy())

            if step % 100 == 0:
                print(f"Training loss (for one batch) at step {step}: {loss_value.numpy()}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        # Compute the mean loss and accuracy for the epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_accuracy = np.mean(epoch_accuracy)

        print(f"Training loss over epoch: {mean_epoch_loss}")
        print(f"Training accuracy over epoch: {mean_epoch_accuracy}")
        
        # Append to history
        history['loss'].append(mean_epoch_loss)
        history['accuracy'].append(mean_epoch_accuracy)

        train_acc_metric.reset_states()

    return history

history = custom_training_loop(model, tf_dataset, epochs=10, steps_per_epoch=len(data_generator))
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Save the Model")

modelFrame.markdown('''Save the trained model to disk.''',
                    unsafe_allow_html=True)

code = '''model.save(f"{MODEL_PATH}/model.h5", save_format="h5")
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Plot Training Results")

modelFrame.markdown('''Visualize the training loss and accuracy over epochs.''',
                    unsafe_allow_html=True)

code = '''def plot_training_results(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_results(history)
'''

modelFrame.code(code, language='python', line_numbers=True)

with open("./ressources/python_code/Text_detection_model.ipynb") as f:
    modelFrame.download_button("Download Notebook", f, "text_detection_model.ipynb")

modelFrame.divider()

modelFrame.header("Training result")

modelFrame.subheader("Model Layers")

modelLayers = pd.read_csv('./ressources/text_detection_model_layers.csv', sep=";", header=0)

modelFrame.dataframe(modelLayers, hide_index=True, use_container_width=True)

modelFrame.markdown('''<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">16,239,168</span> (61.95 MB)
</pre>''', unsafe_allow_html=True)

modelFrame.subheader("Training output")

modelFrame.markdown('''Training of the model was unsuccessfull, even with dataset checking and NaNs and infitines handeling.''',
                    unsafe_allow_html=True)

training_output = modelFrame.container(height=200, border=True)

training_output.text('''Initialized with 21778 images for set 'train'
Checking Dataset...
100%|██████████| 21778/21778 [04:28<00:00, 81.00it/s]
Images batch shape: (4, 224, 224, 3)
Labels batch shape: (4, 3136, 6)
Data generator length (number of batches): 4728

Start of epoch 0
Training loss (for one batch) at step 0: 2.5942912101745605
Seen so far: 4 samples
Training loss (for one batch) at step 100: 2.7891650199890137
Seen so far: 404 samples
Training loss (for one batch) at step 200: 2.814591646194458
Seen so far: 804 samples
Training loss (for one batch) at step 300: 2.7442843914031982
Seen so far: 1204 samples
Training loss (for one batch) at step 400: 1.334847331047058
Seen so far: 1604 samples
Training loss (for one batch) at step 500: 3.178107976913452
Seen so far: 2004 samples
Training loss (for one batch) at step 600: 2.6070315837860107
Seen so far: 2404 samples
Training loss (for one batch) at step 700: 1.9567266702651978
Seen so far: 2804 samples
Training loss (for one batch) at step 800: 1.6604807376861572
Seen so far: 3204 samples
Training loss (for one batch) at step 900: 1.4883748292922974
Seen so far: 3604 samples
Training loss (for one batch) at step 1000: 2.2371127605438232
Seen so far: 4004 samples
Training loss (for one batch) at step 1100: 2.128620147705078
Seen so far: 4404 samples
Training loss (for one batch) at step 1200: 2.0443520545959473
Seen so far: 4804 samples
Training loss (for one batch) at step 1300: 2.3607938289642334
Seen so far: 5204 samples
Training loss (for one batch) at step 1400: 1.5590221881866455
Seen so far: 5604 samples
Training loss (for one batch) at step 1500: 2.0092334747314453
Seen so far: 6004 samples
Training loss (for one batch) at step 1600: 1.1103646755218506
Seen so far: 6404 samples
Training loss (for one batch) at step 1700: 1.7794753313064575
Seen so far: 6804 samples
Training loss (for one batch) at step 1800: 1.9537303447723389
Seen so far: 7204 samples
Training loss (for one batch) at step 1900: 2.664515256881714
Seen so far: 7604 samples
Training loss (for one batch) at step 2000: 1.7767261266708374
Seen so far: 8004 samples
Training loss (for one batch) at step 2100: 2.1291491985321045
Seen so far: 8404 samples
Training loss (for one batch) at step 2200: 1.696402668952942
Seen so far: 8804 samples
Training loss (for one batch) at step 2300: 3.11256742477417
Seen so far: 9204 samples
NaN or Inf detected in model parameters after step 2361 in epoch 0.
Stopping training due to NaN or Inf in model parameters at step 2361 in epoch 0.
''')


output_image = cv2.imread("./ressources/images/text_detection_model_output.png")
modelFrame.image(output_image, channels='BGR', caption="Training history")


#text_detection_model_output.png

=======
import streamlit as st
import pandas as pd
import numpy as np

import cv2

st.set_page_config("NLP App - Text detection model", layout="wide")

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
            img = cv2.resize(image, input_shape[:2])

            new_shape = img.shape # retrieve the new shape of the image

            canvas = np.zeros(input_shape) # Create an 1024x1024 image filled with 0s
            canvas[:new_shape[0], :new_shape[1]] = img # Add the image into the 1024x1024 0s square


            cv2.imwrite(f"./ressources/images/app_output_1.jpg", np.array(canvas))
            output_image.append(cv2.imread("./ressources/images/app_output_1.jpg"))

    return output_image

# ---------------------------------------------------------------------------------------------------

st.title("Text Detection Model")

# -------------------------------------- Testing the model ---------------------------------------------

st.header("Testing the model")

importFrame = st.container()
outputFrame = st.container()
imported_file = importFrame.file_uploader("Import your file(s)", [".jpg", ".png"], True)

col1, col2 = importFrame.columns([1,8])
testFile = col1.button("Test Image", use_container_width=True)
process = col2.button("Process", use_container_width=True)


if process and imported_file or testFile:
    if imported_file == [] or testFile:
        imported_file = [0]
        imported_file[0] = cv2.imread("./ressources/images/20240523_170445.jpg")
        imported_image_channel = 'BGR'
    else:
        imported_image_channel = 'RGB'
    
    statusBar = importFrame.status("Processing...", expanded=False)

    input_images = get_input_model_image(imported_file)

    # Model output

    output_image = cv2.imread("./ressources/images/process_error.jpg")

    statusBar.update(label="Process completed!", state="complete", expanded=True)

    outputFrame.header("Result")
    outputFrame.subheader("First image example")

    # Output the results

    col1, col2, col3 = outputFrame.columns(3)

    col1.image(imported_file[0], channels=imported_image_channel, caption="Imported image")

    col2.image(input_images[0], channels='BGR', caption="Image inpputed into the model")

    col3.image(output_image, channels='BGR', caption="Model output")

elif process:
    importFrame.error("You must import at least one file!")

# ------------------------------------------- Dataset ---------------------------------------------------

datasetFrame = st.container()

datasetFrame.divider()

datasetFrame.header("Dataset")

datasetFrame.markdown(
    '''<div class="jss148 jss187 jss215"><h6 class="jss58 jss76 jss80"><b>General Information</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">Data is available under <a class="jss58 jss88 jss93 jss53 jss55" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> license.</h6></li><li><h6 class="jss58 jss76 jss80">Numbers in the papers should be reported on v0.1 test set. We will soon host a challenge on that.</h6></li><li><h6 class="jss58 jss76 jss80">Reach us out at <a class="jss58 jss88 jss93 jss53 jss55" href="mailto:textvqa@fb.com">textvqa@fb.com</a> for any questions, suggestions and feedback.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Images</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">Images for training and validation set are from OpenImages train set while images for test set are from OpenImages test set.</h6></li><li><h6 class="jss58 jss76 jss80">Validation set's images are contained in the zip for training set's images. The OpenImages dataset can be downloaded from <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/web/download.html">here</a>.</h6></li><li><h6 class="jss58 jss76 jss80"><b>Note:</b> Some of the images in OpenImages are rotated, please make sure to check the <b>Rotation</b> field in the Image IDs files for <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv">train</a> and <a class="jss58 jss88 jss93 jss53 jss55" href="https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv">test</a>.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Dataset Format</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">The json format mostly follows COCO-Text v2, except the "mask" field in "anns" is named as "points" for the polygon annotation.</h6></li><li><h6 class="jss58 jss76 jss80">The "points" field is a list of 2D coordinates like [x1, y1, x2, y2, ...]. Note that (x1,y1) is always the top-left corner of the text (in its own orientation), and order of the points is clockwise (for example, for horizontal text, (x1, y1) and (x2, y2) will form the top line).</h6></li><li><h6 class="jss58 jss76 jss80">The "bbox" field contains horizontal box converted from "points" for convenience, and "area" is computed based on width and height of "bbox". For any conversion to other formats such as rotated boxes or quadrilaterals, "points" should be used as the source of truth.</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Annotation Details</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">In cases when the text is illegible or not in English, polygon is annotated normally but word will be annotated as a single "." symbol.</h6></li><li><h6 class="jss58 jss76 jss80">Word annotations are case-sensitive, and can contain punctuations too.</h6></li><li><h6 class="jss58 jss76 jss80">The annotators were instructed to draw exactly 4 points (quadrilaterals) whenever possible, and only draw more than 4 points when necessary (for cases like curved text).</h6></li></ul><h6 class="jss58 jss76 jss80"><b>Relationship with TextVQA/TextCaps</b></h6><ul class="jss143"><li><h6 class="jss58 jss76 jss80">The image ids in TextOCR match the ids in TextVQA.</h6></li><li><h6 class="jss58 jss76 jss80">train/val/test splits are the same as TextVQA/TextCaps. However due to privacy reasons, we removed 274 images from TextVQA while creating TextOCR.</h6></li></ul></div>''',
    unsafe_allow_html=True
    )

datasetFrame.page_link("https://textvqa.org/textocr/dataset/", label="Download the dataset")

# -------------------------------------- Fine tuning the model ---------------------------------------------

modelFrame = st.container()

modelFrame.divider()

modelFrame.header("Fine tuning the model")

modelFrame.subheader("Import the librairies")

modelFrame.markdown('''<p>We start by importing the necessary libraries, including TensorFlow and Keras for model building, and other utilities for data handling and visualization.</p>''',
                    unsafe_allow_html=True)

code = '''
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tqdm import tqdm

from datetime import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import concurrent.futures
import json
import cv2
import os
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Configure GPU (Optional)")

modelFrame.markdown('''<p>Ensure GPU memory growth is set if needed.</p>''',
                    unsafe_allow_html=True)

code = '''# Ensure GPU memory growth is set if needed
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define Paths and Constants")

modelFrame.markdown('''<p>Set the paths for the dataset and model saving directory. Initialize other constants like batch size.</p>''',
                    unsafe_allow_html=True)

code = '''MODEL_PATH = f"./Models/{dt.now().strftime('%Y%m%d%H%M%S')}/"
json_dir = './Dataset/TextOCR_0.1_train.json'
base_dir = './Dataset/'
batch_size = 4

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Load and Preprocess Data")

modelFrame.markdown('''Load the dataset from a JSON file.''',
                    unsafe_allow_html=True)

code = '''with open(json_dir, 'r') as f:
    data = json.load(f)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Data Generator Class")

modelFrame.markdown('''Define a data generator class to handle data loading and preprocessing.''',
                    unsafe_allow_html=True)

code = '''class TextOCRDataGenerator:
    def __init__(self, data, base_dir, batch_size=4, image_size=(224, 224), set_type='train'):
        self.data = data
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.set_type = set_type
        self.image_ids = [img_id for img_id, img_info in data['imgs'].items() if img_info['set'] == set_type and 'file_name' in img_info]
        self.indexes = np.arange(len(self.image_ids))
        print(f"Initialized with {len(self.image_ids)} images for set '{set_type}'")

    def __len__(self):
        length = len(self.image_ids) // self.batch_size
        print(f"Data generator length (number of batches): {length}")
        return length

    def __data_generation(self, batch_image_ids):
        X = np.empty((self.batch_size, *self.image_size, 3))
        grid_size = 7 # Grid size
        num_boxes = 64 # Number of anchor boxes
        num_coords = 4 # (x1, y1, x2, y2)
        num_classes = 2 # Background and text
        y = np.zeros((self.batch_size, grid_size * grid_size * num_boxes, num_coords + num_classes))

        for i, img_id in enumerate(batch_image_ids):
            img_info = self.data['imgs'][img_id]
            img_path = os.path.join(self.base_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image not found: {img_path}")
                continue
            img = cv2.resize(img, self.image_size)
            img = img / 255.0
            X[i, ] = img

            if img_id in self.data['imgToAnns']:
                for ann_id in self.data['imgToAnns'][img_id]:
                    ann = self.data['anns'][ann_id]
                    x1, y1, x2, y2 = ann['bbox']
                    x1 = x1 / img_info['width']
                    y1 = y1 / img_info['height']
                    x2 = x2 / img_info['width']
                    y2 = y2 / img_info['height']
                    
                    # Check for NaNs and infinities
                    if any(np.isnan([x1, y1, x2, y2])) or any(np.isinf([x1, y1, x2, y2])):
                        print(f"Found NaNs or infinities in the annotation for image {img_id}. Skipping this annotation.")
                        continue
                    
                    # Calculate the cell position
                    cell_x = int((x1 + x2) / 2 * grid_size)
                    cell_y = int((y1 + y2) / 2 * grid_size)
                    cell_index = cell_y * grid_size + cell_x

                    for b in range(num_boxes): # 64 anchor boxes
                        y[i, cell_index * num_boxes + b, :num_coords] = [x1, y1, x2, y2] # bbox coordinates
                        y[i, cell_index * num_boxes + b, num_coords:] = [0, 1] # one-hot class (text)

        return X, y

    def __call__(self):
        for start_idx in range(0, len(self.image_ids), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.image_ids))
            batch_image_ids = self.image_ids[start_idx:end_idx]
            yield self.__data_generation(batch_image_ids)

    def clean_data_parallel(self):
        valid_image_ids = []

        def process_image(img_id):
            try:
                X, y = self.__data_generation([img_id])
                if not (np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any()):
                    return img_id
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
            return None

        print("Checking Dataset...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_image, img_id): img_id for img_id in self.image_ids}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    valid_image_ids.append(result)

        return valid_image_ids
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the Model")

modelFrame.markdown('''Create a function to define the model architecture using VGG16 as the base.''',
                    unsafe_allow_html=True)

code = '''def create_model(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x)
    x = Dropout(0.5)(x) # Added dropout for regularization
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=he_normal())(x)
    x = Dropout(0.5)(x) # Added dropout for regularization
    
    num_classes = 2  # Background and text
    num_boxes = 64  # 64 anchor boxes
    num_coords = 4  # (x1, y1, x2, y2)
    grid_size = 7  # 7x7 grid

    # Ensure the Conv2D filters match the desired reshape dimensions
    bbox_output = Conv2D(num_boxes * num_coords, (1, 1), activation='linear', name='bbox_output')(x)
    class_output = Conv2D(num_boxes * num_classes, (1, 1), activation='softmax', name='class_output')(x)
    
    # Flatten the outputs before reshaping them
    bbox_output = Flatten()(bbox_output)
    class_output = Flatten()(class_output)

    # Reshape to the desired output shape
    bbox_output = Reshape((grid_size * grid_size * num_boxes, num_coords), name='bbox_output_reshape')(bbox_output)
    class_output = Reshape((grid_size * grid_size * num_boxes, num_classes), name='class_output_reshape')(class_output)

    outputs = Concatenate(axis=-1, name='outputs')([bbox_output, class_output])

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Custom Callback for NaN/Inf Monitoring")

modelFrame.markdown('''Create a custom callback to monitor NaNs/Infs in loss.''',
                    unsafe_allow_html=True)

code = '''class NaNInfMonitor(Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            if np.isnan(logs.get('loss')) or np.isinf(logs.get('loss')):
                print(f"NaN or Inf found in loss at batch {batch}")
                self.model.stop_training = True
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Clean the Data")

modelFrame.markdown('''Instantiate the data generator and clean the data.''',
                    unsafe_allow_html=True)

code = '''data_generator = TextOCRDataGenerator(data, base_dir, batch_size=batch_size, image_size=(224, 224), set_type='train')
clean_image_ids = data_generator.clean_data_parallel()

data_generator.image_ids = clean_image_ids
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Compile the Model")

modelFrame.markdown('''Compile the model with a custom optimizer and loss function.''',
                    unsafe_allow_html=True)

code = '''model = create_model()
optimizer = Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
'''
modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Prepare Data for TensorFlow Dataset")

modelFrame.markdown('''Convert the data generator to a TensorFlow dataset for efficient training.''',
                    unsafe_allow_html=True)

code = '''output_signature = (
    tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(batch_size, 7 * 7 * 64, 6), dtype=tf.float32)
)

tf_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
).repeat().prefetch(tf.data.AUTOTUNE)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Custom Training Loop")

modelFrame.markdown('''Define and execute a custom training loop with NaN/Inf handling.''',
                    unsafe_allow_html=True)

code = '''def custom_training_loop(model, dataset, epochs, steps_per_epoch):
    optimizer = model.optimizer
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    # Initialize lists to store loss and accuracy history
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        
        epoch_loss = []
        epoch_accuracy = []

        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            if step >= steps_per_epoch:
                break
            
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                
                # Check for NaNs and infinities in the loss
                if tf.reduce_any(tf.math.is_nan(loss_value)) or tf.reduce_any(tf.math.is_inf(loss_value)):
                    print(f"NaN or Inf loss detected at step {step} in epoch {epoch}. Skipping this batch.")
                    continue

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y_batch_train, logits)

            # Check for NaNs and infinities in model parameters
            nan_inf_detected = False
            for weight in model.trainable_weights:
                if tf.reduce_any(tf.math.is_nan(weight)) or tf.reduce_any(tf.math.is_inf(weight)):
                    print(f"NaN or Inf detected in model parameters after step {step} in epoch {epoch}.")
                    nan_inf_detected = True
                    break
            
            if nan_inf_detected:
                print(f"Stopping training due to NaN or Inf in model parameters at step {step} in epoch {epoch}.")
                return history

            epoch_loss.append(loss_value.numpy())
            epoch_accuracy.append(train_acc_metric.result().numpy())

            if step % 100 == 0:
                print(f"Training loss (for one batch) at step {step}: {loss_value.numpy()}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        # Compute the mean loss and accuracy for the epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_accuracy = np.mean(epoch_accuracy)

        print(f"Training loss over epoch: {mean_epoch_loss}")
        print(f"Training accuracy over epoch: {mean_epoch_accuracy}")
        
        # Append to history
        history['loss'].append(mean_epoch_loss)
        history['accuracy'].append(mean_epoch_accuracy)

        train_acc_metric.reset_states()

    return history

history = custom_training_loop(model, tf_dataset, epochs=10, steps_per_epoch=len(data_generator))
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Save the Model")

modelFrame.markdown('''Save the trained model to disk.''',
                    unsafe_allow_html=True)

code = '''model.save(f"{MODEL_PATH}/model.h5", save_format="h5")
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Plot Training Results")

modelFrame.markdown('''Visualize the training loss and accuracy over epochs.''',
                    unsafe_allow_html=True)

code = '''def plot_training_results(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_results(history)
'''

modelFrame.code(code, language='python', line_numbers=True)

with open("./ressources/python_code/Text_detection_model.ipynb") as f:
    modelFrame.download_button("Download Notebook", f, "text_detection_model.ipynb")

modelFrame.divider()

modelFrame.header("Training result")

modelFrame.subheader("Model Layers")

modelLayers = pd.read_csv('./ressources/text_detection_model_layers.csv', sep=";", header=0)

modelFrame.dataframe(modelLayers, hide_index=True, use_container_width=True)

modelFrame.markdown('''<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">16,239,168</span> (61.95 MB)
</pre>''', unsafe_allow_html=True)

modelFrame.subheader("Training output")

modelFrame.markdown('''Training of the model was unsuccessfull, even with dataset checking and NaNs and infitines handeling.''',
                    unsafe_allow_html=True)

training_output = modelFrame.container(height=200, border=True)

training_output.text('''Initialized with 21778 images for set 'train'
Checking Dataset...
100%|██████████| 21778/21778 [04:28<00:00, 81.00it/s]
Images batch shape: (4, 224, 224, 3)
Labels batch shape: (4, 3136, 6)
Data generator length (number of batches): 4728

Start of epoch 0
Training loss (for one batch) at step 0: 2.5942912101745605
Seen so far: 4 samples
Training loss (for one batch) at step 100: 2.7891650199890137
Seen so far: 404 samples
Training loss (for one batch) at step 200: 2.814591646194458
Seen so far: 804 samples
Training loss (for one batch) at step 300: 2.7442843914031982
Seen so far: 1204 samples
Training loss (for one batch) at step 400: 1.334847331047058
Seen so far: 1604 samples
Training loss (for one batch) at step 500: 3.178107976913452
Seen so far: 2004 samples
Training loss (for one batch) at step 600: 2.6070315837860107
Seen so far: 2404 samples
Training loss (for one batch) at step 700: 1.9567266702651978
Seen so far: 2804 samples
Training loss (for one batch) at step 800: 1.6604807376861572
Seen so far: 3204 samples
Training loss (for one batch) at step 900: 1.4883748292922974
Seen so far: 3604 samples
Training loss (for one batch) at step 1000: 2.2371127605438232
Seen so far: 4004 samples
Training loss (for one batch) at step 1100: 2.128620147705078
Seen so far: 4404 samples
Training loss (for one batch) at step 1200: 2.0443520545959473
Seen so far: 4804 samples
Training loss (for one batch) at step 1300: 2.3607938289642334
Seen so far: 5204 samples
Training loss (for one batch) at step 1400: 1.5590221881866455
Seen so far: 5604 samples
Training loss (for one batch) at step 1500: 2.0092334747314453
Seen so far: 6004 samples
Training loss (for one batch) at step 1600: 1.1103646755218506
Seen so far: 6404 samples
Training loss (for one batch) at step 1700: 1.7794753313064575
Seen so far: 6804 samples
Training loss (for one batch) at step 1800: 1.9537303447723389
Seen so far: 7204 samples
Training loss (for one batch) at step 1900: 2.664515256881714
Seen so far: 7604 samples
Training loss (for one batch) at step 2000: 1.7767261266708374
Seen so far: 8004 samples
Training loss (for one batch) at step 2100: 2.1291491985321045
Seen so far: 8404 samples
Training loss (for one batch) at step 2200: 1.696402668952942
Seen so far: 8804 samples
Training loss (for one batch) at step 2300: 3.11256742477417
Seen so far: 9204 samples
NaN or Inf detected in model parameters after step 2361 in epoch 0.
Stopping training due to NaN or Inf in model parameters at step 2361 in epoch 0.
''')


output_image = cv2.imread("./ressources/images/text_detection_model_output.png")
modelFrame.image(output_image, channels='BGR', caption="Training history")


#text_detection_model_output.png

>>>>>>> b8509ba0de058feb6b6bfeb1473132e18125ecb8
