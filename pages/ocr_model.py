from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs

import streamlit as st
import pandas as pd
import numpy as np

import typing
import cv2


st.set_page_config("NLP App - OCR model", layout="wide")


with st.sidebar:
    st.title("NLP Project")
    st.page_link("app.py", label="App", icon="❌")
    st.page_link("./pages/text_detection_model.py", label="Text detection model", icon="❌")
    st.page_link("./pages/ocr_model.py", label="OCR model", icon="✔️")
    st.page_link("./pages/context_extraction_model.py", label="Context extraction", icon="✔️")

st.title("OCR Model")

# ----------------------------------------------------------------------------------------------------

input_shape = (128, 32, 3)

# -------------------------------------- Testing the model ---------------------------------------------

# -------------------------------------- Run model code ------------------------------------------------

def get_input_model_image(imported_image):
    try:
        file_bytes = np.asarray(bytearray(imported_image.read()), dtype=np.uint8)
        imported_image = cv2.imdecode(file_bytes, 1)
    except:
        pass
    finally:
        #Resize the image to a 1024x0124 square
        img = cv2.resize(imported_image, input_shape[:2])

        cv2.imwrite(f"./ressources/images/app_output_2.jpg", np.array(img))

        return cv2.imread("./ressources/images/app_output_2.jpg")


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

configs = BaseModelConfigs.load("./ressources/Models/OCR/configs.yaml")

model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)


# ---------------------------------------- Vizualizing ----------------------------------------------------

st.header("Testing the model")

importFrame = st.container()
outputFrame = st.container()

imported_file = importFrame.file_uploader("Import your file", [".jpg", ".png"], False)

testFile = importFrame.button("Test Image")

if imported_file is not None or testFile:
    if imported_file is None:
        imported_file = cv2.imread("./ressources/images/13_disputation_22583.jpg")

    outputFrame.subheader("Result")

    try:       
        col1, col2 = outputFrame.columns(2)

        col1.image(imported_file, caption="Imported image")

        input_image = get_input_model_image(imported_file)

        col2.image(input_image, channels='BGR', caption="Image inpputed into the model (zero-padded)")

        prediction_text = model.predict(input_image)
        outputFrame.success(f"Word predicted : {prediction_text}")
    except:
        outputFrame.error("Couldn't find a word")

# ------------------------------------------- Dataset ---------------------------------------------------

datasetFrame = st.container()

datasetFrame.divider()

datasetFrame.header("Dataset")

datasetFrame.subheader("Synthetic Word Dataset")

datasetFrame.markdown(
    '''<p>The MJSynth dataset consists of <b>9 million images</b> covering <b>90k English words</b>, and includes the training, validation and test splits used in our work.</p><p>This dataset is around 10Gb.</p>''',
    unsafe_allow_html=True
    )

datasetFrame.image(cv2.imread("./ressources/images/synthflow.png"))

datasetFrame.page_link("https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth", label="Download the dataset")

datasetFrame.markdown(
    '''<h4>Citation</h4><p><i>Workshop on Deep Learning, NIPS</i>, Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman, 2014</p>''',
    unsafe_allow_html=True
    )

# -------------------------------------- Creating the model ---------------------------------------------

modelFrame = st.container()

modelFrame.divider()

modelFrame.header("Creation of the model")

modelFrame.subheader("Define the model configuration")

code = '''import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("./Models/OCR/", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.height = 32
        self.width = 128
        self.max_text_length = 23
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = 20'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Building the model")

code = '''from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):

    inputs = layers.Input(shape=input_dim, name="input")

    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(squeezed)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Training the model")

code = '''import os
from tqdm import tqdm
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric


configs = ModelConfigs()

data_path = "./Datasets/90kDICT32px"
val_annotation_path = data_path + "/annotation_val.txt"
train_annotation_path = data_path + "/annotation_train.txt"

# Read metadata file and parse it
def read_annotation_file(annotation_path):
    dataset, vocab, max_len = [], set(), 0
    with open(annotation_path, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.split()
            image_path = data_path + line[0][1:]
            label = line[0].split("_")[1]
            dataset.append([image_path, label])
            vocab.update(list(label))
            max_len = max(max_len, len(label))
    return dataset, sorted(vocab), max_len

train_dataset, train_vocab, max_train_len = read_annotation_file(train_annotation_path)
val_dataset, val_vocab, max_val_len = read_annotation_file(val_annotation_path)

# Save vocab and maximum text length to configs
configs.vocab = "".join(train_vocab)
configs.max_text_length = max(max_train_len, max_val_len)
configs.save()

# Create training data provider
train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

# Create validation data provider
val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)

model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)
# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(1)],
    run_eagerly=False
)
model.summary(line_length=110)

# Define path to save the model
os.makedirs(configs.model_path, exist_ok=True)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=10, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.header("Model Layers")

modelLayers = pd.read_csv('./ressources/ocr_layers.csv', sep=";", header=0)

modelFrame.dataframe(modelLayers, hide_index=True, use_container_width=True)
modelFrame.markdown('''<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">252,799</span> (987.50 KB)
</pre>''', unsafe_allow_html=True)

modelFrame.subheader("Training result")

col1, col2 = modelFrame.columns(2)

chart_data = pd.read_csv('./ressources/ocr_loss_results.csv', sep=";", header=0)
col1.line_chart(chart_data, x="Epoch", y=["train_loss", "val_loss"])

chart_data = pd.read_csv('./ressources/ocr_CER_results.csv', sep=";", header=0)
col2.line_chart(chart_data, x="Epoch", y=["train_CER", "val_CER"])

modelFrame.markdown('''<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Training time: </span><span style="color: #00af00; text-decoration-color: #00af00">32h</span> (with NVIDIA GTX 970)
</pre>''', unsafe_allow_html=True)
