# Python In-built packages
from pathlib import Path
import PIL
from ultralytics import YOLO
import os
from os import listdir
from os.path import isfile, join

# External packages
import streamlit as st

# Local Modules
import settings

# Setting page layout
st.set_page_config(
    page_title="Определение наличия дефектов сварных швов",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Определение наличия дефектов сварных швов")

# Sidebar
st.sidebar.header("Настройка модели")

confidence = float(st.sidebar.slider(
    "Чувствительность (confidence), %", 25, 100, 40)) / 100

model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
def load_model(model_path):
    model = YOLO(model_path)
    return model
    
try:
    model = load_model(model_path)
except Exception as ex:
    st.error(f"Не удалось загрузить модель. Проверьте путь к модели: {model_path}")
    st.error(ex)

st.sidebar.header("Изображения")

demo_files = sorted([os.path.join(settings.IMAGES_DIR, f) for f in os.listdir(settings.IMAGES_DIR) if 
    os.path.isfile(os.path.join(settings.IMAGES_DIR, f))])
    
demo_files_cnt = int(st.sidebar.slider("Количество демо-файлов", 1, len(demo_files), 1)) 

source_imgs = None
# If image is selected

source_imgs = st.sidebar.file_uploader(
    "Загрузите файлы...", type=("jpg", "jpeg"), accept_multiple_files=True)

col1, col2 = st.columns(2)

uploaded_images= []
with col1:
    try:
        caption= "Изображения по умолчанию"
        if source_imgs==None or len(source_imgs)==0:
            for demo_file in demo_files[:demo_files_cnt]: 
                source_imgs.append(open(demo_file, 'rb'))
        else:
            caption="Загруженные файлы"
            
        for source_img in source_imgs:
            img= PIL.Image.open(source_img)
            uploaded_images.append(img)
            caption= source_img.name
            st.image(img, caption=caption, use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    for uploaded_image in uploaded_images:
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        caption= 'Дефектов не обнаружено'
        if len(boxes)>0:
            caption=f'Обнаружено дефектов: {str(len(boxes))}'
        st.image(res_plotted, caption= caption,
                 use_column_width=True)



      
