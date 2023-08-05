import streamlit as st
import tensorflow as tf
import re
import string
import preprocessor as p
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

st.markdown(
    """
    <style>
    body {
        background-color: #f2f2f2;  /* Ganti dengan warna latar belakang yang diinginkan */
    }
    </style>
    """,
    unsafe_allow_html=True
)

tokenizer = AutoTokenizer.from_pretrained('itsam26/indobert-indonesia-sexual')
model = TFAutoModelForSequenceClassification.from_pretrained('itsam26/indobert-indonesia-sexual')

pattern = r'[0-9]'

def preprocess_text(text):
    for punctuation in string.punctuation:
        text = p.clean(text) #menghapus tag, hashtag
        text = re.sub(r'http[s]?://\S+','',text) #menghapus URL
        text = text.replace(punctuation, '') #menghapus tanda baca
        text = re.sub(pattern, '', text)#menghapus angka
        text = re.sub(r'\r?\n|\r','',text)#menghapus baris baru
        text = text.lower() #mengubah ke huruf kecil (case folding)
    return text

# Define the Streamlit app
def main():
    # Set app title
    label_map = {
        0: "bukan pelecehan seksual",
        1: "pelecehan seksual"
    }
    st.title("Text Classification IndoBERT")

    # Create input text box
    input_text = st.text_input("Enter a sentence", "")

    # Perform inference when the user submits a text
    if st.button("Predict"):
        # Tokenize the input text
        # Tokenize the input text
        input_text = preprocess_text(input_text)
        inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="tf")

        # Get the input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Reshape the input tensors to match the expected shape
        input_ids = tf.reshape(input_ids, (1, -1))
        attention_mask = tf.reshape(attention_mask, (1, -1))

        # Perform the inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = tf.argmax(outputs.logits, axis=-1).numpy()[0]
        predicted_label = label_map[predicted_class]

        # Display the predicted label
        st.write("Predicted label:", predicted_label)
        
# Run the app
if __name__ == "__main__":
    main()
