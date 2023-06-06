import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Define the available models
models = {
    "BERT": {
        "tokenizer": AutoTokenizer.from_pretrained('itsam26/bert-indonesia-sexual'),
        "model": TFAutoModelForSequenceClassification.from_pretrained('itsam26/bert-indonesia-sexual')
    },
    "RoBERTa": {
        "tokenizer": AutoTokenizer.from_pretrained('itsam26/roberta-indonesia-sexual'),
        "model": TFAutoModelForSequenceClassification.from_pretrained('itsam26/roberta-indonesia-sexual')
    },
    "IndoBERT": {
        "tokenizer": AutoTokenizer.from_pretrained('itsam26/indobert-indonesia-sexual'),
        "model": TFAutoModelForSequenceClassification.from_pretrained('itsam26/indobert-indonesia-sexual')
    },
    "XLNet": {
        "tokenizer": AutoTokenizer.from_pretrained('itsam26/xlnet-indonesia-sexual'),
        "model": TFAutoModelForSequenceClassification.from_pretrained('itsam26/xlnet-indonesia-sexual')
    },
    "IndoXLNet": {
        "tokenizer": AutoTokenizer.from_pretrained('itsam26/indoxlnet-indonesia-sexual'),
        "model": TFAutoModelForSequenceClassification.from_pretrained('itsam26/indoxlnet-indonesia-sexual')
    }
}

# Define the Streamlit app
def main():
    # Set app title
    label_map = {
        0: "bukan pelecehan seksual",
        1: "pelecehan seksual"
    }
    st.title("Text Classification Demo")

    # Create model selection dropdown
    model_name = st.selectbox("Select a model", list(models.keys()))

    # Get the selected model
    selected_model = models[model_name]

    # Create input text box
    input_text = st.text_input("Enter a sentence", "")

    # Perform inference when the user submits a text
    if st.button("Predict"):
        # Tokenize the input text
        # Tokenize the input text
        inputs = selected_model['tokenizer'].encode_plus(input_text, add_special_tokens=True, return_tensors="tf")

        # Get the input tensors
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Reshape the input tensors to match the expected shape
        input_ids = tf.reshape(input_ids, (1, -1))
        attention_mask = tf.reshape(attention_mask, (1, -1))

        # Perform the inference
        outputs = selected_model['model'](input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = tf.argmax(outputs.logits, axis=-1).numpy()[0]
        predicted_label = label_map[predicted_class]

        # Display the predicted label
        st.write("Predicted label:", predicted_label)
        
# Run the app
if __name__ == "__main__":
    main()
