import streamlit as st
from transformers import pipeline

model = pipeline("sentiment-analysis")

def main():
    st.title(" Sentiment Analysis Model ")
    
    # Create an input text box
    input_text = st.text_input("Enter your text", "")

    # Create a button to trigger model inference
    if st.button("Analyze"):
        # Perform inference using the loaded model
        result = model(input_text)
        st.write("Prediction:", result[0]['label'], "| Score:", result[0]['score'])

if __name__ == "__main__":
    main()

    
    