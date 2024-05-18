# # Use a pipeline as a high-level helper
import gradio as gr
import pandas as pd

import gradio as gr
from transformers import pipeline



model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
analyzer = pipeline("text-classification", model=model_name)




def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def read_reviews_and_analyze_sentiment(file_path:str) -> pd.DataFrame:
    # read file using pd
    df = pd.read_excel(file_path)
    # check if reviews is in columns
    if not 'Reviews' in df.columns:
        return ValueError('Review column must be in excel file')
    # create a new column for sentiment
    df['sentiment'] = df['Reviews'].apply(sentiment_analyzer)

    return df



gr.close_all()

demo = gr.Interface(fn=read_reviews_and_analyze_sentiment, 
                    inputs=[gr.File(file_types=['.xlsx'],label="Takes an excel file and analyze it")],
                    outputs=[gr.Dataframe(label='SHows a data frame')]
                    )

demo.launch()