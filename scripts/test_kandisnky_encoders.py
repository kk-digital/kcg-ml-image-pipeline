import os
import sys

sys.path.insert(0, os.getcwd())
from kandinsky.models.clip_text_encoder.clip_text_encoder import KandinskyCLIPTextEmbedder

def main():
    text_encoder=KandinskyCLIPTextEmbedder()
    text_encoder.load_submodels()

    texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
    ]

    prompt_embeds, text_encoder_hidden_states, text_mask= text_encoder.forward(texts)

    print(f"prompt embedding: {prompt_embeds.shape}")
    print(f"hidden state {text_encoder_hidden_states.shape}")
    print(f"mask {text_mask.shape}")

if __name__ == '__main__':
    main()