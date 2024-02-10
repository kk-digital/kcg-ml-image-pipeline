import os
import sys
from PIL import Image

sys.path.insert(0, os.getcwd())
from kandinsky.models.clip_text_encoder.clip_text_encoder import KandinskyCLIPTextEmbedder
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder

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
 
    image_encoder=KandinskyCLIPImageEncoder()
    image_encoder.load_submodels()

    image= Image.open("test_image.jpg")

    image_features= image_encoder.forward(image)

    print(f"image embedding: {image_features}")

if __name__ == '__main__':
    main()