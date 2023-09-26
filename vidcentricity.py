import json
import random
import sounddevice as sd
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModel
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import TextClip, AudioFileClip, ImageClip
import textwrap
import numpy as np  # Add this import at the top

def generate_wrapped_text_image(text, img_size, font_size=50):
    img = Image.new('RGB', img_size, 'black')
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Using default font
    wrapped_text = textwrap.fill(text, width=30)
    text_size = d.multiline_textsize(wrapped_text, font=font)
    position = ((img_size[0] - text_size[0]) // 2, (img_size[1] - text_size[1]) // 2)
    d.multiline_text(position, wrapped_text, fill=(255, 255, 255), font=font)
    return img

def generate_mp4_from_wav_and_text(wav_filename, text):
    audio_path = wav_filename
    img = generate_wrapped_text_image(text, (640, 480))
    img_np = np.array(img)  # Convert PIL Image to numpy array
    img_clip = ImageClip(img_np, ismask=False, transparent=False).set_duration(AudioFileClip(audio_path).duration)
    video = img_clip.set_audio(AudioFileClip(audio_path))
    
    mp4_filename = wav_filename.replace(".wav", ".mp4")
    video.write_videofile(mp4_filename, codec='libx264', fps=24)
    return mp4_filename

def generate_speech(text, processor, model):
    inputs = processor(
        text=[text], 
        voice_preset="v2/en_speaker_9", 
        return_tensors="pt", 
        padding=True,  # Add padding
        truncation=True,  # Add truncation to avoid super long text messing things up
        max_length=512  # Set a max_length if the model has one
    )
    
    speech_values = model.generate(
        input_values=inputs.input_values, 
        attention_mask=inputs.attention_mask, 
        max_length=512,  # Again, assuming the model's max_length is 512, modify as needed
        do_sample=True
    )
    audio_data = speech_values.cpu().numpy().squeeze()
    
    first_word = text.split(" ")[0]
    random_number = random.randint(1000, 9999)
    wav_filename = f"{first_word}{random_number}.wav"
    scipy.io.wavfile.write(wav_filename, rate=24000, data=audio_data)

    sd.play(audio_data, samplerate=24000)
    sd.wait()

    mp4_filename = generate_mp4_from_wav_and_text(wav_filename, text)
    return text, wav_filename, mp4_filename

def main():
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")
    
    try:
        with open("corpus.json", "r") as f:
            corpus = json.load(f)
        print("Loaded existing corpus:", corpus)
    except FileNotFoundError:
        corpus = []

    while True:
        user_input = input("Enter text to speak (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        # Tokenize the text to check its length in terms of tokens
        inputs = processor(text=[user_input], voice_preset="v2/en_speaker_9", return_tensors="pt")
        input_ids = inputs.input_ids[0]

        max_tokens = 512  # Maximum number of tokens
        if len(input_ids) > max_tokens:
            # Break the input_ids into smaller lists of tokens each with max length of 512
            token_chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]

            for chunk in token_chunks:
                # Decode tokens to text string
                chunk_text = processor.batch_decode(chunk, skip_special_tokens=True)[0]
                text, wav_filename, mp4_filename = generate_speech(chunk_text, processor, model)
                corpus.append({"text": text, "wav_filename": wav_filename, "mp4_filename": mp4_filename})

        else:
            text, wav_filename, mp4_filename = generate_speech(user_input, processor, model)
            corpus.append({"text": text, "wav_filename": wav_filename, "mp4_filename": mp4_filename})

        with open("corpus.json", "w") as f:
            json.dump(corpus, f)


if __name__ == "__main__":
    main()
