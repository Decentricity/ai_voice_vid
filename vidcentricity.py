import json
import random
import sounddevice as sd
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModel
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import TextClip, AudioFileClip, ImageClip
import textwrap
import numpy as np  # Add this import at the top
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize 
from moviepy.editor import VideoFileClip, concatenate_videoclips

def chunk_sentences(text):
    paragraphs = text.split('\n')
    chunks = []

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        current_chunk = ""
        
        for i in range(len(sentences)):
            if len(current_chunk.split()) < 15:  # Consider a sentence 'short' if it has fewer than 15 words
                current_chunk += " " + sentences[i]
            else:
                if current_chunk.strip():  # Only add non-empty chunks
                    chunks.append(current_chunk.strip())
                current_chunk = sentences[i]
            
            # Edge case for the last sentence in a paragraph
            if i == len(sentences) - 1:
                if current_chunk.strip():  # Only add non-empty chunks
                    chunks.append(current_chunk.strip())
                    
    # Remove chunks that are just symbols (no alphabets or numbers)
    chunks = [chunk for chunk in chunks if any(c.isalnum() for c in chunk)]

    return chunks

def generate_wrapped_text_image(text, img_size, font_size=30):  # Changed default font_size to 30
    img = Image.new('RGB', img_size, 'black')
    d = ImageDraw.Draw(img)
    
    # Use the specified TrueType font now.
    try:
        font = ImageFont.truetype("./agave-r-autohinted.ttf", font_size)
    except IOError:
        print("TrueType Font not found, using default PIL font.")
        font = ImageFont.load_default()

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
    inputs = processor(text=[text], voice_preset="v2/en_speaker_9", return_tensors="pt")
    speech_values = model.generate(**inputs, do_sample=True)
    audio_data = speech_values.cpu().numpy().squeeze()

    first_word = text.split(" ")[0]
    random_number = random.randint(1000, 9999)
    wav_filename = f"{first_word}{random_number}.wav"
    scipy.io.wavfile.write(wav_filename, rate=24000, data=audio_data)

    sd.play(audio_data, samplerate=24000)
    sd.wait()

    mp4_filename = generate_mp4_from_wav_and_text(wav_filename, text)
    return text, wav_filename, mp4_filename

import argparse  

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate speech from text.")
    parser.add_argument("file", nargs="?", type=str, help="Text file to read input from instead of stdin.")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    try:
        with open("corpus.json", "r") as f:
            corpus = json.load(f)
        print("Loaded existing corpus:", corpus)
    except FileNotFoundError:
        corpus = []

    if args.file:  # If a filename was provided
        with open(args.file, "r") as f:
            user_input = f.read()
        process_text(user_input, processor, model, corpus)
        return  # Exit after processing the file
    else:
        while True:
            user_input = input("Enter text to speak (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            process_text(user_input, processor, model, corpus)

def process_text(user_input, processor, model, corpus):
    chunks = chunk_sentences(user_input)
    mp4_filenames = []

    for chunk in chunks:
        text, wav_filename, mp4_filename = generate_speech(chunk, processor, model)
        mp4_filenames.append(mp4_filename)
        corpus.append({"text": chunk, "wav_filename": wav_filename, "mp4_filename": mp4_filename})

    if len(mp4_filenames) > 1:
        video_clips = [VideoFileClip(mp4) for mp4 in mp4_filenames]
        final_video = concatenate_videoclips(video_clips, method="compose")
        final_mp4_filename = f"final_{random.randint(1000, 9999)}.mp4"
        final_video.write_videofile(final_mp4_filename, codec='libx264', fps=24)
        corpus.append({"text": user_input, "wav_filename": None, "mp4_filename": final_mp4_filename})

    with open("corpus.json", "w") as f:
        json.dump(corpus, f)

if __name__ == "__main__":
    main()
