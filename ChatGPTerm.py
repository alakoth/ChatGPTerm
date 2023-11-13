import wave
import pyaudio
import shutil
import sounddevice as sd
import scipy.io.wavfile as wav
import argparse
import keyboard
import time as t
import librosa
import soundfile as sf
import numpy as np
from colorama import Fore, Style
from halo import Halo
from openai import OpenAI
from pathlib import Path

parser = argparse.ArgumentParser(description='Text-to-Speech using OpenAI API')
parser.add_argument('-ak', '--api_key', type=str, required=True, help='OpenAI API key')
parser.add_argument('-s', '--system', type=str, help='System prompt as a string or path to a .txt or .md file')
parser.add_argument('-tr', '--transcript', type=int, choices=[0, 1, 2, 3], default=0, help='Transcript mode: 0 = no transcript (default), 1 = transcript in terminal only, 2 = transcript saved to disk only, 3 = transcript in terminal and saved to disk')
args = parser.parse_args()

client = OpenAI(api_key=args.api_key)


# Function to play audio file
def play_audio(file_path):
    try:
        try:
            wf = wave.open(str(file_path), 'rb')
        except wave.Error as e:
            audio_data, sample_rate = librosa.load(str(file_path), sr=None)
            temp_file_path = file_path.with_suffix('.temp.wav')
            sf.write(temp_file_path, audio_data, sample_rate)
            wf = wave.open(str(temp_file_path), 'rb')
        
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()
            file_path.unlink(missing_ok=True)
    except wave.Error as e:
        print(f"An error occurred while playing audio: {e}")
        file_path.unlink(missing_ok=True)

# Function to record voice input and save to a WAV file
def record_voice_input(sample_rate=16000):
    while not keyboard.is_pressed('v'):
        pass  
    with Halo(text='Recording...', spinner='dots'):
        recording = []
        while True:
            recording.extend(sd.rec(int(sample_rate * 0.1), samplerate=sample_rate, channels=1))
            sd.wait()
            if keyboard.is_pressed('v'): 
                break
    wav_file_path = Path(__file__).parent / "input_voice.wav"
    wav.write(wav_file_path, sample_rate, np.concatenate(recording, axis=0))
    return wav_file_path

# Function to transcribe speech using Whisper
def transcribe_speech(whisper_model, audio_file_path):
    with Halo(text='Transcribing...', spinner='dots'):
        response = client.audio.transcriptions.create(
            model=whisper_model,
            file=audio_file_path
        )
    return response.text

# Function to synthesize speech from text
def synthesize_speech(text):
    speech_file_path = Path(__file__).parent / "speech.wav"
    with Halo(text='Thinking...', spinner='dots'):
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(speech_file_path)

print("Press V to start talking, press V again to stop")

conversation = []
system_prompt_content = None
if args.system:
    system_prompt_path = Path(args.system)
    if system_prompt_path.is_file():
        with system_prompt_path.open('r', encoding='utf-8') as file:
            system_prompt_content = file.read()
    else:
        system_prompt_content = args.system

# Main loop for continuous conversation
while True:
    try:
        audio_file_path = record_voice_input()
        user_speech = transcribe_speech("whisper-1", audio_file_path)
        messages = []
        if system_prompt_content:
            messages.append({"role": "system", "content": system_prompt_content})
        messages.append({"role": "user", "content": user_speech})
        completion_response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=1024
        )
        response_text = completion_response.choices[0].message.content.strip()
        conversation.append(f"USER: {user_speech}")
        conversation.append(f"ASST: {response_text}")
        if args.transcript in [1, 3]:
            print("\n".join(conversation[-2:]))
        synthesize_speech(response_text)
        speech_file_path = Path("speech.wav")
        if speech_file_path.exists():

            terminal_width = shutil.get_terminal_size().columns
            interrupt_message = "[CTRL+C to interrupt]"
            spaces_to_right_align = terminal_width - len(interrupt_message)
            with Halo(text=f"{'Speaking... '.rjust(spaces_to_right_align) + Fore.RED + interrupt_message + Style.RESET_ALL}", spinner='dots', color='red'):
                play_audio(speech_file_path)
        else:
            print("Error: The response does not contain audio data.\nNo audio file to play.")
    except KeyboardInterrupt:
        try:
            print("\nInterrupted, press CTRL+C again to exit.")
            t.sleep(2)  
        except KeyboardInterrupt:
            print("\nExiting...")
            # Write the conversation to a transcript file based on the transcript flag
            if args.transcript in [2, 3]:
                transcript_file_path = Path(__file__).parent / "conversation_transcript.txt"
                with open(transcript_file_path, "w") as transcript_file:
                    transcript_file.write("\n".join(conversation))
            break
