from speechbrain.inference.speaker import SpeakerRecognition
import speech_recognition as sr
import os
import sys
import sounddevice as sd
from scipy.io.wavfile import write

sentences = [
    "Welcome back, Sarah. Your voice has been successfully authenticated.",
    "Access granted. John, your voice matches our records.",
    "Voice recognition confirmed. Welcome, Alex.",
    "Authentication complete. Emily, you may proceed.",
    "Identity verified. Welcome, Michael, to the system."
]

def record_audio(output_folder, person_name, record = True, duration=10, freq = 44100, format='wav'):
    if record:
        filename = os.path.join(output_folder, f"{person_name}.{format}")
        print("recording - " + filename)
        input("Press enter to continue...")
        recording = sd.rec(int(duration * freq), 
                samplerate=freq, channels=2)
        sd.wait()
        write(filename, freq, recording)
    else:
        for i in range(len(sentences)):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                sentence = sentences[i]
                filename = os.path.join(output_folder, f"{person_name}_{i+1}.{format}")
                print("Please speak -\n" + sentence)
                input("Press enter to continue...")
                print("recording - " + filename)
                audio_data = recognizer.listen(source)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                write(filename, freq, audio_data.get_wav_data())

if __name__ == "__main__":
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    if len(sys.argv) != 2:
        print("Usage: python auth.py <username>")
        exit()
    username = sys.argv[1]
    if not os.path.exists(f"./recordings/{username}/"):
        record_audio(f"./recordings/{username}/", username, False)
    print("recording audio for verification")
    record_audio("./auth/", "auth")
    avg = 0
    for index in range(len(sentences)):
        score, prediction = verification.verify_files(f"recordings/{username}/{username}_{i + 1}.wav", "auth/auth.wav")
        avg += score
    print("Matching score: " + str(avg / len(sentences)))
    if avg / len(sentences) > 0.6:
        print("Authenticated")
    else:
        print("Not Authenticated")
