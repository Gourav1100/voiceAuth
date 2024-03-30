from speechbrain.inference.speaker import SpeakerRecognition
import os
import sys
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(output_folder, person_name, num_recordings=1, duration=6, freq = 44100, format='wav'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(num_recordings):
        filename = os.path.join(output_folder, f"{person_name}_{i+1}.{format}")
        if num_recordings == 1:
            filename = os.path.join(output_folder, f"{person_name}.{format}")
        print("recording - " + filename)
        recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2)
        sd.wait()
        write(filename, freq, recording)

if __name__ == "__main__":
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    if len(sys.argv) != 2:
        print("Usage: python auth.py <username>")
        exit()
    username = sys.argv[1]
    if not os.path.exists(f"./recordings/{username}/"):
        record_audio(f"./recordings/{username}/", username)
    print("recording audio for verification")
    record_audio("./auth/", "auth")
    score, prediction = verification.verify_files(f"recordings/{username}/{username}.wav", "auth/auth.wav")
    print(score, prediction)
