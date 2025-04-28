from gtts import gTTS
import os

# Read Marathi text from file
with open("output.txt", "r", encoding="utf-8") as file:
    marathi_text = file.read().strip()

# Convert text to speech
tts = gTTS(text=marathi_text, lang='mr')

# Save audio file
tts.save("marathi_output.mp3")

# Play the audio (platform-specific)
os.system("start marathi_output.mp3")  # For Windows
# os.system("afplay marathi_output.mp3")  # For macOS
# os.system("mpg123 marathi_output.mp3")  # For Linux
