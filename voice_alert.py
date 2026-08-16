import pyttsx3
import time

# ============================================================
# SafeRouteAI Voice Alert System
# ============================================================

engine = pyttsx3.init()

# Voice settings
engine.setProperty("rate", 155)
engine.setProperty("volume", 1.0)

# Prevent repeating the same warning continuously
last_message = ""
last_time = 0

REPEAT_DELAY = 3


def speak(message):
    global last_message, last_time

    current_time = time.time()

    # Don't repeat the same message every frame
    if (
        message == last_message
        and current_time - last_time < REPEAT_DELAY
    ):
        return

    print("VOICE:", message)

    engine.say(message)
    engine.runAndWait()

    last_message = message
    last_time = current_time


# Test
speak("SafeRoute AI started")

speak("Pothole ahead. Slow down.")

speak("Dog ahead. Slow down.")

speak("Person ahead. Stop the vehicle.")

speak("Vehicle ahead. Maintain safe distance.")

speak("Road is clear.")