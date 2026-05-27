Lernprojekt: „Voice-Roboter in Python“

Die Kinder steuern einen kleinen Roboter per Sprache.

Beispielbefehle:

hallo
links
rechts
hoch
runter
stopp
rot
blau
tanz
hilfe

Das Python-Programm hört zu, erkennt den Befehl und führt eine Aktion aus.

Technischer Aufbau

Für den Anfang würde ich diese Bibliotheken nutzen:

pip install vosk sounddevice

Zusätzlich brauchst du ein kleines deutsches Vosk-Modell.

Für Kinder-Workshops würde ich einen Projektordner vorbereiten:

voice-roboter/
│
├── main.py
├── commands.py
├── model/
│   └── vosk-de-small...
└── README.md

Die Kinder arbeiten hauptsächlich in commands.py, nicht direkt an der komplizierten Audio-Erkennung.

Sehr einfaches Python-Beispiel
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer


MODEL_PATH = "model"

audio_queue = queue.Queue()


def callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))


def handle_command(text):
    if "links" in text:
        print("Roboter geht nach links")
    elif "rechts" in text:
        print("Roboter geht nach rechts")
    elif "hoch" in text:
        print("Roboter geht nach oben")
    elif "runter" in text:
        print("Roboter geht nach unten")
    elif "stopp" in text:
        print("Roboter stoppt")
    elif "rot" in text:
        print("Farbe wird rot")
    elif "blau" in text:
        print("Farbe wird blau")
    elif "hallo" in text:
        print("Hallo, ich bin dein Roboter!")
    else:
        print("Ich habe verstanden:", text)


def main():
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Sprich einen Befehl...")
        while True:
            data = audio_queue.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    print("Erkannt:", text)
                    handle_command(text)


if __name__ == "__main__":
    main()
Für Kinder besser: Befehle auslagern

Damit es nicht zu technisch wird, würde ich eine Datei commands.py machen:

def handle_command(text):
    if "links" in text:
        print("⬅️ Der Roboter geht nach links")

    elif "rechts" in text:
        print("➡️ Der Roboter geht nach rechts")

    elif "tanz" in text:
        print("🤖 Der Roboter tanzt!")

    elif "hallo" in text:
        print("👋 Hallo Mensch!")

    else:
        print("❓ Diesen Befehl kenne ich noch nicht:", text)

Dann ist die Lernaufgabe:

„Füge einen neuen Sprachbefehl hinzu.“

Zum Beispiel:

elif "rakete" in text:
    print("🚀 3, 2, 1, Start!")

Das ist für Kinder ab 9 sehr greifbar.

Didaktischer Ablauf für 90 Minuten
Teil 1: Was ist ein Befehl?

Die Kinder sagen Befehle, der Kursleiter schreibt sie an die Tafel:

links
rechts
spring
stopp
farbe rot

Dann wird erklärt:

Computer brauchen klare Befehle.

Teil 2: Erst ohne Mikrofon

Bevor Sprache dazukommt, würde ich mit Texteingabe starten:

while True:
    text = input("Befehl: ")
    handle_command(text)

Das ist extrem wichtig, weil die Kinder zuerst die Logik verstehen sollen.

Teil 3: Dann mit Sprache

Erst wenn die Logik sitzt, wird input() durch Mikrofon-Erkennung ersetzt.

So verstehen die Kinder:

Vorher: Ich tippe "links"
Nachher: Ich sage "links"

Die Programmlogik bleibt gleich.

Besserer Einstiegscode ohne Voice

Für den allerersten Workshop:

def handle_command(text):
    if text == "links":
        print("Der Roboter geht nach links")

    elif text == "rechts":
        print("Der Roboter geht nach rechts")

    elif text == "stopp":
        print("Der Roboter bleibt stehen")

    else:
        print("Diesen Befehl kenne ich nicht")


while True:
    command = input("Sag dem Roboter etwas: ")
    handle_command(command)

Danach erweitern die Kinder:

elif text == "spring":
    print("Der Roboter springt")
Möglicher Lernpfad über mehrere Stunden
Stunde 1: Textbefehle
input()
print()
if
elif
else

Projekt: Roboter reagiert auf getippte Befehle.

Stunde 2: Eigene Befehle
neue Kommandos hinzufügen
Fehler erkennen
