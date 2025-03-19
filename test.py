# import pyttsx3
# engine = pyttsx3.init()
# def speak(text):
#     engine.say(text)
#     engine.runAndWait()
#     engine.setProperty("rate",100)
# a=input("enter : ")
# if a.lower() == "none":
#     speak("Adverse Drug Reaction is not detected.")
# else:
#     speak("Adverse Drug Reaction is detected.")
import os
print("Data saved at:", os.path.abspath("refined_dataset.csv"))
