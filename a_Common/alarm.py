import winsound
from playsound import playsound


def ring():
    winsound.Beep(frequency=550, duration=1000)


def ring_1():
    playsound("__disertation_experiments/hallelujahshort.mp3")


def ring_pew():
    playsound("__disertation_experiments/pew.mp3")


def ring_sms():
    playsound("__disertation_experiments/sms-alert.mp3")
