import winsound


def ring():
    # Play Windows exit sound.
    # winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
    # winsound.Beep(frequency=150, duration=700)
    # winsound.Beep(frequency=550, duration=700)
    # winsound.Beep(frequency=850, duration=300)
    winsound.Beep(frequency=550, duration=1000)
    # winsound.Beep(frequency=1050, duration=2000)

    # Probably play Windows default sound, if any is registered (because
    # "*" probably isn't the registered name of any sound).
    # winsound.PlaySound("*", winsound.SND_ALIAS)


def ring_1():
    pass
    # winsound.MessageBeep(10)
    from playsound import playsound
    # playsound("__disertation_experiments/Hallelujah-sound-effect.mp3")
    playsound("__disertation_experiments/hallelujahshort.mp3")
    # winsound.PlaySound(, winsound.SND_NODEFAULT)


def ring_pew():
    pass
    # winsound.MessageBeep(10)
    from playsound import playsound
    # playsound("__disertation_experiments/Hallelujah-sound-effect.mp3")
    playsound("__disertation_experiments/pew.mp3")
    # winsound.PlaySound(, winsound.SND_NODEFAULT)


def ring_sms():
    pass
    # winsound.MessageBeep(10)
    from playsound import playsound
    # playsound("__disertation_experiments/Hallelujah-sound-effect.mp3")
    playsound("__disertation_experiments/sms-alert.mp3")
    # winsound.PlaySound(, winsound.SND_NODEFAULT)


if __name__ == '__main__':
    FEATURES = [
        1,  # L
        2,  # S
        4,  # B
        8,  # A
        3,  # LS
        5,  # LB
        6,  # SB
        9,  # LA
        9,  # LA
        9,  # LA
        9,  # LA
        9,  # LA
        10,  # SA
        10,  # SA
        10,  # SA
        10,  # SA
        10,  # SA
        12,  # BA
        12,  # BA
        12,  # BA
        12,  # BA
        12,  # BA
        7,  # LSB
        11,  # LSA
        11,  # LSA
        11,  # LSA
        11,  # LSA
        13,  # LBA
        13,  # LBA
        13,  # LBA
        13,  # LBA
        13,  # LBA
        14,  # SBA
        14,  # SBA
        14,  # SBA
        14,  # SBA
        15,  # LSBA
        15,  # LSBA
        15,  # LSBA
        15,  # LSBA
        15,  # LSBA
    ]
    print(set(FEATURES))
    ring_sms()
