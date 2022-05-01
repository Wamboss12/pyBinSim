import msvcrt

while True:
    if msvcrt.kbhit():
        char = msvcrt.getch()
        key = ord(msvcrt.getch())
        print(type(key), key, "---", type(char), char)