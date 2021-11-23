#1500, 0, 0, 1, 0
#2000, 0, 0, 1, 1

import time

INTERVALO = 5
# vitro, luces, luces2, ...
DEVICES = {"M": 1,  #movil
        "L": 1,     #lámpara
        "T": 0,     #tv
        "A": 0,     #aspiradora
        }

while True:
    tick = time.time()
    text = input("Dato: ")
    
    if (" " in text):
        amp, change = text.split(" ")
    else:
        amp, change= text, False

    amp = float(amp)
    
    if change:
        for device in DEVICES:
            if change == device:
                DEVICES[device] = int(not DEVICES[device])   # Toggle device state

    print("devices: ", DEVICES)

    num_lines = sum(1 for line in open('demo_db.csv'))
    print("N Lines: ", num_lines)

    devices_list = [DEVICES[device] for device in DEVICES]
    devices_list.insert(0, amp)
    
    with open("demo_db.csv", "a") as db:
        db.write(str(devices_list)[1:-1] + "\n")


    while(time.time() - tick < INTERVALO):
        pass
