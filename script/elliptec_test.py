import sys
import glob
import serial


def serial_ports():
    """Lists serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


print(serial_ports())

import time

ser = serial
try:
    ser = serial.Serial("COM4", baudrate=9600, bytesize=8, parity="N", stopbits=1, timeout=0.1)
except Exception:
    print("Exception")

print("Connected")
try:
    ser.write(b"0in")
    print(ser.readline())
    ser.write(b"1in")
    print(ser.readline())
    ser.write(b"2in")
    print(ser.readline())
    ser.write(b"3in")
    print(ser.readline())
    ser.write(b"4in")
    print(ser.readline())
finally:
    ser.close()
    exit()