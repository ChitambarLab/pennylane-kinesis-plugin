import serial
import numpy as np
import binascii
import codecs
import elliptec



# ser = serial.Serial('/dev/ttyUSB0', 9600)



# ser.write(b'0sj00000800')



# print(ser.readline())
# def deg_to_enc(deg):
#     deg = deg*2048
#     return binascii.hexlify(deg.to_bytes(4, byteorder="big", signed=True))
#     print()
#     return int(deg).to_bytes(4, byteorder="big", signed=True)
# def to_deg(enc):
#     return int.from_bytes(binascii.unhexlify(enc),byteorder='big',signed=True) * 1 / 2048
# # print(deg_to_enc(90))
# # print(b'0ma'+bytes(deg_to_enc(90).hex(), 'utf-8'))
# print(b'0ma'+deg_to_enc())
# ser.write(b'0ma'+deg_to_enc(340))
# line = ser.readline()
# print(line)
# print(to_deg(line[3:].strip()))

cont = elliptec.Controller('/dev/ttyUSB0')
motor = elliptec.Rotator(cont, '0')

print(motor.range)
motor.set_angle(359)

print("here!!!")
# while True:
#     try:
#         ser.write(b'0gp')
#         # print(str(ser.readline()[3:11]))
#         # print(ser.readline())
#         print(to_deg(int.from_bytes(bytes.fromhex(ser.readline()[3:].strip().decode('utf-8')), byteorder="big", signed=True)))

#         ser.write(b'0fw')

#         print(to_deg(int.from_bytes(bytes.fromhex(ser.readline()[3:].strip().decode('utf-8')), byteorder="big", signed=True)))
#     except KeyboardInterrupt:
#         break
