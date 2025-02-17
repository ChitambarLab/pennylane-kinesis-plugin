import elliptec
import numpy as np
import time
from pennylane_kinesis import swabian, read_config, write_json, read_json

config_file = "./src/pennylane_kinesis/2024_09_18_timur_elliptec_config.txt"


controller = elliptec.Controller(port="COM4", timeout=5, write_timeout=5,debug=False)
wp_details = read_config(config_file)
rotation_mounts = []
for wp in wp_details:
    rotation_mounts.append(elliptec.Rotator(controller, address=str(wp["address"]),debug=False))

def set_mount_angle(angle, index):
    # print(angle)
    # if index == 0 or index == 2:
    #     set_angle = (angle % 90) + float(wp_details[index]["angle_offset"])
    # else:
    #     set_angle = (angle % 180) + float(wp_details[index]["angle_offset"])

    set_angle = (float(wp_details[index]["angle_offset"]) + angle) % 360
    # print(set_angle)
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(set_angle)  # This will block until the device reaches the desired angle
        if angle_ret is None:
            print("Angle return is None when setting waveplate", index+1, "to ", set_angle)
            print("Trying again...")
            time.sleep(0.01)
    
def set_background_angle(index):
    set_angle = float(wp_details[index]["background_angle"])
    angle_ret = None
    while angle_ret is None:
        angle_ret = rotation_mounts[index].set_angle(set_angle)
        if angle_ret is None:
            print("Angle return is None when setting waveplate", index+1, "to ", set_angle)
            print("Trying again...")
            time.sleep(0.01)


for i in range(len(rotation_mounts)):
    set_mount_angle(0, i)

counter = swabian.Counter()
counter.detector_setup()

counter.shots = None
counter.set_integration_time(1)
print("Setting background angle to take background counts")
for i in range(len(rotation_mounts)):
    set_background_angle(i)

background_counts = counter.get_counts()
print("Background counts", background_counts)
behavior_matrices = []
angle_arrs = []
# try:
#     for ahwp_0_angle in range(0,90,10):
#     # for ahwp_0_angle in range(0,1):
#         for aqwp_0_angle in range(0,180,10):
#         # for aqwp_0_angle in range(0,1):
#             for ahwp_1_angle in range(0,90,10):
#             # for ahwp_1_angle in range(0,1):
#                 for aqwp_1_angle in range(0,180,10):
#                 # for aqwp_1_angle in range(0,1):
#                     print(f"\nAngles are: \nzero-state HWP:{ahwp_0_angle}, zero-state QWP:{aqwp_0_angle}\none-state HWP:{ahwp_1_angle}, one-state QWP:{aqwp_1_angle}")
#                     angles = np.array([ahwp_0_angle, aqwp_0_angle, ahwp_1_angle, aqwp_1_angle])
#                     angle_arrs.append(angles.tolist())
#                     set_mount_angle(ahwp_0_angle, 0)
#                     set_mount_angle(aqwp_0_angle, 1)
#                     zero_state_counts = counter.get_counts().tolist()
#                     # print(zero_state_counts)
#                     set_mount_angle(ahwp_1_angle, 0)
#                     set_mount_angle(aqwp_1_angle, 1)
#                     one_state_counts = counter.get_counts().tolist()
#                     behavior_matrices.append([zero_state_counts, one_state_counts])

counter.shots = 10000
for i in range(len(rotation_mounts)):
    set_mount_angle(0, i)

counter.auto_integration_time()
try:
    for ahwp_0_angle in range(0,90,1):
    # for ahwp_0_angle in range(0,1):
        for bhwp_0_angle in range(0,90,1):
            for aqwp_0_angle in range(0,180,1):
                for bqwp_0_angle in range(0,180,1):
        # for aqwp_0_angle in range(0,1):
            # for ahwp_1_angle in range(0,90,10):
            # # for ahwp_1_angle in range(0,1):
            #     for aqwp_1_angle in range(0,180,10):
                # for aqwp_1_angle in range(0,1):
                    print(f"\nAngles are: \n aHWP:{ahwp_0_angle}, aQWP:{aqwp_0_angle}, bHWP:{bhwp_0_angle}, bQWP:{bqwp_0_angle}")
                    angles = np.array([ahwp_0_angle, aqwp_0_angle, bhwp_0_angle, bqwp_0_angle])
                    angle_arrs.append(angles.tolist())
                    set_mount_angle(ahwp_0_angle, 0)
                    set_mount_angle(aqwp_0_angle, 1)
                    set_mount_angle(bhwp_0_angle, 2)
                    set_mount_angle(bqwp_0_angle, 3)
                    zero_state_counts = counter.get_counts().tolist()
                    # print(zero_state_counts)
                    # set_mount_angle(ahwp_1_angle, 0)
                    # set_mount_angle(aqwp_1_angle, 1)
                    # one_state_counts = counter.get_counts().tolist()
                    behavior_matrices.append(zero_state_counts)
except KeyboardInterrupt:
    json = {"angles": angle_arrs,
        "behavior_matrices": behavior_matrices,
        "background_counts": background_counts.tolist()}
    write_json(json, "behavior_matrices_4wp")

json = {"angles": angle_arrs,
        "behavior_matrices": behavior_matrices,
        "background_counts": background_counts.tolist()}
write_json(json, "behavior_matrices_4wp")

                




