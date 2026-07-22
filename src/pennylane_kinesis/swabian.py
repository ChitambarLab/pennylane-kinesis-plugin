import numpy as np
import time

try:
    import TimeTagger as TT

except ImportError as e:
    print("The Swabian Time Tagger library is not installed.")
    print("Please install it using the instructions on the github README:")
    print("https://github.com/ChitambarLab/pennylane-kinesis-plugin")
    raise e


class Counter:
    """A counter class for the Swabian timetagger.

    :param shots: Number of shots to take. (If None, will not truncate number of counts)
    :type shots: int

    :param shot_rate: The shot rate of the detector.
    :type shot_rate: int

    :param integration_time: The integration time for the detector.
    :type integration_time: float

    :param ttag_channel_transmit: The channel number for the transmit detector.
    :type ttag_channel_transmit: int

    :param ttag_channel_reflect: The channel number for the reflect detector.
    :type ttag_channel_reflect: int
    """

    def __init__(
        self,
        shots=None,
        shot_rate=None,
        integration_time=None,
        ttag_channel_transmit=5,
        ttag_channel_reflect=4,
        truncate=False,
        verbose=False,
    ):
        self.ttag_channel_reflect = ttag_channel_reflect
        self.ttag_channel_transmit = ttag_channel_transmit
        self.integration_time = integration_time
        self.shots = shots
        self.shot_rate = shot_rate
        self.truncate = truncate
        self.verbose = verbose
        self.shot_rate_each = []

    def detector_setup(self):
        self.ttag = TT.createTimeTagger()
        self.ttag_stream = TT.TimeTagStream(
            self.ttag,
            1.5 * self.shots if self.shots is not None else 2e6,
            channels=[self.ttag_channel_reflect, self.ttag_channel_transmit],
        )
        self.counter = TT.Countrate(self.ttag, channels=[self.ttag_channel_reflect, self.ttag_channel_transmit])
        print(
            "Started Swabian timetagger for channels: ",
            self.ttag_channel_reflect,
            " and ",
            self.ttag_channel_transmit,
        )

        if self.shot_rate is None:
            self.auto_shot_rate()
        if self.integration_time is None:
            self.auto_integration_time()

    def auto_shot_rate(self):
        self.counter.startFor(int(3 * 1e12))  # Take data for 3 seconds
        self.counter.waitUntilFinished()
        shot_rate = self.counter.getData()  # Get average shot rate from the channel
        self.shot_rate_each = shot_rate
        self.shot_rate = shot_rate[0] + shot_rate[1]
        print("Shot rates:")
        print(f"T rate: {shot_rate[0]:.2f} counts/s. R rate: {shot_rate[1]:.2f} counts/s")
        return self.shot_rate

    def set_integration_time(self, time):
        # self.shots is user input, calculate integration time based on shot rate from the detector
        self.integration_time = time
        print(f"Set integration time to: {time:.5f} seconds.")

    def auto_integration_time(self):
        self.integration_time = self.shots / self.shot_rate
        print(f"Set integration time to: {self.integration_time:.5f} seconds.")

    def get_counts(self):
        channels = []
        iteration = 0
        while True:
            # if iteration == 0:
            _ = self.ttag_stream.getData()  # Clear the buffer (it will return stored data otherwise)
            # time.sleep(self.integration_time)
            # iteration += 1
            self.ttag_stream.startFor(int(self.integration_time * 1e12))  # Take data for integration time
            self.ttag_stream.waitUntilFinished()
            ttag_buffer = self.ttag_stream.getData()
            # counts = self.counter.getDataTotalCounts()
            # print(counts)
            channels = np.append(channels, ttag_buffer.getChannels())
            # print(iteration)
            break
            # if self.shots is None:
            #     break
            # elif len(channels) >= self.shots:
            #     break

        # if self.shots is not None and self.truncate:
        #     channels = channels[: self.shots]  # truncate to desired number of shots
        h_counts = len(channels[channels == self.ttag_channel_transmit])
        v_counts = len(channels[channels == self.ttag_channel_reflect])

        if self.verbose:
            print()
            print("Measured Singles:")
            print(f"H counts: {h_counts}\t V counts: {v_counts}")

            counts = h_counts + v_counts
            print("total counts : ", counts)

        return np.array([h_counts, v_counts])
