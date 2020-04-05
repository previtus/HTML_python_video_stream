from timeit import default_timer as timer


import sys
sys.path.append('/home/vitek/Vitek/Projects_local_for_ubuntu/GAN_handler')
from progressive_gan_handler import ProgressiveGAN_Handler

class gan_capture(object):
    """
    Generates individual frames from a stream of GAN generated images
    """

    def __init__(self, pro_gan_path = "aerials512vectors1024px_snapshot-010200.pkl"):
        self.pro_handler = ProgressiveGAN_Handler(pro_gan_path)
        self.pro_handler.report()
        example_input = self.pro_handler.example_input()
        example_output = self.pro_handler.infer(example_input)
        print("example_output:", example_output.shape)

    def get_frame(self):
        time_start = timer()

        example_input = self.pro_handler.example_input()
        example_output = self.pro_handler.infer(example_input)
        frame = example_output[0]
        frame_enc = cv2.imencode('.jpg', frame)[1].tobytes()

        time_end = timer()
        fps = 1.0 / (time_end - time_start)
        return frame_enc, fps

    def destroy(self):
        pass


"""

from timeit import default_timer as timer
# Basic measurements
repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()
    example_input = pro_handler.example_input(verbose=False)
    example_output = pro_handler.infer(example_input, verbose=False)
    t_infer = timer() - t_infer
    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of 1 sample) took", t_infer, "sec.")
times = np.asarray(times)
print("Statistics:")
print("prediction time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")
# Batch measurements
how_many = 4 # too big? gpu mem explodes
repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()
    example_inputs = pro_handler.example_input(how_many = how_many, verbose=False)
    example_outputs = pro_handler.infer(example_inputs, verbose=False)
    t_infer = timer() - t_infer
    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of",how_many,"samples) took", t_infer, "sec.")
times = np.asarray(times)
print("Statistics:")
print("prediction of whole",how_many," took time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")
print("prediction as divided for one - avg +- std =", np.mean(times/how_many), "+-", np.std(times/how_many), "sec.")
"""
