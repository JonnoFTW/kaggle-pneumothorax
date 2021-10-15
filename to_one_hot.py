import humanize
import numpy as np

input = np.zeros((1, 10, 10), dtype=np.int32)  # non-one-hot
output = np.zeros((25, 10, 10), dtype=np.float32)  # one hot encoded

print("Input", humanize.naturalsize(input.nbytes))
print("Output", humanize.naturalsize(output.nbytes))


def to_one_hot_image(inp, outp):
    for y in range(inp.shape[1]):
        for x in range(inp.shape[2]):
            index_to_set_to_1 = int(inp[0, y, x])
            outp[int(index_to_set_to_1), y, x] = 1
    return outp


print(to_one_hot_image(input, output))
