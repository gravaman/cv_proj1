import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  BUFFER_SIZE = np.uint8(filter.shape[0])
  HALF_BUFFER = np.uint8((BUFFER_SIZE - 1) / 2)

  framed_image = frame_image(image, BUFFER_SIZE)

  rows, cols, _ = framed_image.shape
  temp_channels = []

  # filter each color channel
  for channel in color_channels(framed_image):
      filtered_channel = np.empty((rows - BUFFER_SIZE * 2, cols - BUFFER_SIZE * 2))

      # filter each row
      for i in range(BUFFER_SIZE, rows - BUFFER_SIZE):
          # filter each column
          for j in range(BUFFER_SIZE, cols - BUFFER_SIZE):
              # get relevant neighbors
              neighbors = channel[i - HALF_BUFFER:i + HALF_BUFFER + 1, j - HALF_BUFFER:j + HALF_BUFFER + 1]
              result = 0

              # implement filter on neighbors
              for index, row in enumerate(neighbors):
                  filter_row = filter[index]
                  filter_transpose = np.transpose(filter_row)
                  result += np.dot(row, filter_transpose)

              # add pixel result to filtered channel
              filtered_channel[i - BUFFER_SIZE][j - BUFFER_SIZE] = result
      temp_channels.append(filtered_channel)

  filtered_image = np.dstack((temp_channels[0], temp_channels[1], temp_channels[2]))
  return filtered_image

def frame_image(image, buffer_size):
    channels = color_channels(image)
    for index, channel in enumerate(channels):
      row_count, col_count = channel.shape
      black_col = np.zeros((row_count + buffer_size * 2, buffer_size))
      black_row = np.zeros((buffer_size, col_count))

      channel = np.concatenate((black_row, channel), axis=0)
      channel = np.append(channel, black_row, axis=0)

      channel = np.concatenate((black_col, channel), axis=1)
      channel = np.append(channel, black_col, axis=1)

      channels[index] = channel

    return np.dstack((channels[0], channels[1], channels[2]))

def color_channels(image):
    return [image[:,:,0], image[:,:,1], image[:,:,2]]

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  low_frequencies = my_imfilter(image1, filter)
  low_frequencies2 = my_imfilter(image2, filter)

  img1_channels = color_channels(image1)
  img2_channels = color_channels(image2)
  low_frequencies1_channels = color_channels(low_frequencies)
  low_frequencies2_channels = color_channels(low_frequencies2)

  high_frequencies = [img2_channel - low_frequencies2_channels[index] for index, img2_channel in enumerate(img2_channels)]:
  hybrid = [high_frequencies[index] + low_channel1 for index, low_channel1 in enumerate(low_frequencies1_channels)]

  return low_frequencies, stacker(high_frequencies), clip_image(stacker(hybrid))

def clip_image(image):
    channels = color_channels(image)
    clipped_image = []
    for channel in channels:
        clipped_channel = []
        for row in channel:
            clipped_row = [clip_val(val) for val in row]
            clipped_channel.append(clipped_row)
        clipped_image.append(clipped_channel)
    return stacker(clipped_image)

def stacker(arr):
    return np.dstack((arr[0], arr[1], arr[2]))

def clip_val(val):
    # clip values below 0 or above 1
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val
