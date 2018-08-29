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

  buffer_size = np.uint8(filter.shape[0])
  half_buffer = np.uint8((buffer_size - 1) / 2)

  # add black frame around image
  framed_image = frame_image(image, buffer_size)

  rows, cols, _ = framed_image.shape
  temp_channels = []

  for channel in color_channels(framed_image):
      filtered_channel = np.empty((rows - buffer_size * 2, cols - buffer_size * 2))

      for i in range(buffer_size, rows - buffer_size):
          for j in range(buffer_size, cols - buffer_size):
              neighbors = channel[i - half_buffer:i + half_buffer + 1, j - half_buffer:j + half_buffer + 1]
              result = 0

              for index, row in enumerate(neighbors):
                  filter_row = filter[index]
                  filter_transpose = np.transpose(filter_row)
                  temp = np.dot(row, filter_transpose)
                  result += temp

              filtered_channel[i - buffer_size][j - buffer_size] = result
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
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    return [red, green, blue]

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

  ############################
  ### TODO: YOUR CODE HERE ###

  raise NotImplementedError('`create_hybrid_image` function in ' +
    '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
