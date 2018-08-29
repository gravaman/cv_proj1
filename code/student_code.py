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

  ############################
  ### TODO: YOUR CODE HERE ###
  buffer_size = np.uint8(filter.shape[0])
  half_buffer = np.uint8((buffer_size - 1) / 2)

  # add black frame around image
  red = image[:,:,0]
  green = image[:,:,1]
  blue = image[:,:,2]

  col_count = red.shape[1]
  black_row = np.zeros((buffer_size, col_count))

  red1 = np.insert(red, 0, black_row, axis=0)
  red1 = np.append(red1, black_row, axis=0)
  green1 = np.insert(green, 0, black_row, axis=0)
  green1 = np.append(green1, black_row, axis=0)
  blue1 = np.insert(blue, 0, black_row, axis=0)
  blue1 = np.append(blue1, black_row, axis=0)

  row1_count = red1.shape[0]
  black_col = np.zeros((row1_count, buffer_size))

  red2 = np.concatenate((black_col, red1), axis=1)
  red2 = np.append(red2, black_col, axis=1)
  green2 = np.concatenate((black_col, green1), axis=1)
  green2 = np.append(green2, black_col, axis=1)
  blue2 = np.concatenate((black_col, blue1), axis=1)
  blue2 = np.append(blue2, black_col, axis=1)

  image2 = np.dstack((red2, green2, blue2))

  # filter updated image
  rows = red2.shape[0]
  cols = red2.shape[1]

  filtered_image = np.empty((rows - buffer_size * 2, cols - buffer_size * 2))

  for i in range(buffer_size, rows - buffer_size):
      for j in range(buffer_size, cols - buffer_size):
          neighbors = red2[i - half_buffer:i + half_buffer + 1, j - half_buffer:j + half_buffer + 1]
          result = 0

          for index, row in enumerate(neighbors):
              filter_row = filter[index]
              filter_transpose = np.transpose(filter_row)
              temp = np.dot(row, filter_transpose)
              result += temp

          filtered_image[i - buffer_size][j - buffer_size] = result

  ### END OF STUDENT CODE ####
  ############################
  return filtered_image

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
