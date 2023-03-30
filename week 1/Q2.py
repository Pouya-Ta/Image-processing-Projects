import numpy as np


# first, we have to define libraries which we want to use in the first question.

# section 1, 2, 3
def matrix(seed, dims):
    array1 = np.array([[seed] * dims[1]] * dims[0])
    for i in range(dims[0]):
        for j in range(dims[1]):
            if i > 0 and j > 0:
                array1[i][j] = (array1[i - 1][j] + array1[i][j - 1] + array1[i - 1][j - 1])
            else:
                array1[i][j] = array1[0][0]
    for j in range(dims[1]):
        array1[1][j] *= -1
    return array1


def check_input(tuple1, input1):
    if type(input1) != int:
        print("Error! INVALID INPUT")
        return 0
    a = len(tuple1)
    for i in range(a):
        if tuple1[i] < 0:
            print("Error! INVALID INPUT")
            return 0

    return 0

seed = int(input('Enter your input number: '))

dims = ()
dims_0 = int(input('Enter number of rows: '))
dims_1 = int(input('Enter number of cols: '))
dims = list(dims)
dims.append(dims_0)
dims.append(dims_1)
dims = tuple(dims)

check_input(dims, seed)
print(matrix(seed, dims))
