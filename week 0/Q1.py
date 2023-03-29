import numpy as np
# first, we have to define libraries which we want to use in the first question.


# section 1
# It imports the numpy library with a shortened name 'np'.
# It initializes an empty list named A and then it assigns 80 random numbers within a range of 10 to 54000 to A using the 'np.random.uniform()' method.
A = []
A = np.random.uniform(10, 54000, 80)


# section 2
# It prints the type of A and its data type.
print(type(A))
print(A.dtype)


# section 3
# It rounds off the numbers in A to their nearest integers using the 'np.round()' method and assigns the rounded values to A1.
A1 = np.round(A)


# section 4
# It initializes and assigns a set of values to variables of different data types using the 'np.array()' method with the 'dtype' parameter.
# It then prints the values for each variable.
int_A = np.array(A1, dtype='int')
print("int: ", int_A)

int8_A = np.array(A1, dtype='int8')
print("int8: ", int8_A)

uint8_A = np.array(A1, dtype='uint8')
print("uint8: ", uint8_A)

int16_A = np.array(A1, dtype='int16')
print("int16: ", int16_A)

uint16_A = np.array(A1, dtype='uint16')
print("uint16: ", uint16_A)

int32_A = np.array(A1, dtype='int32')
print("int32: ", int32_A)

int64_A = np.array(A1, dtype='int64')
print("int64: ", int64_A)

float_A = np.array(A1, dtype='float')
print("float: ", float_A)

float32_A = np.array(A1, dtype='float32')
print("float32: ", float32_A)

float64_A = np.array(A1, dtype='float64')
print("float64: ", float64_A)

# section 5
# It reshapes A1 into a 2D array with dimensions (8,10) and assigns it to A2.
A2 = A1.reshape((8, 10))
print("The reshaped format is: ", A2)

# section 6
# It finds and prints the maximum and minimum values in the array A1.
Max_list = max(A1)
print(Max_list)

Min_list = min(A1)
print(Min_list)


# section 7
# It converts the data type of elements in A2 from 'int64' to 'int8' using the 'np.array()' method with the 'dtype' parameter and assigns it to A3.
A3 = np.array(A2, dtype='int8')

# section 8
# It creates two empty lists: C_two and R_three.
# It then iterates through the 2nd column of A2 to append elements to C_two, convert it to a tuple and print it.
# Similarly, it iterates through the 3rd row of A2 to append elements to R_three, convert it to a tuple and print it.
C_two = []
R_three = []
for i in range(8):
    C_two.append(A2[i][1])

C_two = tuple(C_two)
for j in range(1, 10):
    R_three.append(A2[2][j])

R_three = tuple(R_three)
print("C_two is: ", C_two, "and R_three is: ", R_three)

# section 9
# It creates a dictionary from the two tuples C_two and R_three using the 'zip()' and 'dict()' methods and prints the dictionary
x = zip(C_two, R_three)
print(dict(x))

