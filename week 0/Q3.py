import numpy as np
import matplotlib.pyplot as plt
# first, we have to define libraries which we want to use in the first question.


# section 1
std_num = 9933014


# section 2
r = int(input('Enter the particular number >= 3 for your matrix: '))
while r < 3:
    print("Invalid input!")
    r = int(input('Enter the particular number >= 3 for your matrix: '))


def circle(radius):
    d = 2 * radius + 1
    c = np.zeros((d, d), dtype='uint8')

    for x in range(d):
        for y in range(d):
            dist = np.sqrt((radius - x) ** 2 + (radius - y) ** 2)

            if (dist - r) <= 0:
                c[x, y] = 255
    return c


print("original circle: ", circle(r))


# section 3
m = circle(r)
sum1 = 0
f = 0
def getSum(n):
    sum2 = 0
    for digit in str(n):
        sum2 += int(digit)
    return sum2

sum1 = getSum(std_num)


print("The second input for the next function is: ", sum1)

def noise(input1, input2):
    d = 2 * r + 1
    noise1 = np.random.uniform(0, input2, (d, d))
    n_c = input1.astype('float')
    for x in range(d):
        for y in range(d):
            if input1[x][y] == 0:
                n_c[x][y] = input1[x][y] + noise1[x][y]
            else:
                n_c[x][y] = input1[x][y] - noise1[x][y]
    return n_c.astype('uint8')

print("noisy circle: ", noise(m, sum1))


# section 4
fig = plt.figure()
plt.axis('off')
plt.title("HW0_Image_9933014")
org = fig.add_subplot(1, 2, 1)
org.set_title("Original")
org.imshow(m, cmap='gray', vmin=0, vmax=255)

noisy = fig.add_subplot(1, 2, 2)
noisy.set_title("Noisy-noise domain: (0, 29)")
noisy.imshow(noise(m, sum1), cmap='gray', vmin=0, vmax=255)
plt.show()

# section 5
x = range(0, len(m))
y = range(0, len(m))
a, b = np.meshgrid(x, y)

fig2 = plt.figure()
plt.suptitle("HW0_Surface_9933014")

img1 = fig2.add_subplot(1, 2, 1, projection='3d')
img1.plot_surface(a, b, m, cmap='gray', vmin=0, vmax=255)
plt.title('Original in 3d')

img2 = fig2.add_subplot(1, 2, 2, projection='3d')
img2.plot_surface(a, b, noise(m, sum1), cmap='gray', vmin=0, vmax=255)
plt.title('Noisy one in 3d')

plt.show()
