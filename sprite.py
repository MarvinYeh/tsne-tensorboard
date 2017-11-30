import numpy as np
import cv2

data = np.genfromtxt('labels.csv', dtype=int)

# np.savetxt('labels.tsv',data)
black = cv2.imread('black.jpg')
red = cv2.imread('red.jpg')


# a = np.concatenate((black,red),axis=1)
# cv2.imshow('test',a)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()

#np.savetxt('test.tsv',data,fmt='%d')



for i in range(100):

    for j in range(100):
        if j % 100 == 0:
            if data[i*100+j] == 0:
                current = black
            else:
                current = red
        else:
            if data[i*100+j] == 0:
                target = black
            else:
                target = red
            current = np.concatenate((current,target),axis=1)

    if i == 0:
        all_image = current
    else:
        all_image = np.concatenate((all_image,current),axis =0)

cv2.imwrite('all.png',all_image)