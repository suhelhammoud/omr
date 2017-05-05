
from enum import Enum

class QBlob:
    '''Question white blob'''

    def __init__(self, x, y, size=0):
        self.x = x
        self.y = y
        self.size = size

    def toString(self):
        return str(self.x) + "\t"+ str(self.y) +"\t" + str(self.size)

    def com_value(self):
        return self.x + 10000 * self.y


if __name__ == '__main__':
    a = QBlob(1, 1)
    b = QBlob(1, 2)
    c = QBlob(2, 1)
    d = QBlob(2, 2)

    l1 = [a, b, c, d]
    for i in l1:
        print(i.toString())

    print('-----------------')
    l2 = sorted(l1, key = lambda x: x.com_value())
    for i in l2:
        print(i.toString())

    e = [[1], [3]]
    for i in e:
        print(i)