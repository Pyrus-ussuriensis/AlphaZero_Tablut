#https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-in-any-base-to-a-string

import string
digs = string.digits + string.ascii_letters

def int2base(x, base, length):
    if x < 0:
        raise ValueError("x must be non-negative")
    out = [0]*length
    for k in range(length):
        out[k] = x % base
        x //= base
    if x != 0:
        raise ValueError("x >= base**length")
    return out  # [d0,d1,d2,d3] = [x1,y1,x2,y2]

def base2int(move, n):
    x1, y1, x2, y2 = move
    return x1+y1*n+x2*n**2+y2*n**3
    

def test():
    size=7
    validmoves = [[3, 0, 1, 0], [3, 0, 2, 0], [3, 0, 4, 0], [3, 0, 5, 0], [3, 1, 0, 1], [3, 1, 1, 1], [3, 1, 2, 1], [3, 1, 4, 1], [3, 1, 5, 1], [3, 1, 6, 1], [0, 3, 0, 1], [0, 3, 0, 2], [0, 3, 0, 4], [0, 3, 0, 5], [1, 3, 1, 0], [1, 3, 1, 1], [1, 3, 1, 2], [1, 3, 1, 4], [1, 3, 1, 5], [1, 3, 1, 6], [3, 6, 1, 6], [3, 6, 2, 6], [3, 6, 4, 6], [3, 6, 5, 6], [3, 5, 0, 5], [3, 5, 1, 5], [3, 5, 2, 5], [3, 5, 4, 5], [3, 5, 5, 5], [3, 5, 6, 5], [6, 3, 6, 1], [6, 3, 6, 2], [6, 3, 6, 4], [6, 3, 6, 5], [5, 3, 5, 0], [5, 3, 5, 1], [5, 3, 5, 2], [5, 3, 5, 4], [5, 3, 5, 5], [5, 3, 5, 6]]
    print(validmoves)
    for m in validmoves:
        i = m[0]+m[1]*size+m[2]*size**2+m[3]*size**3
        print(i,":",int2base(i,size,4))

#test()

