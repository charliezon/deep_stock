from decimal import Decimal as D
import math

def round_float(g, pos=2):
    if g<0:
        f = -g
    else:
        f = g
    p1 = pow(D('10'), D(str(pos+1)))
    last = D(str(int(D(str(f))*p1)))%D('10')
    p = pow(D('10'), D(str(pos)))
    if last >= 5:
        result = float(math.ceil(D(str(f))*p)/p)
    else:
        result = float(math.floor(D(str(f))*p)/p)
    if g<0:
        return -result
    else:
        return result