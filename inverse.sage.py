

# This file was *autogenerated* from the file inverse.sage
from sage.all_cmdline import *   # import sage library

_sage_const_4 = Integer(4); _sage_const_2 = Integer(2); _sage_const_3 = Integer(3); _sage_const_1 = Integer(1)
def main():
    if len(sys.argv) < _sage_const_4 :
        print("Usage: sage inverse_polynomial.sage <coeffs_a> <coeffs_phi> <q>")
        return
    f = int(sys.argv[_sage_const_2 ])
    phi = euler_phi(f)
    K = CyclotomicField(f)
    q = int(sys.argv[_sage_const_3 ])
    coeffs_a = [int(c) for c in sys.argv[_sage_const_1 ].strip('[]').split(',')]
    F = Zmod(q)['a']; (a,) = F._first_ngens(1)
    inv_c_q = K(F(coeffs_a).inverse_mod(F(K.gen().minpoly())))

    if inv_c_q is None:
        print("None")
    else:
        print(list(inv_c_q))

if __name__ == "__main__":
    main()

