def main():
    if len(sys.argv) < 4:
        print("Usage: sage inverse_polynomial.sage <coeffs_a> <coeffs_phi> <q>")
        return
    f = int(sys.argv[2])
    phi = euler_phi(f)
    K = CyclotomicField(f)
    q = int(sys.argv[3])
    coeffs_a = [int(c) for c in sys.argv[1].strip('[]').split(',')]
    F.<a> = Zmod(q)[]
    inv_c_q = K(F(coeffs_a).inverse_mod(F(K.gen().minpoly())))

    if inv_c_q is None:
        print("None")
    else:
        print(list(inv_c_q))

if __name__ == "__main__":
    main()
