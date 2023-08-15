import math
# 定義函數
# 定義函數


def solve(a, b, c):
    # use formula ( -b +- sqrt(b*b - 4*a*c) ) / (2*a)
    D = b*b - (4*a*c)
    if D > 0.0:
        sq = math.sqrt(D)
        x1 = (-b + sq)/(2*a)
        x2 = (-b - sq)/(2*a)
        print(" ============== x1 = %.2f" % x1, end='')
        print(", x2 = {:.2f}, 2 real roots <<<".format(x2))
    elif D < 0.0:
        sq = math.sqrt(-D)
        x1 = complex(-b/(2*a),  sq/(2*a))
        x2 = complex(-b/(2*a), -sq/(2*a))
        # use data member
        print(" ============== x1 = (%.2f + %.2f J)" %
              (x1.real, x1.imag), end='')  # method 1

        print(", (x2 = {:.2f}".format(x1.real), end='')  # method 2
        print(" - {:.2f} J) <<<".format(x1.imag))
    else:
        sq = 0.0
        x1 = x2 = -b/(2*a)
        print(" ============== x1 = x2 = %.2f, same real roots <<<" % x1)


    # end if
while (True):
    print("Solving 2nd order equation(a X^2 + b X + c = 0)...")

    a = eval(input("enter a (0 to quit) > "))

    if (a == 0):
        break  # exit while

    b = eval(input("enter b > "))

    c = eval(input("enter c > "))
    solve(a, b, c)
# end while
