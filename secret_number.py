target_number = 31
previous = 0

def check(number):
    global previous, target_number

    return_value = -1
    if previous == target_number:
        return_value = 0
    elif previous != 0:
        return_value = -1 if previous > target_number else 1

    previous = number
    return return_value


def solution():
    lo, hi = 1, 100
    previous_guess = None

    while lo <= hi:
        mid = (lo + hi) // 2
        result = check(mid)

        # The result belongs to previous_guess, not mid.
        if previous_guess is not None:
            if result == 0:
                return previous_guess
            elif result < 0:
                hi = previous_guess - 1
            else:
                lo = previous_guess + 1

        previous_guess = mid

    return None


if __name__ == "__main__":
    print(solution())  # 30
