target_number = 30
previous = 0

def check(number) -> (int, int):
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
    pending = None

    while lo <= hi:
        mid = (lo + hi) // 2

        # The response is for the previous guess.
        result = check(mid)

        if pending is not None:
            guess, result = pending
            if result == 0:       # previous guess was correct
                return guess
            elif result < 0:      # secret < guess
                hi = guess - 1
            else:                 # secret > guess
                lo = guess + 1

        pending = (mid, result)

if __name__ == "__main__":
    print(solution())
