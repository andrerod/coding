import heapq

class GPUCreditGrant:
    def __init__(self, key: str, qty: int, start_time: int, end_time: int):
        self._key = key
        self._qty = qty
        self._start_time = start_time
        self._end_time = end_time

    def __lt__(self, other):
        if self._end_time == other._end_time:
            return self._start_time < other._start_time

        return self._end_time < other._end_time

class GPUCreditCalculator:
    def __init__(self):
        self._grants = []

    def _prune_expired_grants(self, timestamp: int) -> None:
        while self._grants and self._grants[0]._end_time <= timestamp:
            heapq.heappop(self._grants)

    def grant(self, key: str, qty: int, start_time: int, end_time: int) -> None:
        new_grant = GPUCreditGrant(key, qty, start_time, end_time)

        heapq.heappush(self._grants, new_grant)

    def get_balance(self, timestamp: int) -> int:
        self._prune_expired_grants(timestamp)

        total_balance = 0
        for grant in self._grants:
            if grant._start_time <= timestamp:
                total_balance += grant._qty

        return total_balance

    def subtract(self, qty: int, timestamp: int) -> bool:
        self._prune_expired_grants(timestamp)

        # Step 1: check that we have enough balance
        available_balance = 0
        for grant in self._grants:
            if grant._start_time <= timestamp:
                available_balance += grant._qty

        if available_balance < qty:
            return False

        temp_pool = []

        remaining_to_deduct = qty
        while remaining_to_deduct > 0 and self._grants:
            current_grant = heapq.heappop(self._grants)

            if current_grant._start_time > timestamp:
                temp_pool.append(current_grant)
                continue

            if current_grant._qty > remaining_to_deduct:
                current_grant._qty -= remaining_to_deduct
                remaining_to_deduct = 0
                temp_pool.append(current_grant)
            else:
                remaining_to_deduct -= current_grant._qty

        for g in temp_pool:
            heapq.heappush(self._grants, g)

        return True


# --- Verification & Example Usage ---
if __name__ == "__main__":
    calc = GPUCreditCalculator()

    # Grant A: 100 credits, valid from t=0 to t=10
    calc.grant("A", 100, 0, 10)
    # Grant B: 50 credits, valid from t=2 to t=5 (expires sooner!)
    calc.grant("B", 50, 2, 5)

    # Check balance at t=3 (Both active: 100 + 50 = 150)
    print(f"Balance at t=3: {calc.get_balance(3)}")

    # Subtract 70 credits at t=3
    # Should take all 50 from Grant B (expires at 5) and 20 from Grant A (expires at 10)
    success = calc.subtract(70, 3)
    print(f"Deduction successful: {success}")

    # Balance at t=4 should be 80 remaining from Grant A
    print(f"Balance at t=4: {calc.get_balance(4)}")

    # At t=6, Grant B would have expired anyway, but Grant A still has 80
    print(f"Balance at t=6: {calc.get_balance(6)}")

    # Attempting to deduct more than available (e.g., 90 credits)
    overdraft = calc.subtract(90, 6)
    print(f"Overdraft allowed? {overdraft}")
