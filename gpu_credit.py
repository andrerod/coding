import heapq

class CreditGrant:
    def __init__(self, grant_id: str, amount: float, start_time: int, expiration_time: int):
        self.grant_id = grant_id
        self.amount = amount
        self.start_time = start_time
        self.expiration_time = expiration_time

    # The min-heap uses this to sort grants by their expiration time (Earliest Expiry First)
    def __lt__(self, other):
        if self.expiration_time == other.expiration_time:
            return self.start_time < other.start_time  # Tie-breaker: FIFO by start time
        return self.expiration_time < other.expiration_time

class GPUCreditCalculator:
    def __init__(self):
        # Min-heap storing active CreditGrant objects
        self.active_grants = []

    def _prune_expired_grants(self, timestamp: int):
        """Helper to remove grants that have expired by the given timestamp."""
        while self.active_grants and self.active_grants[0].expiration_time <= timestamp:
            heapq.heappop(self.active_grants)

    def grant(self, grant_id: str, amount: float, start_time: int, expiration_time: int) -> None:
        """Adds a new credit grant to the system."""
        if amount <= 0 or start_time >= expiration_time:
            return

        new_grant = CreditGrant(grant_id, amount, start_time, expiration_time)
        heapq.heappush(self.active_grants, new_grant)

    def get_balance(self, timestamp: int) -> float:
        """Returns the total active, unexpired credit balance at a specific timestamp."""
        self._prune_expired_grants(timestamp)

        total_balance = 0.0
        for grant in self.active_grants:
            # A grant is only active if the current timestamp is >= its start_time
            if grant.start_time <= timestamp:
                total_balance += grant.amount
        return total_balance

    def subtract(self, amount: float, timestamp: int) -> bool:
        """
        Consumes 'amount' of credits at a given timestamp using the soonest-expiring credits first.
        Returns True if successful, False if there are insufficient credits (atomic operation).
        """
        if amount <= 0:
            return True

        self._prune_expired_grants(timestamp)

        # 1. First pass: Verify if there is enough total balance across available grants
        available_balance = 0.0
        valid_grants = []

        for g in self.active_grants:
            if g.start_time <= timestamp:
                available_balance += g.amount
                valid_grants.append(g)

        if available_balance < amount:
            return False  # Insufficient credits: Transaction fails atomically

        # 2. Second pass: Deduct credits greedily starting from the earliest expiring grant
        remaining_to_deduct = amount
        temp_pool = []

        while remaining_to_deduct > 0 and self.active_grants:
            current_grant = heapq.heappop(self.active_grants)

            # If the grant hasn't started yet, skip deducting from it for now
            if current_grant.start_time > timestamp:
                temp_pool.append(current_grant)
                continue

            if current_grant.amount > remaining_to_deduct:
                # This grant can fully satisfy the remaining deduction
                current_grant.amount -= remaining_to_deduct
                remaining_to_deduct = 0
                heapq.heappush(self.active_grants, current_grant)
            else:
                # This grant is completely exhausted
                remaining_to_deduct -= current_grant.amount
                # We do not push it back, effectively deleting it

        # Put back any grants that haven't started yet
        for g in temp_pool:
            heapq.heappush(self.active_grants, g)

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
