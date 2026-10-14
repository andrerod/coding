def run_allocator(n, operations):
    free = [(0, n)]  # Half-open intervals [start, end)
    owners = {}
    result = []

    for op in operations:
        if op[0] == 1:
            _, size, owner = op
            allocated = -1

            for i, (start, end) in enumerate(free):
                if end - start >= size:
                    allocated = start
                    if end - start == size:
                        free.pop(i)
                    else:
                        free[i] = (start + size, end)

                    if owner not in owners:
                        owners[owner] = []
                    owners[owner].append((start, size))
                    break

            result.append(allocated)

        else:
            _, owner = op
            blocks = owners.pop(owner, [])
            result.append(sum(size for _, size in blocks))

            for start, size in blocks:
                free.append((start, start + size))

            free.sort()
            merged = []
            for start, end in free:
                if merged and merged[-1][1] == start:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            free = merged

    return result
