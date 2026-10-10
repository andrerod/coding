import heapq
from concurrent.futures import as_completed, ProcessPoolExecutor

# 0 -> 1 -> 2
#  \        |
#   \> 3 -> 4


def kahns(nodes, adjacent):
    deps = [0] * nodes

    for org, dst in adjacent:
        deps[org] += 1

    visited = []

    to_visit = [n for n in range(len(deps)) if deps[n] == 0]
    heapq.heapify(to_visit)

    while to_visit:
        current_idx = heapq.heappop(to_visit)
        visited.append(current_idx)

        for org, dst in adjacent:
            if dst == current_idx:
                deps[org] -= 1

                if deps[org] == 0:
                    heapq.heappush(to_visit, org)

    # check for cycles
    if nodes != len(visited):
        raise Exception("loops found")

    return visited


def dijsktra(nodes, graph, start, end):
    distances = [999] * nodes
    distances[start] = 0

    to_visit = [(0, start)]
    prev = {}

    while to_visit:
        current_distance, current_idx = heapq.heappop(to_visit)

        if current_idx == end:
            return distances[current_idx], prev

        if current_distance > distances[current_idx]:
            continue

        for org, dst in graph:
            if org == current_idx:
                proposed_distance = current_distance + 1
                if proposed_distance < distances[dst]:
                    distances[dst] = proposed_distance
                    prev[dst] = current_idx
                    heapq.heappush(to_visit, (proposed_distance, dst))

    # No path to end
    return (-1, prev)


def print_path(prev, start, end):
    path = [end]

    current_idx = end
    while current_idx != start:
        previous_idx = prev[current_idx]
        path.append(previous_idx)
        current_idx = previous_idx

    path.reverse()
    return path


def main():
    graph = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4), (2, 4)]

    with ProcessPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(kahns, 5, graph))
        futures.append(executor.submit(dijsktra, 5, graph, 0, 4))

        for future in as_completed(futures):
            res = future.result()

            if isinstance(res, list):
                print(res)
            else:
                shortest, prev = res
                print(f"shortest is {shortest} {print_path(prev, 0, 4)}")

        print("all done")


if __name__ == "__main__":
    main()
