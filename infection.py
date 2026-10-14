def infect_adjacent(grid, x, y):
    additional_infected = 0

    for n_x, n_y in [(x-1, y), (x+1, y), (x, y - 1), (x, y + 1)]:
        if n_x < 0 or n_x >= len(grid):
            continue

        if n_y < 0 or n_y >= len(grid[n_x]):
            continue

        if grid[n_x][n_y] == 1:
            grid[n_x][n_y] = 2
            additional_infected += 1

    return additional_infected

def all_infected(grid):
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y] == 1:
                return False

    return True

def infect_for_day(grid):
    original = [row[:] for row in grid]
    additional_infected = 0

    for x in range(len(original)):
        for y in range(len(original[x])):
            if original[x][y] == 2:
                additional_infected += infect_adjacent(grid, x, y)

    return additional_infected

def days_until_all_infected(grid):
    days = 0
    while True:
        days += 1
        new_infected = infect_for_day(grid)

        if new_infected == 0:
            if all_infected(grid):
                if days == 1:
                    return 0
                else:
                    return max(0, days - 1)
            else:
                return -1

    return -1
