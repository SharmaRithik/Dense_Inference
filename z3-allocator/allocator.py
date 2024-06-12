from z3 import *

# Define the optimizer
opt = Optimize()

# Variables for each task's core type
task_cores = [Int(f"task_{i}_core") for i in range(5)]  # 5 tasks

# Variables to hold the total time for each core type
time_small_cores = Real("time_small_cores")
time_medium_cores = Real("time_medium_cores")
time_big_cores = Real("time_big_cores")

# Time matrices from your data
times = [
    [1.73742, 0.568922, 0.520377],  # Task 1
    [1.73681, 0.516808, 0.48112],   # Task 2
    [1.73397, 0.520244, 0.535664],  # Task 3
    [1.7398, 0.539943, 0.483729],   # Task 4
    [1.75972, 0.568565, 0.520906]   # Task 5
]

# Constraint for valid core choices and calculating total times
for var in task_cores:
    opt.add(Or(var == 0, var == 1, var == 2))

opt.add(time_small_cores == Sum([If(task_cores[i] == 0, times[i][0], 0) for i in range(5)]))
opt.add(time_medium_cores == Sum([If(task_cores[i] == 1, times[i][1], 0) for i in range(5)]))
opt.add(time_big_cores == Sum([If(task_cores[i] == 2, times[i][2], 0) for i in range(5)]))

# Ensure contiguous allocation of cores by blocks
block_start = [Bool(f"block_start_{i}") for i in range(5)]
opt.add(block_start[0] == True)  # First task is always the start of a block

for i in range(1, 5):
    opt.add(Implies(block_start[i], task_cores[i] != task_cores[i-1]))  # New block starts when core type changes
    opt.add(Implies(Not(block_start[i]), task_cores[i] == task_cores[i-1]))  # Same block means same core type

# Allow only a limited number of blocks (contiguous segments)
max_blocks = 3  # Adjust this number as needed
opt.add(Sum([If(block_start[i], 1, 0) for i in range(5)]) <= max_blocks)

# Define a helper function to calculate the maximum of three real numbers
def max3(x, y, z):
    return If(x > y, If(x > z, x, z), If(y > z, y, z))

# Objective: minimize the maximum execution time among the cores
max_time = max3(time_small_cores, time_medium_cores, time_big_cores)
opt.minimize(max_time)

# Solve the problem
if opt.check() == sat:
    model = opt.model()
    core_allocations = [model.eval(task_cores[i]) for i in range(5)]
    times_small = model.eval(time_small_cores)
    times_medium = model.eval(time_medium_cores)
    times_big = model.eval(time_big_cores)
    optimal_max_time = model.eval(max_time)
    print("Optimal core allocations:", core_allocations)
    print("Total time on small cores:", times_small)
    print("Total time on medium cores:", times_medium)
    print("Total time on big cores:", times_big)
    print("Optimal total time to complete all tasks:", optimal_max_time)
else:
    print("Failed to find an optimal solution")

