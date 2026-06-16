import main
# route = [1, 5, 3, 6, 2, 4, 7, 8, 9, 10]
# route = ["a", "b", "c", "d", "e", "f", "g", "h"]
# print(main.insertion(route))
# print(main.two_opt(route))
# print(main.swap(route))
# print(main.segment_shuffle(route))

weights, values = main.load_problem_knapsack("datasets/Delivery_Logistics.csv")
# print(values)
solution = main.knapsack_random_solution(weights, 1000)
print(main.knapsack_value(solution, weights, values, 1000))

solution = [1,0,0,1,0,1,1,1,0,1,1,1,0,1,1]
print(solution)
print(main.one_bit_flip(solution))
print(main.two_bit_flip(solution))
print(main.three_bit_flip(solution))
print(main.four_bit_flip(solution))
# print(main.knapsack_value(solution, weights, values, 100))