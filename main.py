import numpy as np
import random
import math


def load_matrix(filename):
    return np.loadtxt(filename, delimiter=',')

try:
    dist_matrix = load_matrix('distance_matrix.csv')
    N_CITIES = len(dist_matrix)
    print(f"Матрица успешно загружена. Количество городов: {N_CITIES}")
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    exit()

def route_length(route):
    """Вычисление суммарной длины замкнутого маршрута"""
    return sum(dist_matrix[route[i], route[(i + 1) % N_CITIES]] for i in range(N_CITIES))


# ===============================
# 1. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ (GA)
# ===============================
def genetic_algorithm(pop_size=100, gens=1000, mut_rate=0.15):
    population = [random.sample(range(N_CITIES), N_CITIES) for _ in range(pop_size)]

    for _ in range(gens):
        population.sort(key=route_length)
        next_gen = population[:10]  # Элитизм: сохраняем 10 лучших

        while len(next_gen) < pop_size:
            # Селекция (турнир)
            p1, p2 = random.choices(population[:50], k=2)

            # Ordered Crossover (OX) - сохраняет структуру перестановки
            start, end = sorted(random.sample(range(N_CITIES), 2))
            child = [-1] * N_CITIES
            child[start:end] = p1[start:end]
            p2_filtered = [c for c in p2 if c not in child]
            child = [p2_filtered.pop(0) if c == -1 else c for c in child]

            # Мутация Swap
            if random.random() < mut_rate:
                i, j = random.sample(range(N_CITIES), 2)
                child[i], child[j] = child[j], child[i]
            next_gen.append(child)
        population = next_gen

    best = min(population, key=route_length)
    return best, route_length(best)


# ===============================
# 2. МУРАВЬИНЫЙ АЛГОРИТМ (ACO)
# ===============================
def ant_colony(ants=30, iters=100, alpha=1, beta=2.5, rho=0.1):
    pheromone = np.ones((N_CITIES, N_CITIES))
    best_route = None
    best_len = float('inf')

    for _ in range(iters):
        all_routes = []
        for _ in range(ants):
            route = [random.randint(0, N_CITIES - 1)]
            while len(route) < N_CITIES:
                curr = route[-1]
                # Вычисление вероятностей перехода
                probs = []
                for j in range(N_CITIES):
                    if j not in route:
                        eta = 1.0 / (dist_matrix[curr, j] + 1e-10)  # Видимость
                        probs.append((pheromone[curr, j] ** alpha) * (eta ** beta))
                    else:
                        probs.append(0)
                probs = np.array(probs) / sum(probs)
                next_city = np.random.choice(range(N_CITIES), p=probs)
                route.append(next_city)

            l = route_length(route)
            all_routes.append((route, l))
            if l < best_len:
                best_len = l
                best_route = route

        # Испарение и обновление феромонов
        pheromone *= (1 - rho)
        for route, length in all_routes:
            for i in range(N_CITIES):
                pheromone[route[i], route[(i + 1) % N_CITIES]] += 100 / length

    return best_route, best_len


# ===============================
# 3. ИМИТАЦИЯ ОТЖИГА (SA)
# ===============================
def simulated_annealing(t_init=5000, t_min=0.1, cooling_rate=0.995):
    curr_route = list(range(N_CITIES))
    random.shuffle(curr_route)
    curr_len = route_length(curr_route)

    best_route, best_len = curr_route[:], curr_len
    T = t_init

    while T > t_min:
        new_route = curr_route[:]
        # 2-opt мутация: инверсия сегмента
        i, j = sorted(random.sample(range(N_CITIES), 2))
        new_route[i:j] = reversed(new_route[i:j])
        new_len = route_length(new_route)

        delta = new_len - curr_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            curr_route, curr_len = new_route, new_len
            if curr_len < best_len:
                best_route, best_len = curr_route[:], curr_len
        T *= cooling_rate

    return best_route, best_len


# ===============================
# ЗАПУСК ТЕСТОВ
# ===============================
if __name__ == "__main__":
    print("-" * 30)
    print("Результаты работы алгоритмов:")

    # 1. GA
    r_ga, l_ga = genetic_algorithm()
    print(f"1. Генетический алгоритм:  Длина = {l_ga}")

    # 2. ACO
    r_aco, l_aco = ant_colony()
    print(f"2. Муравьиный алгоритм:    Длина = {l_aco}")

    # 3. SA
    r_sa, l_sa = simulated_annealing()
    print(f"3. Алгоритм отжига:        Длина = {l_sa}")
    print("-" * 30)
    print(f"Точный оптимум (для справки): 1899")