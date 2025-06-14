import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq

class OverlapGraph:
    """
    Класс для представления графа перекрытий, используемого при сборке генома.
    Включает методы для обработки проблем типа tips, bubbles и упрощения графа.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.read_counts = defaultdict(int)  # Для отслеживания покрытия

    def add_read(self, read_id, sequence):
        """Добавление рида в граф"""
        if read_id not in self.graph.nodes:
            self.graph.add_node(read_id, sequence=sequence)
            self.read_counts[read_id] = 1
        else:
            self.read_counts[read_id] += 1

    def add_edge(self, read_id1, read_id2, overlap_length):
        """Добавление ребра между ридами с указанием длины перекрытия"""
        self.graph.add_edge(read_id1, read_id2, overlap=overlap_length)

    def build_from_reads(self, reads_dict, min_overlap=50):
        """
        Построение графа перекрытий из словаря ридов.

        Parameters:
        - reads_dict: словарь вида {read_id: sequence}
        - min_overlap: минимальная длина перекрытия для создания ребра
        """
        # Добавляем все риды в граф
        for read_id, sequence in reads_dict.items():
            self.add_read(read_id, sequence)

        # Находим перекрытия между всеми парами ридов
        read_ids = list(reads_dict.keys())
        for i in tqdm(range(len(read_ids))):
            for j in range(len(read_ids)):
                if i != j:
                    read_id1, read_id2 = read_ids[i], read_ids[j]
                    seq1, seq2 = reads_dict[read_id1], reads_dict[read_id2]

                    # Находим максимальное перекрытие между концом первого рида и началом второго
                    max_overlap = 0
                    for k in range(min_overlap, min(len(seq1), len(seq2)) + 1):
                        if seq1[-k:] == seq2[:k]:
                            max_overlap = k

                    if max_overlap >= min_overlap:
                        self.add_edge(read_id1, read_id2, max_overlap)

                    # Проверяем перекрытие только длиной min_overlap
                    if seq1[-min_overlap:] == seq2[:min_overlap]:
                        self.add_edge(read_id1, read_id2, min_overlap)

    def remove_tips(self, max_tip_length=50):
        """
        Удаление тупиковых путей (tips) из графа.
        Tips - это короткие пути, которые не имеют продолжения.

        Parameters:
        - max_tip_length: максимальная длина тупика для удаления
        """
        modified = True

        while modified:
            modified = False

            # Находим вершины с нулевой исходящей степенью (потенциальные кончики тупиков)
            potential_tips = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

            for tip_end in potential_tips:
                # Проходим в обратном направлении от кончика тупика
                tip_path = [tip_end]
                current = tip_end

                while True:
                    # Получаем все входящие вершины
                    predecessors = list(self.graph.predecessors(current))

                    # Если нет входящих вершин или их больше одной - это не тупик
                    if len(predecessors) != 1:
                        break

                    # Если входящая вершина имеет больше одного исходящего ребра - это ответвление, не тупик
                    predecessor = predecessors[0]
                    if self.graph.out_degree(predecessor) > 1:
                        # Мы нашли корень тупика
                        tip_path.append(predecessor)
                        break

                    # Добавляем предшественника в путь и продолжаем
                    current = predecessor
                    tip_path.append(current)

                    # Проверка на длину тупика
                    if len(tip_path) > max_tip_length:
                        break

                # Если длина тупика меньше максимально допустимой, удаляем его
                if 1 < len(tip_path) <= max_tip_length:
                    # Удаляем только участок тупика после точки ветвления
                    for i in range(len(tip_path) - 1):
                        if self.graph.has_edge(tip_path[i+1], tip_path[i]):
                            self.graph.remove_edge(tip_path[i+1], tip_path[i])

                    modified = True

            # Повторяем для входящих тупиков (у которых нет входящих рёбер)
            potential_tips = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

            for tip_start in potential_tips:
                tip_path = [tip_start]
                current = tip_start

                while True:
                    successors = list(self.graph.successors(current))

                    if len(successors) != 1:
                        break

                    successor = successors[0]
                    if self.graph.in_degree(successor) > 1:
                        tip_path.append(successor)
                        break

                    current = successor
                    tip_path.append(current)

                    if len(tip_path) > max_tip_length:
                        break

                if 1 < len(tip_path) <= max_tip_length:
                    for i in range(len(tip_path) - 1):
                        if self.graph.has_edge(tip_path[i], tip_path[i+1]):
                            self.graph.remove_edge(tip_path[i], tip_path[i+1])

                    modified = True

    def remove_bubbles(self, max_bubble_length=100):
        """
        Удаление пузырей (bubbles) из графа.
        Bubble - это две или более альтернативные пути между двумя вершинами.

        Parameters:
        - max_bubble_length: максимальная длина пузыря для обработки
        """
        bubbles_removed = 0

        # Находим все вершины с более чем одним исходящим ребром (потенциальные начала пузырей)
        for start_node in list(self.graph.nodes):
            if self.graph.out_degree(start_node) <= 1:
                continue

            # Используем поиск в ширину (BFS) для нахождения всех путей от стартовой вершины
            # в пределах максимальной длины пузыря
            visited = {}  # Словарь {node: [paths_to_node]}
            queue = deque([(start_node, [start_node])])

            while queue:
                node, path = queue.popleft()

                # Пропускаем слишком длинные пути
                if len(path) > max_bubble_length + 1:  # +1 для учета начальной точки
                    continue

                # Добавляем путь к текущей вершине
                if node not in visited:
                    visited[node] = []
                visited[node].append(path)

                # Продолжаем поиск
                for neighbor in self.graph.successors(node):
                    # Избегаем циклов в пути
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))

            # Ищем пузыри - вершины, к которым ведет более одного пути от start_node
            for end_node, paths in visited.items():
                if end_node == start_node or len(paths) < 2:
                    continue

                # Проверяем, что пути расходятся сразу после start_node
                # и являются действительно различными путями
                divergent_paths = []
                next_nodes = set()

                for path in paths:
                    if len(path) >= 2 and path[0] == start_node:
                        next_node = path[1]
                        if next_node not in next_nodes:
                            next_nodes.add(next_node)
                            divergent_paths.append(path)

                # Если есть несколько различных путей, считаем это пузырем
                if len(divergent_paths) >= 2:
                    # Вычисляем покрытие для каждого пути
                    path_coverages = []
                    for path in divergent_paths:
                        # Среднее покрытие пути
                        path_coverage = sum(self.read_counts[node] for node in path) / len(path)
                        path_coverages.append((path_coverage, path))

                    # Сортируем по покрытию (от наибольшего к наименьшему)
                    path_coverages.sort(reverse=True)

                    # Оставляем путь с наибольшим покрытием
                    best_path = path_coverages[0][1]

                    # Удаляем альтернативные пути
                    for _, path in path_coverages[1:]:
                        # Удаляем рёбра только для внутренних узлов пути
                        # (сохраняем начальную и конечную вершины)
                        for i in range(len(path) - 1):
                            if path[i] != start_node or path[i+1] != end_node:
                                if self.graph.has_edge(path[i], path[i+1]):
                                    self.graph.remove_edge(path[i], path[i+1])

                    bubbles_removed += 1

        return bubbles_removed

    def simplify_linear_paths(self):
        """
        Упрощение линейных участков графа, где между вершинами только одна дуга.
        Объединяет последовательные вершины в одну.
        """
        modified = True

        while modified:
            modified = False

            for node in list(self.graph.nodes):
                # Проверяем, является ли вершина частью линейного пути
                if self.graph.in_degree(node) == 1 and self.graph.out_degree(node) == 1:
                    predecessor = list(self.graph.predecessors(node))[0]
                    successor = list(self.graph.successors(node))[0]

                    # Избегаем циклы
                    if predecessor == successor or predecessor == node or successor == node:
                        continue

                    # Объединяем вершины, если это возможно
                    if self.graph.out_degree(predecessor) == 1 and self.graph.in_degree(successor) == 1:
                        # Получаем последовательности
                        pred_seq = self.graph.nodes[predecessor].get('sequence', '')
                        node_seq = self.graph.nodes[node].get('sequence', '')
                        succ_seq = self.graph.nodes[successor].get('sequence', '')

                        # Вычисляем перекрытия
                        pred_overlap = self.graph.edges[predecessor, node].get('overlap', 0)
                        node_overlap = self.graph.edges[node, successor].get('overlap', 0)

                        # Объединяем последовательности
                        if pred_seq and node_seq and succ_seq:
                            # Объединяем с учетом перекрытий
                            combined_seq = pred_seq[:-pred_overlap] + node_seq[:-node_overlap] + succ_seq

                            # Создаем новую вершину
                            new_node_id = f"{predecessor}_{node}_{successor}"
                            self.graph.add_node(new_node_id, sequence=combined_seq)

                            # Переносим рёбра
                            for pred_of_pred in self.graph.predecessors(predecessor):
                                if pred_of_pred != node and pred_of_pred != successor:
                                    overlap = self.graph.edges[pred_of_pred, predecessor].get('overlap', 0)
                                    self.graph.add_edge(pred_of_pred, new_node_id, overlap=overlap)

                            for succ_of_succ in self.graph.successors(successor):
                                if succ_of_succ != node and succ_of_succ != predecessor:
                                    overlap = self.graph.edges[successor, succ_of_succ].get('overlap', 0)
                                    self.graph.add_edge(new_node_id, succ_of_succ, overlap=overlap)

                            # Обновляем счетчик покрытия
                            self.read_counts[new_node_id] = (self.read_counts[predecessor] +
                                                             self.read_counts[node] +
                                                             self.read_counts[successor]) / 3

                            # Удаляем старые вершины
                            self.graph.remove_node(predecessor)
                            self.graph.remove_node(node)
                            self.graph.remove_node(successor)

                            modified = True
                            break

        return modified

    def assemble_contigs(self):
        """
        Собирает контиги из графа перекрытий.
        Возвращает список контигов.
        """
        contigs = []

        # Находим компоненты связности в графе
        for component in nx.weakly_connected_components(self.graph):
            subgraph = self.graph.subgraph(component)

            # Находим длинные пути в компоненте
            paths = []

            # Стартуем от вершин с нулевой входящей степенью
            start_nodes = [node for node in subgraph.nodes if subgraph.in_degree(node) == 0]

            # Если таких нет, выбираем любую вершину
            if not start_nodes:
                start_nodes = list(subgraph.nodes)

            for start_node in start_nodes:
                # Используем поиск длиннейшего пути в DAG
                # Если граф не является DAG, можно использовать другие алгоритмы
                try:
                    paths_from_node = nx.all_simple_paths(subgraph, start_node,
                                                        [node for node in subgraph.nodes if subgraph.out_degree(node) == 0])
                    paths.extend(list(paths_from_node))
                except nx.NetworkXNoPath:
                    continue

            if not paths:
                continue

            # Выбираем самый длинный путь
            longest_path = max(paths, key=len, default=[])

            if not longest_path:
                continue

            # Собираем контиг из последовательностей на пути
            contig = ""
            for i, node in enumerate(longest_path):
                if i == 0:
                    contig = subgraph.nodes[node].get('sequence', '')
                else:
                    prev_node = longest_path[i-1]
                    overlap = subgraph.edges[prev_node, node].get('overlap', 0)
                    seq = subgraph.nodes[node].get('sequence', '')
                    if overlap > 0 and seq:
                        contig += seq[overlap:]
                    else:
                        contig += seq

            contigs.append(contig)

        return contigs

        # contigs = []
        # used_nodes = set()

        # # 1. Добавляем изолированные вершины
        # for node in self.graph.nodes:
        #     if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) == 0:
        #         seq = self.graph.nodes[node].get("sequence", "")
        #         if seq:
        #             contigs.append(seq)
        #             used_nodes.add(node)

        # # 2. Поиск гамильтоновых путей
        # def backtrack(path, visited):
        #     if len(visited) == total_nodes:
        #         return path  # полный путь найден

        #     last_node = path[-1]
        #     for neighbor in self.graph.successors(last_node):
        #         if neighbor not in visited and neighbor not in used_nodes:
        #             visited.add(neighbor)
        #             result = backtrack(path + [neighbor], visited)
        #             if result:
        #                 return result
        #             visited.remove(neighbor)
        #     return None

        # total_nodes = len(self.graph.nodes)

        # for start_node in self.graph.nodes:
        #     if start_node in used_nodes:
        #         continue

        #     path = backtrack([start_node], {start_node})
        #     if path:
        #         # Собираем последовательность
        #         contig = self.graph.nodes[path[0]].get("sequence", "")
        #         for i in range(1, len(path)):
        #             prev = path[i - 1]
        #             curr = path[i]
        #             overlap = self.graph.edges[prev, curr].get("overlap", 0)
        #             curr_seq = self.graph.nodes[curr].get("sequence", "")
        #             contig += curr_seq[overlap:]
        #         contigs.append(contig)

        #         # Помечаем узлы как использованные
        #         used_nodes.update(path)

        # return contigs

    def visualize(self, filename=None):
        """
        Визуализирует граф перекрытий.

        Parameters:
        - filename: имя файла для сохранения изображения. Если None, отображает график.
        """
        plt.figure(figsize=(12, 8))

        # Позиции вершин
        pos = nx.spring_layout(self.graph)

        # Рисуем вершины
        nx.draw_networkx_nodes(self.graph, pos, node_size=1500, node_color='lightblue')

        # Рисуем рёбра
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, min_source_margin=15, min_target_margin=20)

        # Рисуем метки вершин
        nx.draw_networkx_labels(self.graph, pos, font_size=10)

        # Рисуем метки рёбер (длина перекрытия)
        edge_labels = {(u, v): d.get('overlap', '') for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Граф перекрытий")
        plt.axis('off')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()