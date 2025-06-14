import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import numpy as np
import logging

class DeBruijnGraph:
    def __init__(self, k=21):
        """
        Инициализация графа Де Брюйна.

        Parameters:
        - k: длина k-мера (по умолчанию 21)
        """
        self.k = k
        self.graph = nx.DiGraph()

    def _get_kmers(self, sequence):
        """
        Генерация k-меров из последовательности.
        """
        if len(sequence) < self.k:
            return []
        return [sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)]

    def add_sequence(self, sequence):
        """
        Добавление k-меров из последовательности в граф.
        Узлы — (k-1)-меры, рёбра — k-меры.
        """
        kmers = self._get_kmers(sequence)
        for kmer in kmers:
            prefix = kmer[:-1]
            suffix = kmer[1:]
            if self.graph.has_edge(prefix, suffix):
                self.graph[prefix][suffix]['weight'] += 1
                self.graph[prefix][suffix]['kmers'].append(kmer)
            else:
                self.graph.add_edge(prefix, suffix, weight=1, kmers=[kmer])

    def build_from_reads(self, reads):
        """
        Построение графа из набора ридов.
        """
        for read in reads:
            self.add_sequence(read)

    def visualize(self, filename=None, with_labels=True, node_size=500):
        """
        Визуализация графа Де Брюйна.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        edge_weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + 5 * (w / max_weight) for w in edge_weights]

        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color='skyblue')
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.7)

        if with_labels and len(self.graph.nodes) <= 50:
            nx.draw_networkx_labels(self.graph, pos, font_size=8)

        plt.title("Граф Де Брюйна")
        plt.axis('off')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def remove_tips(self):
      """
      Удаляет 'типы' — изолированные короткие пути (длиной 1), которые начинаются или заканчиваются тупиками.
      """
      to_remove = set()

      for node in list(self.graph.nodes):
          # Удаляем исходящие типы
          if self.graph.out_degree(node) == 1 and self.graph.in_degree(node) == 0:
              succ = next(self.graph.successors(node))
              if self.graph.in_degree(succ) == 1:
                  to_remove.add((node, succ))

          # Удаляем входящие типы
          elif self.graph.in_degree(node) == 1 and self.graph.out_degree(node) == 0:
              pred = next(self.graph.predecessors(node))
              if self.graph.out_degree(pred) == 1:
                  to_remove.add((pred, node))

      for u, v in to_remove:
          if self.graph.has_edge(u, v):
              self.graph.remove_edge(u, v)
          # Удаляем узлы, если у них больше нет рёбер
          if u in self.graph and self.graph.degree(u) == 0:
              self.graph.remove_node(u)
          if v in self.graph and self.graph.degree(v) == 0:
              self.graph.remove_node(v)


    def remove_bubbles(self):
        """
        Удаляет 'пузыри' — альтернативные пути между двумя одинаковыми узлами начала и конца.
        """
        for node in list(self.graph.nodes):
            successors = list(self.graph.successors(node))
            if len(successors) <= 1:
                continue

            for i in range(len(successors)):
                for j in range(i + 1, len(successors)):
                    path1 = nx.shortest_path(self.graph, source=successors[i], target=None, method='dijkstra')
                    path2 = nx.shortest_path(self.graph, source=successors[j], target=None, method='dijkstra')
                    common_nodes = set(path1).intersection(set(path2))
                    for target in common_nodes:
                        if target != node:
                            # Сравним длины путей
                            try:
                                p1 = nx.shortest_path(self.graph, node, target)
                                p2 = nx.shortest_path(self.graph, node, target)
                                if len(p1) != len(p2):
                                    to_remove = p1 if len(p1) > len(p2) else p2
                                    for u, v in zip(to_remove, to_remove[1:]):
                                        if self.graph.has_edge(u, v):
                                            self.graph.remove_edge(u, v)
                                    break
                            except nx.NetworkXNoPath:
                                continue

    def simplify_linear_paths(self):
        """
        Упрощает линейные пути в графе, объединяя их в суперузлы.
        """
        paths_to_merge = []
        visited = set()

        for node in list(self.graph.nodes):
            if node in visited:
                continue
            if self.graph.in_degree(node) != 1 or self.graph.out_degree(node) != 1:
                for succ in list(self.graph.successors(node)):
                    path = [node, succ]
                    current = succ
                    while (current in self.graph and
                           self.graph.in_degree(current) == 1 and
                           self.graph.out_degree(current) == 1):
                        next_node = next(iter(self.graph.successors(current)), None)
                        if next_node is None or next_node in path:
                            break
                        path.append(next_node)
                        current = next_node
                    if len(path) > 2:
                        paths_to_merge.append(path)
                        visited.update(path)

        for path in paths_to_merge:
            # Создаём объединённую метку
            merged_label = path[0]
            for node in path[1:]:
                merged_label += node[-1]

            self.graph.add_node(merged_label)

            # Добавляем рёбра от предков первого узла
            for pred in list(self.graph.predecessors(path[0])):
                if pred not in path and self.graph.has_node(pred):
                    self.graph.add_edge(pred, merged_label, weight=1, kmers=[])

            # Добавляем рёбра к потомкам последнего узла
            for succ in list(self.graph.successors(path[-1])):
                if succ not in path and self.graph.has_node(succ):
                    self.graph.add_edge(merged_label, succ, weight=1, kmers=[])

        # Теперь удаляем старые узлы и рёбра
        for path in paths_to_merge:
            for u, v in zip(path, path[1:]):
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
            for node in path:
                if self.graph.has_node(node):
                    self.graph.remove_node(node)

    def assemble_contigs(self):
      """
      Сборка контигов из графа.
      Для каждой слабой компоненты связности ищется самый длинный путь (не обязательно эйлеров).
      Возвращает список строк (контигов).
      """
      contigs = []

      for component in nx.weakly_connected_components(self.graph):
          subgraph = self.graph.subgraph(component).copy()

          # Если подграф изолированной вершины
          if subgraph.number_of_edges() == 0:
              contigs.extend(list(subgraph.nodes))
              continue

          # Нахождение самого длинного пути в подграфе
          def dfs_longest_path(graph, start, visited, path):
              visited.add(start)
              path.append(start)

              max_path = list(path)
              for neighbor in graph.successors(start):
                  if neighbor not in visited:
                      new_path = dfs_longest_path(graph, neighbor, visited.copy(), list(path))
                      if len(new_path) > len(max_path):
                          max_path = new_path
              return max_path

          longest = []
          for node in subgraph.nodes:
              path = dfs_longest_path(subgraph, node, set(), [])
              if len(path) > len(longest):
                  longest = path

          # Преобразуем путь узлов в строку
          if longest:
              sequence = longest[0]
              for node in longest[1:]:
                  sequence += node[-1]
              contigs.append(sequence)

      return contigs

def through_model(self, model_path="classifier.pkl", scaler_path="scaler.pkl", log_path="model_decisions.log"):
    """
    Применяет обученную модель для фильтрации рёбер в графе:
    - Находит разветвляющиеся узлы.
    - Для каждого кандидата рассчитывает признаки.
    - Прогоняет их через модель.
    - Оставляет только наиболее вероятное ребро.
    - Все действия логируются в файл.
    """

    # Настройка логгера
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)s: %(message)s')
    #print("=== Start through_model ===")

    # Загрузка модели и нормализатора
    model = load("classifier.pkl")
    scaler = load("scaler.pkl")

    k = self.k
    total_nodes = 0
    total_removed = 0

    for node in list(self.graph.nodes):
        successors = list(self.graph.successors(node))
        if len(successors) <= 1:
            continue

        total_nodes += 1
        features = []
        candidates = []

        #print(f"\nNode: {node} has {len(successors)} outgoing edges.")

        for succ in successors:
            edge_data = self.graph.get_edge_data(node, succ)
            if not edge_data:
                continue

            overlap_len = get_overlap_length(node, succ)
            if overlap_len <= 10:
                #print(f"  Skipped edge {node} -> {succ}: overlap too short ({overlap_len})")
                continue

            # Восстановим "идеальное" перекрытие
            # reconstructed = node[-(k - 1):] + succ[-1]
            # kmer_seq = kmers[0]

            mismatches = 0
            mismatch_pct = 0

            logging.info(f"  Edge {node} -> {succ}: overlap={overlap_len}, mismatches={mismatches}, mismatch_pct={mismatch_pct:.2f}")

            feat = [overlap_len, mismatch_pct, mismatches]
            #print(feat)
            features.append(feat)
            candidates.append((node, succ))

        #print(features)
        if len(features) <= 1:
            #print(f"  Not enough candidates to evaluate.\n")
            continue

        X_scaled = scaler.transform(features)
        probs = model.predict_proba(X_scaled)[:, 1]

        #print(probs)

        max_idx = np.argmax(probs)
        for i, (u, v) in enumerate(candidates):
            print(f"    Candidate {u}->{v}: proba={probs[i]:.4f}")
            if i != max_idx:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    total_removed += 1
                    #print(f"    Removed edge {u}->{v} (lower score)\n")
            #else:
                #print(f"    Kept edge {u}->{v} (highest score)\n")

    print(f"=== Finished through_model ===")
    logging.info(f"Processed {total_nodes} branching nodes, removed {total_removed} edges.\n")
    print(f"[through_model] Done: {total_nodes} branching nodes processed, {total_removed} edges removed. Log saved to '{log_path}'.")

# Добавить в класс
DeBruijnGraph.through_model = through_model