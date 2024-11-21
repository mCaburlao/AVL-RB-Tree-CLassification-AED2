import time
from avl import AVLTree  # Supondo que a classe AVLTree esteja no arquivo avl_tree.py
from rb import RedBlackTree  # Supondo que a classe RedBlackTree esteja no arquivo red_black_tree.py

# Função para medir o tempo de inserção, busca e remoção
def benchmark_tree_operations(tree_class, num_operations=1000):
    tree = tree_class()

    # Medir tempo de inserção
    start_time = time.time()
    for i in range(num_operations):
        if isinstance(tree, AVLTree):
            tree.insert_value(i)  # Para AVL, usamos insert_value
        elif isinstance(tree, RedBlackTree):
            tree.insert(i)  # Para Red-Black, usamos insert
    insert_time = time.time() - start_time

    # Medir tempo de busca
    start_time = time.time()
    for i in range(num_operations):
        if isinstance(tree, AVLTree):
            tree.search_value(i)  # Para AVL
        elif isinstance(tree, RedBlackTree):
            tree.searchTree(i)  # Para Red-Black
    search_time = time.time() - start_time

    # Medir tempo de remoção
    start_time = time.time()
    for i in range(num_operations):
        if isinstance(tree, AVLTree):
            tree.delete_value(i)  # Para AVL
        elif isinstance(tree, RedBlackTree):
             tree.delete_node(i)
    delete_time = time.time() - start_time

    return insert_time, search_time, delete_time

# Função principal para rodar o benchmark
def run_benchmark():
    num_operations = 1000  # Número de operações a serem realizadas

    # Benchmark para AVL Tree
    print("Benchmarking AVL Tree...")
    avl_insert_time, avl_search_time, avl_delete_time = benchmark_tree_operations(AVLTree, num_operations)
    print(f"AVL Tree - Inserção: {avl_insert_time:.6f}s, Busca: {avl_search_time:.6f}s, Remoção: {avl_delete_time:.6f}s")

    # Benchmark para Red-Black Tree
    print("Benchmarking Red-Black Tree...")
    rb_insert_time, rb_search_time, rb_delete_time = benchmark_tree_operations(RedBlackTree, num_operations)
    print(f"Red-Black Tree - Inserção: {rb_insert_time:.6f}s, Busca: {rb_search_time:.6f}s, Remoção: {rb_delete_time:.6f}s")

# Rodando o benchmark
if __name__ == "__main__":
    run_benchmark()
