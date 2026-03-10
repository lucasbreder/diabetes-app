"""
Algoritmo Genético para Otimização de Hiperparâmetros
=====================================================
Módulo que implementa um AG completo com:
- Codificação de genes para hiperparâmetros de cada modelo
- Operadores de seleção (torneio), cruzamento (uniforme) e mutação
- Função fitness multiobjetivo (recall, especificidade, F1, equidade)
- Comparação com modelos baseline (sem otimização)
- Suporte a múltiplos experimentos com configurações distintas
"""

import random
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. ESPAÇO DE HIPERPARÂMETROS (representação dos genes)
# ============================================================

HYPERPARAMETER_SPACE = {
    "KNeighborsClassifier": {
        "n_neighbors":  {"type": "int",  "low": 1,    "high": 30},
        "weights":      {"type": "cat",  "choices": ["uniform", "distance"]},
        "metric":       {"type": "cat",  "choices": ["euclidean", "manhattan", "minkowski"]},
        "p":            {"type": "int",  "low": 1,    "high": 5},
    },
    "DecisionTreeClassifier": {
        "max_depth":         {"type": "int",  "low": 2,   "high": 30},
        "min_samples_split": {"type": "int",  "low": 2,   "high": 40},
        "min_samples_leaf":  {"type": "int",  "low": 1,   "high": 20},
        "criterion":         {"type": "cat",  "choices": ["gini", "entropy"]},
        "class_weight":      {"type": "cat",  "choices": [None, "balanced"]},
    },
    "LogisticRegression": {
        "C":            {"type": "float", "low": 0.001, "high": 100.0},
        "max_iter":     {"type": "int",   "low": 100,   "high": 2000},
        "solver":       {"type": "cat",   "choices": ["lbfgs", "liblinear", "saga"]},
        "class_weight": {"type": "cat",   "choices": [None, "balanced"]},
    },
    "RandomForestClassifier": {
        "n_estimators":      {"type": "int",  "low": 50,  "high": 800},
        "max_depth":         {"type": "int",  "low": 2,   "high": 30},
        "min_samples_split": {"type": "int",  "low": 2,   "high": 40},
        "min_samples_leaf":  {"type": "int",  "low": 1,   "high": 20},
        "criterion":         {"type": "cat",  "choices": ["gini", "entropy"]},
        "class_weight":      {"type": "cat",  "choices": [None, "balanced", "balanced_subsample"]},
    },
}


# ============================================================
# 2. GERAÇÃO DE INDIVÍDUOS (cromossomos)
# ============================================================

def random_gene(gene_spec):
    """Gera um valor aleatório para um gene de acordo com sua especificação."""
    if gene_spec["type"] == "int":
        return random.randint(gene_spec["low"], gene_spec["high"])
    elif gene_spec["type"] == "float":
        return random.uniform(gene_spec["low"], gene_spec["high"])
    elif gene_spec["type"] == "cat":
        return random.choice(gene_spec["choices"])


def create_individual(model_name):
    """Cria um indivíduo (cromossomo) com genes aleatórios."""
    space = HYPERPARAMETER_SPACE[model_name]
    return {param: random_gene(spec) for param, spec in space.items()}


def create_population(model_name, pop_size):
    """Cria uma população inicial de indivíduos."""
    return [create_individual(model_name) for _ in range(pop_size)]


# ============================================================
# 3. FUNÇÃO FITNESS — multiobjetivo para saúde da mulher
# ============================================================

def _compute_specificity(y_true, y_pred):
    """Especificidade = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _compute_equity(y_true, y_pred, age_groups):
    """
    Equidade entre grupos etários: 1 - desvio padrão do recall por grupo.
    Quanto menor a variância do recall entre grupos, maior a equidade.
    """
    recalls = []
    for group_label in sorted(age_groups.unique()):
        mask = age_groups == group_label
        if mask.sum() == 0:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        # Só calcula se houver positivos no grupo
        if y_t.sum() > 0:
            recalls.append(recall_score(y_t, y_p, zero_division=0))

    if len(recalls) < 2:
        return 1.0  # sem variação possível
    return max(0.0, 1.0 - np.std(recalls))


def build_model(model_name, params):
    """Instancia um modelo sklearn a partir do nome e dos hiperparâmetros."""
    constructors = {
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
    }
    return constructors[model_name](**params)


def fitness(individual, model_name, X, y, ages,
            w_recall=0.40, w_specificity=0.20, w_f1=0.25, w_equity=0.15,
            cv_folds=5):
    """
    Avalia o fitness de um indivíduo usando validação cruzada estratificada.

    Pesos padrão:
      - Recall (sensibilidade): 40% — prioridade para doenças críticas
      - Especificidade: 20% — evitar alarmes falsos
      - F1-score: 25% — equilíbrio precisão/sensibilidade
      - Equidade: 15% — desempenho justo entre faixas etárias
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    recalls, specificities, f1s, equities = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        age_test = ages.iloc[test_idx]

        try:
            model = build_model(model_name, individual)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            specificities.append(_compute_specificity(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            equities.append(_compute_equity(y_test, y_pred, age_test))
        except Exception:
            # Combinação inválida de hiperparâmetros ⇒ fitness = 0
            return 0.0

    score = (
        w_recall * np.mean(recalls)
        + w_specificity * np.mean(specificities)
        + w_f1 * np.mean(f1s)
        + w_equity * np.mean(equities)
    )
    return score


# ============================================================
# 4. OPERADORES GENÉTICOS
# ============================================================

# ---- Seleção por torneio ----
def tournament_selection(population, fitnesses, k=3):
    """Seleciona um indivíduo via torneio de tamanho k."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[best_idx])


# ---- Cruzamento uniforme ----
def crossover(parent1, parent2, crossover_rate=0.8):
    """Cruzamento uniforme: cada gene é herdado de um dos pais com 50%."""
    if random.random() > crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    child1, child2 = {}, {}
    for gene in parent1:
        if random.random() < 0.5:
            child1[gene] = copy.deepcopy(parent1[gene])
            child2[gene] = copy.deepcopy(parent2[gene])
        else:
            child1[gene] = copy.deepcopy(parent2[gene])
            child2[gene] = copy.deepcopy(parent1[gene])
    return child1, child2


# ---- Mutação ----
def mutate(individual, model_name, mutation_rate=0.15):
    """Mutação: cada gene tem chance de ser substituído por valor aleatório."""
    space = HYPERPARAMETER_SPACE[model_name]
    mutant = copy.deepcopy(individual)
    for gene, spec in space.items():
        if random.random() < mutation_rate:
            mutant[gene] = random_gene(spec)
    return mutant


# ============================================================
# 5. LOOP PRINCIPAL DO ALGORITMO GENÉTICO
# ============================================================

def genetic_algorithm(
    model_name,
    X, y, ages,
    pop_size=30,
    generations=25,
    crossover_rate=0.8,
    mutation_rate=0.15,
    tournament_k=3,
    elitism_count=2,
    fitness_weights=None,
    verbose=True,
):
    """
    Executa o algoritmo genético completo.

    Retorna:
      - best_individual: melhor conjunto de hiperparâmetros encontrado
      - best_fitness: fitness do melhor indivíduo
      - history: lista de dicts com estatísticas por geração
    """
    if fitness_weights is None:
        fitness_weights = {
            "w_recall": 0.40,
            "w_specificity": 0.20,
            "w_f1": 0.25,
            "w_equity": 0.15,
        }

    # População inicial
    population = create_population(model_name, pop_size)
    history = []
    best_overall = None
    best_overall_fit = -1.0

    for gen in range(generations):
        # Avaliar fitness de todos
        fitnesses = [
            fitness(ind, model_name, X, y, ages, **fitness_weights)
            for ind in population
        ]

        # Estatísticas
        gen_best_fit = max(fitnesses)
        gen_avg_fit = np.mean(fitnesses)
        gen_best_idx = int(np.argmax(fitnesses))

        if gen_best_fit > best_overall_fit:
            best_overall_fit = gen_best_fit
            best_overall = copy.deepcopy(population[gen_best_idx])

        history.append({
            "generation": gen + 1,
            "best_fitness": gen_best_fit,
            "avg_fitness": gen_avg_fit,
            "best_params": copy.deepcopy(population[gen_best_idx]),
        })

        if verbose:
            print(
                f"  Geração {gen + 1:>3}/{generations} | "
                f"Melhor: {gen_best_fit:.4f} | "
                f"Média: {gen_avg_fit:.4f}"
            )

        # Elitismo: preservar os N melhores
        elite_indices = sorted(
            range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
        )[:elitism_count]
        new_population = [copy.deepcopy(population[i]) for i in elite_indices]

        # Gerar restante da nova população
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = mutate(c1, model_name, mutation_rate)
            c2 = mutate(c2, model_name, mutation_rate)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population

    return best_overall, best_overall_fit, history
