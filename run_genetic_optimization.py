"""
Script principal: Otimização via Algoritmo Genético
====================================================
Executa 3 experimentos com configurações distintas do AG,
compara resultados otimizados vs. baselines e gera relatório + gráficos.

Uso:
    python run_genetic_optimization.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from pre_processor.dataset_pre_processor import dataset_pre_processor
from genetic_optimizer.genetic_algorithm import (
    genetic_algorithm,
    build_model,
    fitness,
    _compute_specificity,
    _compute_equity,
    HYPERPARAMETER_SPACE,
)

warnings.filterwarnings("ignore")

# Aplicar estilos caso a fonte exista
try:
    sys.path.insert(0, "./analysis")
    from analysis.styles import apply_styles
    apply_styles()
except Exception:
    pass

# ============================================================
# Configurações dos 3 experimentos
# ============================================================

EXPERIMENTS = [
    {
        "name": "Exp1 — Pop. pequena / Mutação alta",
        "pop_size": 20,
        "generations": 15,
        "crossover_rate": 0.8,
        "mutation_rate": 0.30,
        "tournament_k": 3,
        "elitism_count": 2,
    },
    {
        "name": "Exp2 — Pop. média / Mutação moderada",
        "pop_size": 30,
        "generations": 20,
        "crossover_rate": 0.85,
        "mutation_rate": 0.15,
        "tournament_k": 4,
        "elitism_count": 3,
    },
    {
        "name": "Exp3 — Pop. grande / Mutação baixa",
        "pop_size": 40,
        "generations": 25,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "tournament_k": 5,
        "elitism_count": 4,
    },
]

# Modelos avaliados
MODEL_NAMES = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
]

# Baselines (mesmos hiperparâmetros usados no train_model.py original)
BASELINES = {
    "KNeighborsClassifier": {"n_neighbors": 5},
    "DecisionTreeClassifier": {},
    "LogisticRegression": {"class_weight": "balanced", "max_iter": 1000},
    "RandomForestClassifier": {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "random_state": 42,
    },
}


# ============================================================
# Funções auxiliares de avaliação
# ============================================================

def evaluate_model(model, X_test, y_test, age_test):
    """Retorna dict com recall, specificity, f1, equity."""
    y_pred = model.predict(X_test)
    return {
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": _compute_specificity(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "equity": _compute_equity(y_test, y_pred, age_test),
    }


def _age_groups(ages):
    """Cria faixas etárias para cálculo de equidade."""
    bins = [0, 25, 35, 50, 100]
    labels = ["≤25", "26-35", "36-50", ">50"]
    return pd.cut(ages, bins=bins, labels=labels)


# ============================================================
# Gráficos
# ============================================================

def plot_convergence(all_histories, experiment_name, save_dir):
    """Plota curvas de convergência (melhor e média) de cada modelo."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Convergência — {experiment_name}", fontsize=15)
    axes = axes.flatten()

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[idx]
        hist = all_histories[model_name]
        gens = [h["generation"] for h in hist]
        best = [h["best_fitness"] for h in hist]
        avg = [h["avg_fitness"] for h in hist]

        ax.plot(gens, best, label="Melhor fitness", linewidth=2)
        ax.plot(gens, avg, label="Fitness médio", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(model_name)
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(os.path.join(save_dir, f"convergence_{experiment_name.split('—')[0].strip().replace(' ', '_')}.png"),
                dpi=150)
    plt.close()


def plot_comparison_bars(baseline_metrics, optimized_metrics, experiment_name, save_dir):
    """Gráfico de barras comparativo: baseline vs otimizado para cada métrica."""
    metrics = ["recall", "specificity", "f1", "equity"]
    metric_labels = ["Recall\n(Sensibilidade)", "Especificidade", "F1-Score", "Equidade"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Baseline vs AG Otimizado — {experiment_name}", fontsize=15)
    axes = axes.flatten()

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[idx]
        base_vals = [baseline_metrics[model_name][m] for m in metrics]
        opt_vals = [optimized_metrics[model_name][m] for m in metrics]

        bars1 = ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#7bafd4")
        bars2 = ax.bar(x + width / 2, opt_vals, width, label="AG Otimizado", color="#f4a261")

        ax.set_title(model_name, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Valores sobre as barras
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    plt.savefig(os.path.join(save_dir, f"comparison_{experiment_name.split('—')[0].strip().replace(' ', '_')}.png"),
                dpi=150)
    plt.close()


def plot_experiment_summary(all_exp_results, save_dir):
    """Gráfico resumo comparando o fitness final dos 3 experimentos."""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names = [exp["name"].split("—")[0].strip() for exp in EXPERIMENTS]
    x = np.arange(len(MODEL_NAMES))
    width = 0.25

    for i, exp_name in enumerate(exp_names):
        fitnesses = [all_exp_results[exp_name][m]["best_fitness"] for m in MODEL_NAMES]
        bars = ax.bar(x + i * width, fitnesses, width, label=exp_name)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Comparação de Fitness — 3 Experimentos", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODEL_NAMES, fontsize=9)
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(save_dir, "experiment_summary.png"), dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs("./graphs", exist_ok=True)

    # 1. Carregar e pré-processar dados
    print("=" * 60)
    print("  OTIMIZAÇÃO VIA ALGORITMO GENÉTICO")
    print("  Diagnóstico de Diabetes — Saúde da Mulher")
    print("=" * 60)

    df_raw = pd.read_csv("./dataset/diabetes.csv")
    df_normalized, imputer, scaler = dataset_pre_processor(df_raw)

    X = df_normalized.drop("Outcome", axis=1)
    y = df_normalized["Outcome"]

    # Usar coluna Age original (antes do scaling) para grupos de equidade
    ages_raw = df_raw["Age"]
    age_groups = _age_groups(ages_raw)

    # Split fixo para avaliação final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    age_train = age_groups.iloc[X_train.index]
    age_test = age_groups.iloc[X_test.index]

    # 2. Treinar baselines e avaliar
    print("\n--- Avaliando modelos BASELINE (sem otimização) ---")
    baseline_metrics = {}
    for model_name in MODEL_NAMES:
        params = BASELINES[model_name]
        model = build_model(model_name, params)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, age_test)
        baseline_metrics[model_name] = metrics
        print(f"  {model_name:30s} | Recall={metrics['recall']:.4f} | "
              f"Spec={metrics['specificity']:.4f} | F1={metrics['f1']:.4f} | "
              f"Equity={metrics['equity']:.4f}")

    # 3. Executar os 3 experimentos
    all_exp_results = {}  # { exp_short_name : { model_name: { best_fitness, best_params, metrics, history } } }

    for exp in EXPERIMENTS:
        exp_short = exp["name"].split("—")[0].strip()
        print(f"\n{'=' * 60}")
        print(f"  {exp['name']}")
        print(f"  Pop={exp['pop_size']} | Ger={exp['generations']} | "
              f"Cross={exp['crossover_rate']} | Mut={exp['mutation_rate']} | "
              f"Torneio k={exp['tournament_k']}")
        print(f"{'=' * 60}")

        exp_results = {}
        exp_histories = {}

        for model_name in MODEL_NAMES:
            print(f"\n  >> Otimizando {model_name} ...")
            best_params, best_fit, history = genetic_algorithm(
                model_name=model_name,
                X=X_train,
                y=y_train,
                ages=age_train,
                pop_size=exp["pop_size"],
                generations=exp["generations"],
                crossover_rate=exp["crossover_rate"],
                mutation_rate=exp["mutation_rate"],
                tournament_k=exp["tournament_k"],
                elitism_count=exp["elitism_count"],
                verbose=True,
            )

            # Treinar modelo final com melhores parâmetros em todo X_train
            final_model = build_model(model_name, best_params)
            final_model.fit(X_train, y_train)
            metrics = evaluate_model(final_model, X_test, y_test, age_test)

            exp_results[model_name] = {
                "best_fitness": best_fit,
                "best_params": best_params,
                "metrics": metrics,
            }
            exp_histories[model_name] = history

            print(f"     Melhor fitness CV: {best_fit:.4f}")
            print(f"     Teste -> Recall={metrics['recall']:.4f} | "
                  f"Spec={metrics['specificity']:.4f} | F1={metrics['f1']:.4f} | "
                  f"Equity={metrics['equity']:.4f}")
            print(f"     Hiperparâmetros: {best_params}")

        all_exp_results[exp_short] = exp_results

        # Gráficos do experimento
        opt_metrics = {m: exp_results[m]["metrics"] for m in MODEL_NAMES}
        plot_convergence(exp_histories, exp["name"], "./graphs")
        plot_comparison_bars(baseline_metrics, opt_metrics, exp["name"], "./graphs")

    # 4. Gráfico-resumo dos 3 experimentos
    plot_experiment_summary(all_exp_results, "./graphs")

    # 5. Relatório final em texto
    print_final_report(baseline_metrics, all_exp_results)


def print_final_report(baseline_metrics, all_exp_results):
    """Imprime o relatório completo de comparação."""
    print("\n" + "=" * 80)
    print("  RELATÓRIO FINAL — AG vs BASELINE")
    print("=" * 80)

    for model_name in MODEL_NAMES:
        print(f"\n{'─' * 70}")
        print(f"  Modelo: {model_name}")
        print(f"{'─' * 70}")

        base = baseline_metrics[model_name]
        print(f"  {'Configuração':<30s} | {'Recall':>7s} | {'Spec':>7s} | {'F1':>7s} | {'Equity':>7s}")
        print(f"  {'-' * 30}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
        print(f"  {'BASELINE':<30s} | {base['recall']:>7.4f} | {base['specificity']:>7.4f} | "
              f"{base['f1']:>7.4f} | {base['equity']:>7.4f}")

        for exp_short, results in all_exp_results.items():
            m = results[model_name]["metrics"]
            delta_recall = m["recall"] - base["recall"]
            print(f"  {exp_short:<30s} | {m['recall']:>7.4f} | {m['specificity']:>7.4f} | "
                  f"{m['f1']:>7.4f} | {m['equity']:>7.4f}  (Δrecall={delta_recall:+.4f})")

        # Melhor experimento para este modelo
        best_exp = max(
            all_exp_results.keys(),
            key=lambda e: (
                0.40 * all_exp_results[e][model_name]["metrics"]["recall"]
                + 0.20 * all_exp_results[e][model_name]["metrics"]["specificity"]
                + 0.25 * all_exp_results[e][model_name]["metrics"]["f1"]
                + 0.15 * all_exp_results[e][model_name]["metrics"]["equity"]
            ),
        )
        best_m = all_exp_results[best_exp][model_name]
        print(f"\n  ★ Melhor config: {best_exp}")
        print(f"    Hiperparâmetros otimizados: {best_m['best_params']}")

    print(f"\n{'=' * 80}")
    print("  Gráficos salvos em ./graphs/")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
