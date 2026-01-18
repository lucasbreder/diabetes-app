# Arquivo: estilo.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def apply_styles():
    # --- 1. Configuração da Fonte ---
    caminho_fonte = './fonts/DMSans-Medium.ttf'
    
    # Verifica se a fonte existe antes de tentar carregar para não dar erro
    if os.path.exists(caminho_fonte):
        fm.fontManager.addfont(caminho_fonte)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DM Sans']
    else:
        print(f"Aviso: Fonte não encontrada em {caminho_fonte}. Usando fonte padrão.")

    # --- 2. Cores ---
    plt.rcParams['text.color'] = '#4d4d4d'
    plt.rcParams['axes.labelcolor'] = '#000'
    plt.rcParams['xtick.color'] = '#4d4d4d'
    plt.rcParams['ytick.color'] = '#4d4d4d'

    # --- 3. Tamanhos ---
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    # --- 4. Bordas e Layout ---
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.edgecolor'] = '#888888'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['patch.edgecolor'] = 'white'
    plt.rcParams['patch.force_edgecolor'] = True
    
    plt.rcParams['figure.constrained_layout.use'] = True