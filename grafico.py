import pandas as pd
import matplotlib.pyplot as plt

experimentos = [
    {
        "input_filename": "rewards_qlearning_ambiente1.txt",
        "output_filename": "curva_qlearning_ambiente1.png",
        "plot_title": "Q-Learning en Ambiente 1 (Grid Simple)"
    },
    {
        "input_filename": "rewards_sarsa_ambiente1.txt",
        "output_filename": "curva_sarsa_ambiente1.png",
        "plot_title": "SARSA en Ambiente 1 (Grid Simple)"
    },
    {
        "input_filename": "rewards_qlearning_ambiente2.txt",
        "output_filename": "curva_qlearning_ambiente2.png",
        "plot_title": "Q-Learning en Ambiente 2 (Cliff Walking)"
    },
    {
        "input_filename": "rewards_sarsa_ambiente2.txt",
        "output_filename": "curva_sarsa_ambiente2.png",
        "plot_title": "SARSA en Ambiente 2 (Cliff Walking)"
    }
]

for exp in experimentos:
    try:
        data = pd.read_csv(exp["input_filename"], header=None, names=['Episodio', 'Recompensa'])

        plt.figure(figsize=(12, 7))
        plt.plot(data['Episodio'], data['Recompensa'], label='Recompensa Acumulada por Episodio', color='blue', linewidth=1.5)

        plt.title(exp["plot_title"], fontsize=16)
        plt.xlabel('Episodio', fontsize=12)
        plt.ylabel('Recompensa Acumulada', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        if "ambiente1" in exp["input_filename"]:
            plt.ylim(top=110)
        
        plt.savefig(exp["output_filename"])
        
        plt.close()
        
        print(f"Gráfico '{exp['output_filename']}' generado correctamente.")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{exp['input_filename']}'. Asegúrate de que el archivo existe y tiene el nombre correcto.")