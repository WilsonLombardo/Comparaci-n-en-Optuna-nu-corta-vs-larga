---

### 2. `main_analysis.py`
This is the Python script that implements the logic we discussed: loading the data (I put placeholders for your arrays `a` and `rui`) and running the "graphical arsenal" and normality tests.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Configuración de estilo para gráficos académicos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def run_normality_tests(data, var_name):
    """
    Ejecuta un set completo de pruebas de normalidad y genera gráficos diagnósticos.
    """
    print(f"\n--- Análisis Estadístico para: {var_name} ---")
    
    # 1. Tests Numéricos
    # Shapiro-Wilk (mejor para n < 50, pero útil como referencia)
    stat_sh, p_sh = stats.shapiro(data)
    print(f"Test Shapiro-Wilk: Estadístico={stat_sh:.4f}, p-value={p_sh:.4f}")
    
    # Kolmogorov-Smirnov (con corrección Lilliefors implícita si comparamos con normal estandarizada ajustada)
    # Nota: Usamos kstest estandarizando los datos primero
    data_std = (data - np.mean(data)) / np.std(data)
    stat_ks, p_ks = stats.kstest(data_std, 'norm')
    print(f"Test Kolmogorov-Smirnov: Estadístico={stat_ks:.4f}, p-value={p_ks:.4f}")

    interpretation = "Parece Normal (No rechazamos H0)" if p_sh > 0.05 else "No parece Normal (Rechazamos H0)"
    print(f">> Conclusión rápida (Shapiro): {interpretation}")

    # 2. Arsenal Gráfico (Boxplot, Histograma, QQ-Plot)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Diagnóstico de Distribución: {var_name}', fontsize=16)

    # Histograma + KDE
    sns.histplot(data, kde=True, ax=axs[0, 0], color='skyblue')
    axs[0, 0].set_title('Histograma y Densidad (KDE)')

    # Boxplot
    sns.boxplot(x=data, ax=axs[0, 1], color='lightgreen')
    axs[0, 1].set_title('Boxplot (Detección de Outliers)')

    # QQ-Plot
    stats.probplot(data, dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title('Q-Q Plot (Normalidad)')
    axs[1, 0].get_lines()[0].set_color('#1f77b4') # Puntos
    axs[1, 0].get_lines()[1].set_color('red')     # Línea de referencia

    # Plot de la serie (si es temporal)
    axs[1, 1].plot(data, marker='o', linestyle='-', color='purple', alpha=0.7)
    axs[1, 1].set_title('Serie de Tiempo / Secuencia')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar gráfico
    filename = f"analisis_{var_name}.png"
    plt.savefig(filename)
    print(f"Gráfico guardado como: {filename}")
    plt.show()

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CARGA DE DATOS
    # Copia aquí tus arrays 'a' y 'rui' (o 'nu_corta' y 'nu_larga')
    # ---------------------------------------------------------
    
    # Placeholder: Reemplaza esto con tus datos reales exportados del modelo
    # Ejemplo con datos aleatorios para que el script sea funcional al probarlo
    np.random.seed(42)
    
    # Variable 1 (ej. 'a' o 'nu_corta')
    a = np.random.normal(loc=0.5, scale=0.1, size=100) 
    
    # Variable 2 (ej. 'rui' o 'nu_larga') - Simulado algo no normal (ej. lognormal)
    rui = np.random.lognormal(mean=0, sigma=0.5, size=100)

    # EJECUCIÓN
    print("Iniciando análisis de variables del modelo SEIQR...")
    
    # Análisis de la variable 'a'
    run_normality_tests(a, "Variable_A_NuCorta")
    
    # Análisis de la variable 'rui'
    run_normality_tests(rui, "Variable_Rui_NuLarga")

    numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
