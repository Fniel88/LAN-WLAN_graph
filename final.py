# final.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib.table import Table
from matplotlib.patches import Patch

# ------------------------
# 1) DEFINICIÓN DE NODOS, IPS Y GRAFOS
# ------------------------
G = nx.Graph()

labels = {
    "Router": "Router\n192.168.1.1",
    "Switch-LAN": "Switch-LAN",
    "PC-Admin": "PC-Admin\n192.168.1.10",
    "Laptop-Admin": "Laptop-Admin\n192.168.1.11",
    "Impresora": "Impresora\n192.168.1.20",
    "Servidor HTTP": "Servidor HTTP\n192.168.1.100",
    "Access Point": "Access Point\n192.168.2.1",
    "Laptop Profesor": "Laptop Profesor\n192.168.2.10",
    "Tablet": "Tablet\n192.168.2.11",
    "Smartphone": "Smartphone\n192.168.2.12"
}

core_nodes = ["Router", "Switch-LAN"]
lan_nodes = ["PC-Admin", "Laptop-Admin", "Impresora", "Servidor HTTP"]
wlan_nodes = ["Access Point", "Laptop Profesor", "Tablet", "Smartphone"]

# Conexiones LAN (cables UTP)
lan_edges = [
    ("Router", "Switch-LAN"),
    ("Switch-LAN", "PC-Admin"),
    ("Switch-LAN", "Laptop-Admin"),
    ("Switch-LAN", "Impresora"),
    ("Switch-LAN", "Servidor HTTP")
]

# Conexiones WLAN (representadas como enlaces desde el AP; se dibujan después como discontinua)
wlan_edges = [
    ("Access Point", "Laptop Profesor"),
    ("Access Point", "Tablet"),
    ("Access Point", "Smartphone")
]

G.add_nodes_from(labels.keys())
G.add_edges_from(lan_edges)
G.add_edges_from(wlan_edges)

# POSICIONES (ajustadas manualmente para que los WLAN queden cerca del AP)
pos = {
    "Router": (0.0, 0.0),
    "Switch-LAN": (0.0, -0.8),
    "PC-Admin": (-0.8, -1.6),
    "Laptop-Admin": (0.0, -1.6),
    "Impresora": (0.8, -1.6),
    "Servidor HTTP": (-0.2, -0.2),
    "Access Point": (1.6, -0.2),
    "Laptop Profesor": (1.8, 0.5),
    "Tablet": (1.6, -0.9),
    "Smartphone": (2.3, -0.6)
}

# ------------------------
# 2) CODIFICACIÓN MANCHESTER (mensaje -> binario -> tren de pulsos)
# ------------------------
mensaje = "HOLA"
binario = ''.join(format(ord(c), '08b') for c in mensaje)

def manchester(bits):
    # Lista plana: por cada bit, dos muestras [1,-1] o [-1,1]
    samples = []
    for b in bits:
        if b == '1':
            samples.extend([1, -1])
        else:
            samples.extend([-1, 1])
    return np.array(samples, dtype=float)

senal = manchester(binario)

# ------------------------
# 3) FFT (espectro)
# ------------------------
N = len(senal)
yf = np.abs(fft(senal))
# d = 1 (una unidad por muestra; eje relativo de frecuencia)
xf = fftfreq(N, d=1)

# ------------------------
# 4) DIBUJO COMPLETO (topología + señal + espectro + tabla de enrutamiento + leyenda)
# ------------------------

fig = plt.figure(figsize=(20, 12))  # Figura más grande

# ==== NUEVO LAYOUT MEJORADO ====
# 3 filas, 2 columnas para dar espacio a las 4 visualizaciones
grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1.2, 1, 1])

# Topología ocupa las primeras 2 filas de la izquierda
ax1 = fig.add_subplot(grid[0:2, 0])
# Señal Manchester ocupa la primera fila derecha
ax2 = fig.add_subplot(grid[0, 1])
# FFT ocupa la segunda fila derecha (posición mucho mejor)
ax3 = fig.add_subplot(grid[1, 1])
# Tabla ocupa toda la tercera fila
ax4 = fig.add_subplot(grid[2, :])  # Ocupa ambas columnas

ax1.set_title("Topología Avanzada de Red Híbrida", fontsize=14, fontweight="bold")

# Dibujar nodos por grupos con tamaños grandes
nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_color="#e63946", node_size=2500, edgecolors="k", ax=ax1)
nx.draw_networkx_nodes(G, pos, nodelist=lan_nodes, node_color="#1d3557", node_size=2000, edgecolors="k", ax=ax1)
nx.draw_networkx_nodes(G, pos, nodelist=wlan_nodes, node_color="#2a9d8f", node_size=1800, edgecolors="k", ax=ax1)

# Dibujar aristas: LAN (sólidas) y WLAN (discontinuas)
nx.draw_networkx_edges(G, pos, edgelist=lan_edges, edge_color="#457b9d", width=2.2, style='solid', ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=wlan_edges, edge_color="#2b9348", width=2.0, style='dashed', ax=ax1)

# Etiquetas: nombre + IP en dos líneas
for n, (x, y) in pos.items():
    ax1.text(x, y, labels[n], fontsize=9, ha="center", va="center",
             bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.4'),
             color='white')

ax1.axis('off')

# --- Leyenda / índice de colores ---
legend_patches = [
    Patch(facecolor="#e63946", edgecolor="k", label="Nodos Core (Router/Switch)"),
    Patch(facecolor="#1d3557", edgecolor="k", label="LAN (UTP)"),
    Patch(facecolor="#2a9d8f", edgecolor="k", label="WLAN (Wireless)"),
    Patch(facecolor="#457b9d", edgecolor="#457b9d", label="Cable UTP (enlaces LAN)"),
    Patch(facecolor="#2b9348", edgecolor="#2b9348", label="Enlace inalámbrico (WLAN, discontinua)")
]
ax1.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)

# --- B: Señal Manchester (arriba derecha) ---
ax2.step(np.arange(len(senal)), senal, where='mid', linewidth=2)
ax2.set_title("Señal Digital – Codificación Manchester", fontsize=12, fontweight="bold")
ax2.set_xlabel("Tiempo (muestras)")
ax2.set_ylabel("Amplitud")
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle="--", alpha=0.4)

# Marcar bits en el eje (cada par de muestras = 1 bit)
bit_ticks = np.arange(0.5, len(senal)+0.5, 2)
bit_labels = list(binario)
ax2.set_xticks(bit_ticks)
ax2.set_xticklabels(bit_labels, fontsize=8)
ax2.annotate(f"Mensaje: '{mensaje}'  ->  Binario: {binario}", xy=(0.01, -1.3), xycoords='data', fontsize=9)

# --- C: Espectro FFT (centro derecha) - MEJOR UBICADO ---
half = N // 2
ax3.plot(xf[:half], yf[:half], linewidth=2.0, color='purple')
ax3.set_title("Espectro de Frecuencia (FFT)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Frecuencia (unidades relativas)")
ax3.set_ylabel("Magnitud")
ax3.grid(True, linestyle="--", alpha=0.4)
# Añadir área sombreada para mejor visualización
ax3.fill_between(xf[:half], yf[:half], alpha=0.3, color='purple')

# --- D: Tabla de Enrutamiento (abajo, ancha) ---
ax4.axis('off')

# Contenido de la tabla
contenido = [
    ["Red / Máscara", "Gateway", "Interfaz", "Descripción"],
    ["192.168.1.0/24", "192.168.1.1", "LAN", "Segmento Administración"],
    ["192.168.2.0/24", "192.168.2.1", "WLAN", "Segmento Académico"],
    ["0.0.0.0/0", "192.168.1.1", "WAN", "Salida a Internet (NAT)"]
]

# Tabla más ancha (ocupa ambas columnas)
tabla = Table(ax4, bbox=[0.05, 0.1, 0.9, 0.8])

for i, fila in enumerate(contenido):
    for j, celda in enumerate(fila):
        if i == 0:
            facecolor = "#1d3557"
            text_color = "white"
            fontweight = "bold"
            fontsize = 13
        else:
            facecolor = "#f1faee" if i % 2 == 1 else "#e9ecef"
            text_color = "black"
            fontweight = "normal"
            fontsize = 11
        
        cell = tabla.add_cell(i, j, width=0.25, height=0.25, 
                             text=celda, loc="center", 
                             facecolor=facecolor, edgecolor='black')
        cell.get_text().set_fontsize(fontsize)
        cell.get_text().set_color(text_color)
        cell.get_text().set_fontweight(fontweight)

ax4.set_title("Tabla de Enrutamiento del Router", fontsize=14, fontweight="bold", pad=15)
ax4.add_table(tabla)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Guardar resultado como imagen
plt.savefig("topologia_final_mejorada.png", dpi=200, bbox_inches="tight")
plt.show()

print("Imagen guardada como 'topologia_final_mejorada.png' en el directorio actual.")
